#!/usr/bin/env python3
"""
Weekly Literature Watcher: LLMs in University Didactics
Searches arXiv and OpenAlex for new papers on LLMs in education.
Produces a curated Markdown digest and imports items to Zotero with PDFs.
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Set
import re
import sqlite3

import arxiv
import requests
from pyzotero import zotero
from sqlite_utils import Database
from unidecode import unidecode


# ==================== Configuration ====================

# Environment variables (secrets)
ZOTERO_LIBRARY_ID = os.environ.get("ZOTERO_LIBRARY_ID", "")
ZOTERO_API_KEY = os.environ.get("ZOTERO_API_KEY", "")
OPEN_ALEX_API_KEY = os.environ.get("OPEN_ALEX_API_KEY", "")

# Contact email for OpenAlex API (polite pool) - optional, non-secret
OPENALEX_MAILTO = os.environ.get("OPENALEX_MAILTO", "github-bot@users.noreply.github.com")

# File paths
JOURNALS_CONFIG_PATH = Path(__file__).parent / "journals.json"
STATE_PATH = Path(__file__).parent / "state.json"
DIGEST_PATH = Path(__file__).parent / "digest.md"
DB_PATH = Path(__file__).parent / "literature.db"
REFERENCES_PATH = Path(__file__).parent / "references.json"

# API defaults
ARXIV_MAX_RESULTS = 100
OPENALEX_LOOKBACK_DAYS = 90  # Increased from 10 to 90 for better coverage
OPENALEX_MAX_RESULTS_PER_JOURNAL = 200  # Increased for pagination
MAX_NEW_ITEMS_TOTAL = 50

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
ARXIV_POLITENESS_DELAY = 3  # seconds between arXiv API calls

# Education term definitions for filtering and scoring
# FIX 3: Split into setting terms (strong university/classroom context) vs ambiguous terms

# EDU_SETTING_TERMS: Strong university/classroom context - must match at least one for gate
EDU_SETTING_TERMS = [
    "higher education", "university", "college", "undergraduate", "graduate",
    "postsecondary", "course", "classroom", "lecture", "seminar",
    "assignment", "grading", "rubric", "learning outcomes", "learning objectives",
    "instructional design", "pedagogy", "pedagogical", "didactic", "didactics",
    "teaching", "tutoring", "tutor", "instructor", "edtech", "mooc"
]

# EDU_AMBIGUOUS_TERMS: Weak signals (also used in ML contexts) - NOT sufficient alone
EDU_AMBIGUOUS_TERMS = [
    "student", "teacher", "curriculum", "assessment", "feedback", "training",
    "learner", "education", "educational", "instruction", "instructional"
]

# Education-specific "learning" phrases (contextual, not bare "learning")
EDU_LEARNING_PHRASES = [
    "learning outcomes", "learning analytics", "learning experience",
    "learning environment", "learning platform", "e-learning",
    "learning management system", "learning design", "learning objectives"
]

# NEGATIVE_STRONG_TERMS: System/optimization/robotics indicators - exclude unless EDU_SETTING present
NEGATIVE_STRONG_TERMS = [
    "quantization", "optimizer", "gpu", "cuda", "scheduling", "kv cache",
    "inference speedup", "ssd offload", "benchmark", "benchmarking", "embodied",
    "robot", "robotics", "motion generation", "text2sql", "ner", "named entity recognition",
    "speech recognition", "world model", "compiler", "kernel", "throughput",
    "memory bandwidth", "hardware acceleration", "sparse attention", "model compression"
]


# ==================== Data Models ====================

@dataclass
class Paper:
    """Normalized paper representation from any source."""
    identifier: str  # Unique ID: arxiv:<id> or openalex:<id> or doi:<doi>
    title: str
    authors: List[str]
    abstract: str
    url: str  # Landing page
    pdf_url: Optional[str] = None
    doi: Optional[str] = None
    publication_date: Optional[str] = None
    journal_name: Optional[str] = None
    source_type: str = "unknown"  # arxiv, openalex
    source_metadata: Dict = field(default_factory=dict)
    relevance_score: float = 0.0
    education_intent_pass: bool = False
    pdf_downloaded: bool = False
    pdf_path: Optional[Path] = None
    attachment_missing: bool = False
    citekey: Optional[str] = None
    citekey_status: str = "pending"  # "pending"|"synced"|"missing"
    citekey_synced_utc: Optional[str] = None
    zotero_item_key: Optional[str] = None
    zotero_item_version: Optional[int] = None
    zotero_attachment_key: Optional[str] = None
    zotero_collection_key: Optional[str] = None
    zotero_collection_name: Optional[str] = None
    import_status: Optional[str] = None
    last_error: Optional[str] = None


@dataclass
class State:
    """Persistent state for deduplication."""
    seen_arxiv_ids: Set[str] = field(default_factory=set)
    seen_openalex_ids: Set[str] = field(default_factory=set)
    seen_dois: Set[str] = field(default_factory=set)
    resolved_source_ids: Dict[str, str] = field(default_factory=dict)  # journal name -> OpenAlex source ID
    last_run: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "seen_arxiv_ids": list(self.seen_arxiv_ids),
            "seen_openalex_ids": list(self.seen_openalex_ids),
            "seen_dois": list(self.seen_dois),
            "resolved_source_ids": self.resolved_source_ids,
            "last_run": self.last_run,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "State":
        return cls(
            seen_arxiv_ids=set(data.get("seen_arxiv_ids", [])),
            seen_openalex_ids=set(data.get("seen_openalex_ids", [])),
            seen_dois=set(data.get("seen_dois", [])),
            resolved_source_ids=data.get("resolved_source_ids", {}),
            last_run=data.get("last_run"),
        )


# ==================== Utility Functions ====================

def load_config() -> Dict:
    """Load journals configuration from JSON file."""
    with open(JOURNALS_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_state() -> State:
    """Load state from JSON file or create new state."""
    if STATE_PATH.exists():
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return State.from_dict(data)
    return State()


def save_state(state: State):
    """Save state to JSON file."""
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, indent=2)
    print(f"State saved to {STATE_PATH}")


def retry_with_backoff(func, *args, **kwargs):
    """Execute function with exponential backoff retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait_time = RETRY_DELAY * (2 ** attempt)
            print(f"Error: {e}. Retrying in {wait_time}s... (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait_time)


def normalize_text(text: str) -> str:
    """Normalize text for matching: lowercase, remove extra whitespace."""
    return re.sub(r'\s+', ' ', text.lower()).strip()


def check_hard_education_intent(paper: Paper, config: Dict) -> Dict:
    """
    FIX 3: Hard education-intent gate with explainability.
    Returns dict with: gate_pass (bool), matched_llm, matched_edu_setting, matched_edu_ambiguous,
                       matched_negative, exclude_reason (str or None)
    
    Gate logic:
    1. Must match at least one LLM term (from config)
    2. Must match at least one EDU_SETTING_TERM (strong university/classroom context)
    3. EDU_AMBIGUOUS_TERMS alone are NOT sufficient
    4. Disambiguation exclusions for curriculum-learning, feedback-loop, distillation contexts
    5. NEGATIVE_STRONG_TERMS exclude unless EDU_SETTING_TERM present
    """
    default_rules = config.get("default_rules", {})
    llm_terms = [normalize_text(t) for t in default_rules.get("llm_terms", [])]
    
    edu_setting_norm = [normalize_text(t) for t in EDU_SETTING_TERMS]
    edu_ambiguous_norm = [normalize_text(t) for t in EDU_AMBIGUOUS_TERMS]
    negative_terms_norm = [normalize_text(t) for t in NEGATIVE_STRONG_TERMS]
    
    title_norm = normalize_text(paper.title)
    abstract_norm = normalize_text(paper.abstract)
    full_text = f"{title_norm} {abstract_norm}"
    
    # Track matches for explainability
    matched_llm = [t for t in llm_terms if t in full_text]
    matched_edu_setting = [t for t in edu_setting_norm if t in full_text]
    matched_edu_ambiguous = [t for t in edu_ambiguous_norm if t in full_text]
    matched_negative = [t for t in negative_terms_norm if t in full_text]
    
    result = {
        "gate_pass": False,
        "matched_llm_terms": matched_llm,
        "matched_edu_setting_terms": matched_edu_setting,
        "matched_edu_ambiguous_terms": matched_edu_ambiguous,
        "matched_negative_terms": matched_negative,
        "exclude_reason": None
    }
    
    # Rule 1: Must have LLM term
    if not matched_llm:
        result["exclude_reason"] = "no_llm_term"
        return result
    
    # Rule 2: Must have at least one EDU_SETTING_TERM for strong context
    if not matched_edu_setting:
        result["exclude_reason"] = "no_edu_setting_term"
        return result
    
    # Rule 3: Disambiguation - curriculum learning (ML technique)
    if "curriculum learning" in full_text:
        result["exclude_reason"] = "curriculum_learning_ml_context"
        return result
    if "curriculum" in full_text and any(x in full_text for x in ["augmentation", "dataset", "training strategy", "similarity curriculum"]):
        if not any(x in full_text for x in ["course", "classroom", "university", "higher education"]):
            result["exclude_reason"] = "curriculum_ml_context"
            return result
    
    # Rule 4: Disambiguation - feedback loop (algorithmic/RL)
    feedback_algo_terms = ["feedback loop", "perceptual feedback", "execution feedback", "reward", "reinforcement learning"]
    if any(x in full_text for x in feedback_algo_terms):
        if not any(x in full_text for x in ["course", "classroom", "student", "grading", "assessment"]):
            result["exclude_reason"] = "feedback_algorithmic_context"
            return result
    
    # Rule 5: Disambiguation - distillation (model compression)
    if "distillation" in full_text or "teacher-student" in full_text:
        if not any(x in full_text for x in ["classroom", "tutoring", "course", "university", "higher education", "grading"]):
            result["exclude_reason"] = "distillation_model_compression"
            return result
    
    # Rule 6: Negative strong terms (system/optimization) - exclude unless strong EDU_SETTING
    if matched_negative:
        # Check if we have very strong education context
        very_strong_edu = any(x in full_text for x in ["university course", "classroom", "higher education", "assignment", "grading", "rubric"])
        if not very_strong_edu:
            result["exclude_reason"] = f"system_optimization_context: {matched_negative[0]}"
            return result
    
    # All checks passed
    result["gate_pass"] = True
    return result


def calculate_relevance_score(paper: Paper, config: Dict, journal_config: Optional[Dict] = None) -> float:
    """
    FIX 3: Calculate relevance score based on keyword matches in title and abstract.
    Returns a deterministic, explainable score.
    Updated to use EDU_SETTING_TERMS and EDU_AMBIGUOUS_TERMS.
    """
    default_rules = config.get("default_rules", {})
    scoring = default_rules.get("relevance_scoring", {})
    weights = scoring.get("weights", {})
    
    # Override with journal-specific rules if provided
    if journal_config and "rules_override" in journal_config:
        override_scoring = journal_config["rules_override"].get("relevance_scoring", {})
        if override_scoring:
            # Merge weights
            weights = {**weights, **override_scoring.get("weights", {})}
    
    title_norm = normalize_text(paper.title)
    abstract_norm = normalize_text(paper.abstract)
    
    score = 0.0
    
    # LLM terms
    llm_terms = [normalize_text(t) for t in default_rules.get("llm_terms", [])]
    for term in llm_terms:
        if term in title_norm:
            score += weights.get("llm_term_in_title", 6)
            break  # Count once per category
    for term in llm_terms:
        if term in abstract_norm:
            score += weights.get("llm_term_in_abstract", 3)
            break
    
    # FIX 3: Strong education setting terms (university/classroom context)
    edu_setting_norm = [normalize_text(t) for t in EDU_SETTING_TERMS]
    
    for term in edu_setting_norm:
        if term in title_norm:
            score += weights.get("education_term_in_title", 4)
            break
    for term in edu_setting_norm:
        if term in abstract_norm:
            score += weights.get("education_term_in_abstract", 2)
            break
    
    # FIX 3: Weak education terms (bonus if present, but not primary signal)
    edu_ambiguous_norm = [normalize_text(t) for t in EDU_AMBIGUOUS_TERMS]
    
    for term in edu_ambiguous_norm:
        if term in title_norm or term in abstract_norm:
            score += 1  # Small bonus
            break
    
    # Bonus phrases in title
    bonus_phrases = [normalize_text(p) for p in scoring.get("bonus_phrases_title", [])]
    for phrase in bonus_phrases:
        if phrase in title_norm:
            score += weights.get("bonus_phrase_in_title", 2)
            break
    
    # FIX 3: Enhanced penalty phrases (system/optimization papers)
    penalty_phrases = [normalize_text(p) for p in scoring.get("penalty_phrases_anywhere", [])]
    penalty_phrases.extend([normalize_text(t) for t in NEGATIVE_STRONG_TERMS])
    for phrase in penalty_phrases:
        if phrase in title_norm or phrase in abstract_norm:
            score += weights.get("penalty_obvious_non_education", -6)
            break
    
    return score


def passes_min_score(score: float, config: Dict, journal_config: Optional[Dict] = None) -> bool:
    """Check if score meets minimum threshold."""
    default_min = config.get("default_rules", {}).get("relevance_scoring", {}).get("min_score_default", 8)
    
    if journal_config and "rules_override" in journal_config:
        override_scoring = journal_config["rules_override"].get("relevance_scoring", {})
        min_score = override_scoring.get("min_score", default_min)
    else:
        min_score = default_min
    
    return score >= min_score


def check_education_intent_terms(paper: Paper, journal_config: Dict) -> bool:
    """
    For Tier B practice journals that require education intent,
    check if at least one education intent term is present in title/abstract.
    """
    if not journal_config.get("rules_override", {}).get("require_education_intent_terms", False):
        return True
    
    intent_terms = journal_config["rules_override"].get("education_intent_terms", [])
    intent_terms_norm = [normalize_text(t) for t in intent_terms]
    
    title_norm = normalize_text(paper.title)
    abstract_norm = normalize_text(paper.abstract)
    
    for term in intent_terms_norm:
        if term in title_norm or term in abstract_norm:
            return True
    
    return False


# ==================== arXiv Pipeline ====================

def search_arxiv(config: Dict, state: State) -> tuple[List[Paper], Dict]:
    """
    FIX 3: Search arXiv for papers on LLMs in education with enhanced precision gating.
    Returns (list of Paper objects, statistics dict with gating details).
    Query now requires EDU_SETTING_TERMS (not just ambiguous).
    """
    print("\n=== Searching arXiv ===")
    
    default_rules = config.get("default_rules", {})
    llm_terms = default_rules.get("llm_terms", [])
    
    # FIX 3: Use setting terms for query (not ambiguous-only)
    edu_setting_terms = EDU_SETTING_TERMS
    
    # Build search query: (LLM terms) AND (education setting terms)
    llm_query_parts = [f'"{term}"' for term in llm_terms[:6]]  # Use top LLM terms
    llm_query = " OR ".join(llm_query_parts)
    
    edu_query_parts = [f'"{term}"' for term in edu_setting_terms[:12]]  # Use top setting terms
    edu_query = " OR ".join(edu_query_parts)
    
    # Combined query with AND operator
    query = f"({llm_query}) AND ({edu_query})"
    
    print(f"arXiv query structure: (LLM terms) AND (EDU_SETTING terms)")
    print(f"LLM terms: {', '.join(llm_terms[:6])}")
    print(f"EDU_SETTING terms: {', '.join(edu_setting_terms[:12])}")
    print(f"Max results: {ARXIV_MAX_RESULTS}")
    
    search = arxiv.Search(
        query=query,
        max_results=ARXIV_MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    
    papers = []
    fetched_count = 0
    stats = {
        "fetched": 0,
        "excluded_by_gate": 0,
        "excluded_by_disambiguation": 0,
        "excluded_by_negative_terms": 0,
        "passed": 0
    }
    
    try:
        results = list(search.results())
        fetched_count = len(results)
        stats["fetched"] = fetched_count
        print(f"Retrieved {fetched_count} results from arXiv")
        
        for result in results:
            arxiv_id = result.entry_id.split("/")[-1].replace("v", "v")  # Keep version
            # Strip version for dedup check
            arxiv_id_no_version = re.sub(r'v\d+$', '', arxiv_id)
            identifier = f"arxiv:{arxiv_id_no_version}"
            
            # Skip if already seen
            if arxiv_id in state.seen_arxiv_ids or arxiv_id_no_version in state.seen_arxiv_ids:
                continue
            
            # Extract authors
            authors = [author.name for author in result.authors]
            
            # Build Paper object
            paper = Paper(
                identifier=identifier,
                title=result.title,
                authors=authors,
                abstract=result.summary,
                url=result.entry_id,
                pdf_url=result.pdf_url,
                doi=result.doi,
                publication_date=result.published.isoformat() if result.published else None,
                source_type="arxiv",
                source_metadata={"arxiv_id": arxiv_id_no_version},
            )
            
            # FIX 3: Apply enhanced education-intent gate with explainability
            gate_result = check_hard_education_intent(paper, config)
            
            if not gate_result["gate_pass"]:
                # Track reason for exclusion
                reason = gate_result["exclude_reason"]
                if reason == "no_edu_setting_term":
                    stats["excluded_by_gate"] += 1
                elif reason and ("curriculum" in reason or "feedback" in reason or "distillation" in reason):
                    stats["excluded_by_disambiguation"] += 1
                elif reason and "system_optimization" in reason:
                    stats["excluded_by_negative_terms"] += 1
                else:
                    stats["excluded_by_gate"] += 1
                continue
            
            paper.education_intent_pass = True
            # Store explainability in source_metadata
            paper.source_metadata.update({
                "gate_result": gate_result,
            })
            
            # Calculate preliminary score for logging
            paper.relevance_score = calculate_relevance_score(paper, config)
            
            papers.append(paper)
            stats["passed"] += 1
            
        time.sleep(ARXIV_POLITENESS_DELAY)  # Politeness delay
        
    except Exception as e:
        print(f"Error searching arXiv: {e}")
    
    # Sort by score for better logging
    papers.sort(key=lambda p: p.relevance_score, reverse=True)
    
    print(f"Fetched: {stats['fetched']}, Excluded by gate: {stats['excluded_by_gate']}, "
          f"Excluded by disambiguation: {stats['excluded_by_disambiguation']}, "
          f"Excluded by negative terms: {stats['excluded_by_negative_terms']}, "
          f"Passed: {stats['passed']}")
    if papers:
        print(f"Top 5 titles with education intent and scores:")
        for i, paper in enumerate(papers[:5], 1):
            print(f"  {i}. [{paper.relevance_score:.1f}] {paper.title[:70]}...")
    
    return papers, stats


# ==================== OpenAlex Pipeline ====================

def resolve_openalex_source_id(journal_name: str, issn: List[str], state: State) -> Optional[Dict]:
    """
    Resolve journal name/ISSN to OpenAlex source ID and verify it's a journal.
    Uses cached values from state if available.
    Returns dict with source_id and source_type, or None if resolution fails.
    """
    # Check cache first
    if journal_name in state.resolved_source_ids:
        cached_id = state.resolved_source_ids[journal_name]
        print(f"  Using cached source ID for '{journal_name}': {cached_id}")
        return {"source_id": cached_id, "source_type": "journal"}  # Assume cached ones are verified
    
    print(f"  Resolving OpenAlex source ID for: {journal_name}")
    
    # Try ISSN first if available
    if issn:
        for issn_value in issn:
            try:
                url = f"https://api.openalex.org/sources?filter=issn:{issn_value}"
                headers = {"User-Agent": f"mailto:{OPENALEX_MAILTO}"}
                if OPEN_ALEX_API_KEY:
                    url += f"&api_key={OPEN_ALEX_API_KEY}"
                
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if data.get("results") and len(data["results"]) > 0:
                    result = data["results"][0]
                    source_id = result["id"].split("/")[-1]
                    source_type = result.get("type", "unknown")
                    
                    # Verify it's a journal
                    if source_type != "journal":
                        print(f"  Warning: Resolved source via ISSN {issn_value} is type '{source_type}', not 'journal'. Skipping.")
                        continue
                    
                    print(f"  Resolved via ISSN {issn_value}: {source_id} (type: {source_type})")
                    state.resolved_source_ids[journal_name] = source_id
                    return {"source_id": source_id, "source_type": source_type}
            except Exception as e:
                print(f"  Error resolving via ISSN {issn_value}: {e}")
    
    # Try by name
    try:
        url = f"https://api.openalex.org/sources?search={requests.utils.quote(journal_name)}"
        headers = {"User-Agent": f"mailto:{OPENALEX_MAILTO}"}
        if OPEN_ALEX_API_KEY:
            url += f"&api_key={OPEN_ALEX_API_KEY}"
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("results") and len(data["results"]) > 0:
            # Try to match by name similarity
            for result in data["results"]:
                result_name = result.get("display_name", "")
                source_type = result.get("type", "unknown")
                
                # Only consider journal-type sources
                if source_type != "journal":
                    continue
                
                if normalize_text(result_name) == normalize_text(journal_name):
                    source_id = result["id"].split("/")[-1]
                    print(f"  Resolved via name: {source_id} (type: {source_type})")
                    state.resolved_source_ids[journal_name] = source_id
                    return {"source_id": source_id, "source_type": source_type}
            
            # Use first journal result as fallback
            for result in data["results"]:
                source_type = result.get("type", "unknown")
                if source_type == "journal":
                    source_id = result["id"].split("/")[-1]
                    print(f"  Resolved via name (best journal match): {source_id} (type: {source_type})")
                    state.resolved_source_ids[journal_name] = source_id
                    return {"source_id": source_id, "source_type": source_type}
    except Exception as e:
        print(f"  Error resolving via name: {e}")
    
    print(f"  Could not resolve source ID for {journal_name}")
    return None


def search_openalex_journal(journal: Dict, config: Dict, state: State, lookback_days: int) -> Optional[List[Paper]]:
    """
    FIX 2: Search OpenAlex for papers in a specific journal with detailed logging.
    Returns list of Paper objects, or None if journal search failed.
    Uses journal:<SOURCE_ID> filter (NOT primary_location.source.id) for correct results.
    """
    journal_name = journal["name"]
    print(f"\n  Searching: {journal_name}")
    
    # Resolve source ID with type verification
    issn_list = journal.get("identifiers", {}).get("issn", [])
    source_info = resolve_openalex_source_id(journal_name, issn_list, state)
    
    if not source_info or not source_info.get("source_id"):
        print(f"  ❌ Skipping {journal_name}: could not resolve source ID")
        return None  # Signal failure
    
    source_id = source_info["source_id"]
    source_type = source_info.get("source_type", "unknown")
    print(f"  ✓ Resolved source ID: {source_id}, Type: {source_type}")
    
    # Build search query with LLM and education terms (matching arXiv logic)
    default_rules = config.get("default_rules", {})
    llm_terms = default_rules.get("llm_terms", [])
    
    # Use module-level education term constants (combine setting + ambiguous for OpenAlex)
    edu_terms = EDU_SETTING_TERMS[:6] + EDU_AMBIGUOUS_TERMS[:4]
    
    # Combine terms for search
    search_terms = llm_terms[:4] + edu_terms
    search_query = " OR ".join(search_terms)
    
    # Date filter
    from_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    print(f"  📅 Publication date threshold: >{from_date} ({lookback_days} days lookback)")
    
    # Build filter - CRITICAL: use journal:<SOURCE_ID> (NOT primary_location.source.id)
    filters = f"journal:{source_id},publication_date:>{from_date}"
    print(f"  🔍 Filter: {filters}")
    
    # Build URL
    base_url = "https://api.openalex.org/works"
    params = {
        "filter": filters,
        "search": search_query,
        "sort": "publication_date:desc",
        "per-page": min(OPENALEX_MAX_RESULTS_PER_JOURNAL, 200),
    }
    
    url = f"{base_url}?filter={filters}&search={requests.utils.quote(search_query)}&sort=publication_date:desc&per-page={params['per-page']}"
    
    headers = {"User-Agent": f"mailto:{OPENALEX_MAILTO}"}
    if OPEN_ALEX_API_KEY:
        url += f"&api_key={OPEN_ALEX_API_KEY}"
    
    # FIX 2: Log request URL (without API key for security)
    url_no_key = url.replace(f"&api_key={OPEN_ALEX_API_KEY}", "&api_key=***") if OPEN_ALEX_API_KEY else url
    print(f"  🌐 Request URL: {url_no_key}")
    
    papers = []
    try:
        response = retry_with_backoff(requests.get, url, headers=headers, timeout=30)
        
        # FIX 2: Log HTTP status
        print(f"  📡 HTTP status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"  ⚠️  WARNING: Non-200 status for {journal_name}: {response.status_code}")
            print(f"  Response snippet: {response.text[:200]}")
            return []  # Continue to next journal
        
        response.raise_for_status()
        data = response.json()
        
        meta = data.get("meta", {})
        total_count = meta.get("count", 0)
        results = data.get("results", [])
        
        # FIX 2: Enhanced logging for 0 results
        print(f"  📊 Total matching works: {total_count}, Retrieved in this page: {len(results)}")
        
        if total_count == 0:
            print(f"  ⚠️  WARNING: 0 results for {journal_name}")
            print(f"  📋 Response meta: {meta}")
            print(f"  💡 Troubleshooting:")
            print(f"     - Verify filter uses 'journal:{source_id}' (NOT 'primary_location.source.id')")
            print(f"     - Check publication_date range ({lookback_days} days may not have new papers)")
            print(f"     - Verify search terms match paper topics in this journal")
        
        for work in results:
            openalex_id = work["id"].split("/")[-1]
            identifier = f"openalex:{openalex_id}"
            
            # Skip if already seen
            if openalex_id in state.seen_openalex_ids:
                continue
            
            doi_raw = work.get("doi", "")
            doi = doi_raw.replace("https://doi.org/", "") if doi_raw else None
            if doi and doi in state.seen_dois:
                continue
            
            # Extract metadata
            title = work.get("title", "Untitled")
            authors = []
            for authorship in work.get("authorships", []):
                author = authorship.get("author", {})
                if author and author.get("display_name"):
                    authors.append(author["display_name"])
            
            abstract = work.get("abstract_inverted_index")
            if abstract:
                # Reconstruct abstract from inverted index
                abstract_text = reconstruct_abstract_from_inverted_index(abstract)
            else:
                abstract_text = ""
            
            # Get URLs
            landing_page = work.get("doi", work.get("id", ""))
            if landing_page.startswith("http"):
                url = landing_page
            else:
                url = f"https://doi.org/{landing_page}" if landing_page else work.get("id", "")
            
            # Try to get PDF URL from open access location
            pdf_url = None
            best_oa = work.get("best_oa_location")
            if best_oa and best_oa.get("pdf_url"):
                pdf_url = best_oa["pdf_url"]
            
            pub_date = work.get("publication_date")
            
            paper = Paper(
                identifier=identifier,
                title=title,
                authors=authors,
                abstract=abstract_text,
                url=url,
                pdf_url=pdf_url,
                doi=doi,
                publication_date=pub_date,
                journal_name=journal_name,
                source_type="openalex",
                source_metadata={
                    "openalex_id": openalex_id,
                    "journal_tier": journal.get("tier", "unknown"),
                },
            )
            
            # Apply education-intent check (use enhanced gate for consistency)
            gate_result = check_hard_education_intent(paper, config)
            paper.education_intent_pass = gate_result["gate_pass"]
            
            papers.append(paper)
        
        # TODO: Implement pagination if needed (using cursor)
        # For now, we rely on per-page=200 to get enough results
        
        print(f"  ✓ Collected {len(papers)} new papers from {journal_name}")
        return papers
        
    except Exception as e:
        # FIX 2: Enhanced error logging
        print(f"  ❌ Error searching OpenAlex for {journal_name}: {e}")
        import traceback
        print(f"  Traceback: {traceback.format_exc()[:300]}")
        return []  # Continue to next journal


def reconstruct_abstract_from_inverted_index(inverted_index: Dict) -> str:
    """Reconstruct abstract text from OpenAlex inverted index."""
    if not inverted_index:
        return ""
    
    # Create a list to hold words at their positions
    # Filter out empty position lists to avoid crashes
    all_positions = [pos for positions in inverted_index.values() if positions for pos in positions]
    if not all_positions:
        return ""
    
    max_pos = max(all_positions)
    words = [""] * (max_pos + 1)
    
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
    
    return " ".join(words).strip()


def search_openalex(config: Dict, state: State) -> tuple[List[Paper], Dict]:
    """
    FIX 2: Search OpenAlex for papers in whitelisted journals with enhanced diagnostics.
    Returns (list of Paper objects, statistics dict).
    """
    print("\n=== Searching OpenAlex ===")
    
    # FIX 2: Diagnostic logging at start
    print(f"OPEN_ALEX_API_KEY present: {'yes' if OPEN_ALEX_API_KEY else 'no'}")
    
    journals = config.get("journals", [])
    print(f"Loaded {len(journals)} journals from journals.json")
    
    if not journals:
        print("⚠️  OpenAlex skipped: reason=no journals loaded from journals.json")
        return [], {"executed": False, "skip_reason": "no_journals", "fetched": 0, "curated": 0}
    
    if not OPEN_ALEX_API_KEY:
        print("⚠️  WARNING: OpenAlex API key not configured - requests may be rate-limited")
    
    lookback_days = config.get("default_rules", {}).get("lookback_days", OPENALEX_LOOKBACK_DAYS)
    from_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    print(f"Lookback window: {lookback_days} days; publication_date threshold: >{from_date}")
    
    print(f"Searching {len(journals)} journals")
    
    stats = {
        "executed": True,
        "skip_reason": None,
        "fetched": 0,
        "curated": 0,
        "journals_searched": 0,
        "journals_failed": 0
    }
    
    all_papers = []
    for journal in journals:
        stats["journals_searched"] += 1
        papers = search_openalex_journal(journal, config, state, lookback_days)
        if papers is None:  # Failed journal
            stats["journals_failed"] += 1
        else:
            all_papers.extend(papers)
            stats["fetched"] += len(papers)
        time.sleep(1)  # Politeness delay between journal queries
    
    print(f"\nOpenAlex executed: yes")
    print(f"Found {len(all_papers)} new papers from OpenAlex (after deduplication)")
    print(f"Journals searched: {stats['journals_searched']}, Failed: {stats['journals_failed']}")
    
    return all_papers, stats


# ==================== Curation ====================

def curate_papers(papers: List[Paper], config: Dict) -> List[Paper]:
    """
    Score, filter, and rank papers by relevance.
    Returns curated list of papers.
    """
    print("\n=== Curating Papers ===")
    print(f"Total papers before curation: {len(papers)}")
    
    # Score all papers
    journals_by_name = {j["name"]: j for j in config.get("journals", [])}
    
    for paper in papers:
        journal_config = None
        if paper.journal_name and paper.journal_name in journals_by_name:
            journal_config = journals_by_name[paper.journal_name]
        
        paper.relevance_score = calculate_relevance_score(paper, config, journal_config)
    
    # Filter by education intent pass (hard gate for arXiv)
    edu_intent_filtered = []
    for paper in papers:
        if paper.source_type == "arxiv" and not paper.education_intent_pass:
            print(f"  Filtered (education intent): {paper.title[:60]}...")
            continue
        edu_intent_filtered.append(paper)
    
    print(f"After education intent filtering: {len(edu_intent_filtered)}")
    
    # Filter by minimum score and education intent for OpenAlex
    curated = []
    for paper in edu_intent_filtered:
        journal_config = None
        if paper.journal_name and paper.journal_name in journals_by_name:
            journal_config = journals_by_name[paper.journal_name]
        
        # Check minimum score
        if not passes_min_score(paper.relevance_score, config, journal_config):
            continue
        
        # Check education intent for Tier B practice journals
        if journal_config and not check_education_intent_terms(paper, journal_config):
            print(f"  Filtered (Tier B education intent): {paper.title[:60]}...")
            continue
        
        curated.append(paper)
    
    # Sort by relevance score (descending)
    curated.sort(key=lambda p: p.relevance_score, reverse=True)
    
    # Limit total items
    max_items = config.get("default_rules", {}).get("max_new_items_total_per_run", MAX_NEW_ITEMS_TOTAL)
    curated = curated[:max_items]
    
    print(f"Papers after curation: {len(curated)}")
    print(f"Top curated papers:")
    for i, paper in enumerate(curated[:5]):
        print(f"  {i+1}. [{paper.relevance_score:.1f}] [{paper.source_type}] {paper.title[:60]}...")
    
    return curated


# ==================== Zotero Collections ====================

def get_or_create_zotero_collection(zot: zotero.Zotero, collection_name: str) -> Optional[str]:
    """
    Get or create a Zotero collection by name.
    Returns collection key or None on error.
    """
    try:
        # Get all collections
        collections = zot.collections()
        
        # Search for collection by name
        for coll in collections:
            if coll.get("data", {}).get("name") == collection_name:
                return coll["key"]
        
        # Collection not found, create it
        print(f"  Creating Zotero collection: {collection_name}")
        template = zot.collection_template()
        template["name"] = collection_name
        created = zot.create_collections([template])
        
        if created.get("success"):
            coll_key = created["success"]["0"]
            print(f"  Created collection: {coll_key}")
            return coll_key
        else:
            print(f"  Failed to create collection: {created}")
            return None
            
    except Exception as e:
        print(f"  Error getting/creating collection '{collection_name}': {e}")
        return None


def resolve_zotero_collections(zot: zotero.Zotero) -> Dict[str, str]:
    """
    Resolve Zotero collection keys for arXiv and peer-reviewed collections.
    Returns dict with collection_name -> collection_key.
    """
    collections = {}
    
    # Resolve arXiv collection
    arxiv_key = get_or_create_zotero_collection(zot, "AI-arxiv-pubs")
    if arxiv_key:
        collections["AI-arxiv-pubs"] = arxiv_key
    
    # Resolve peer-reviewed collection
    peer_reviewed_key = get_or_create_zotero_collection(zot, "AI-peer-reviewed-pubs")
    if peer_reviewed_key:
        collections["AI-peer-reviewed-pubs"] = peer_reviewed_key
    
    return collections


# ==================== PDF Download ====================

def download_pdf(paper: Paper) -> bool:
    """
    Download PDF for a paper if URL is available.
    Returns True if successful, False otherwise.
    Updates paper.pdf_downloaded and paper.pdf_path.
    """
    if not paper.pdf_url:
        paper.attachment_missing = True
        return False
    
    try:
        # Create temp directory for PDFs
        pdf_dir = Path("/tmp/llm_edu_pdfs")
        pdf_dir.mkdir(exist_ok=True)
        
        # Generate filename
        safe_id = paper.identifier.replace(":", "_").replace("/", "_")
        pdf_path = pdf_dir / f"{safe_id}.pdf"
        
        # Download PDF
        response = retry_with_backoff(requests.get, paper.pdf_url, timeout=30)
        response.raise_for_status()
        
        # Save to file
        with open(pdf_path, "wb") as f:
            f.write(response.content)
        
        paper.pdf_downloaded = True
        paper.pdf_path = pdf_path
        return True
        
    except Exception as e:
        print(f"  Error downloading PDF for {paper.identifier}: {e}")
        paper.attachment_missing = True
        return False


# ==================== Zotero Import ====================

def import_to_zotero(paper: Paper, zot: zotero.Zotero, collection_keys: Dict[str, str]) -> bool:
    """
    Import paper to Zotero with PDF attachment, citekey, and collection routing.
    Returns True if successful (item created and recorded as seen).
    """
    if not ZOTERO_LIBRARY_ID or not ZOTERO_API_KEY:
        print("  Skipping Zotero import: credentials not configured")
        return False
    
    try:
        # Determine item type and collection
        if paper.source_type == "arxiv":
            item_type = "report"
            collection_name = "AI-arxiv-pubs"
        else:
            item_type = "journalArticle"
            collection_name = "AI-peer-reviewed-pubs"
        
        # Get collection key
        collection_key = collection_keys.get(collection_name)
        if collection_key:
            paper.zotero_collection_key = collection_key
            paper.zotero_collection_name = collection_name
        
        # Create item template
        template = zot.item_template(item_type)
        
        # Fill in metadata
        template["title"] = paper.title
        template["abstractNote"] = paper.abstract[:10000]  # Limit length
        template["url"] = paper.url
        template["date"] = paper.publication_date or ""
        
        if paper.doi:
            template["DOI"] = paper.doi
        
        if paper.journal_name:
            template["publicationTitle"] = paper.journal_name
        
        # For arXiv, set publisher
        if paper.source_type == "arxiv":
            template["institution"] = "arXiv"
        
        # Add creators (authors)
        template["creators"] = []
        for author in paper.authors[:50]:  # Limit number of authors
            # Try to parse into family/given
            parts = author.strip().split()
            if len(parts) >= 2:
                template["creators"].append({
                    "creatorType": "author",
                    "firstName": " ".join(parts[:-1]),
                    "lastName": parts[-1],
                })
            else:
                template["creators"].append({
                    "creatorType": "author",
                    "name": author,
                })
        
        # Add collection
        if collection_key:
            template["collections"] = [collection_key]
        
        # Add traceability in extra field (NO citekey - will be synced from Zotero later)
        extra_lines = []
        
        # Add Work-ID for traceability
        extra_lines.append(f"Work-ID: {paper.identifier}")
        
        # Add source tracking
        extra_lines.append(f"Source: {paper.identifier}")
        
        if paper.relevance_score:
            extra_lines.append(f"Relevance Score: {paper.relevance_score:.1f}")
        
        if paper.attachment_missing:
            extra_lines.append("Note: PDF attachment not available")
        
        template["extra"] = "\n".join(extra_lines)
        
        # Create item in Zotero
        created = zot.create_items([template])
        
        if not created.get("success"):
            print(f"  Failed to create Zotero item for {paper.identifier}")
            paper.import_status = "failed"
            paper.last_error = "Failed to create item"
            return False
        
        item_key = created["success"]["0"]
        paper.zotero_item_key = item_key
        
        # Try to get the item version
        try:
            item_data = zot.item(item_key)
            if item_data and "version" in item_data:
                paper.zotero_item_version = item_data["version"]
        except Exception as e:
            print(f"  Warning: Could not fetch item version: {e}")
            pass  # Version is optional
        
        print(f"  Created Zotero item: {item_key}")
        
        # Upload PDF attachment if available
        if paper.pdf_downloaded and paper.pdf_path and paper.pdf_path.exists():
            try:
                result = zot.attachment_simple([str(paper.pdf_path)], item_key)
                if result:
                    paper.zotero_attachment_key = str(result)
                print(f"  Uploaded PDF attachment for {item_key}")
                paper.import_status = "imported_with_pdf"
            except Exception as e:
                print(f"  Error uploading PDF attachment: {e}")
                paper.import_status = "imported_no_pdf"
                paper.last_error = f"PDF upload failed: {e}"
        else:
            paper.import_status = "imported_no_pdf"
        
        return True
        
    except Exception as e:
        print(f"  Error importing to Zotero: {e}")
        paper.import_status = "failed"
        paper.last_error = str(e)
        return False


# ==================== Citekey Sync from Zotero ====================

def sync_citekeys_from_zotero(db: Database, zot: Optional[zotero.Zotero]) -> Dict[str, int]:
    """
    FIX 1: Sync citekeys from Zotero to SQLite for items with pending citekeys.
    
    Better BibTeX pins citekeys by adding "Citation Key: <key>" as a line in the Extra field.
    Auto-pinning is controlled by BBT preference "Automatically pin citation key after X seconds".
    Citekeys won't appear in the cloud until Zotero Desktop + sync has happened.
    
    Returns dict with statistics: {"candidates": N, "synced": N, "pending": N, "missing": N}
    """
    print("\n=== Syncing Citekeys from Zotero ===")
    
    if not zot:
        print("Skipping citekey sync: Zotero client not configured")
        return {"candidates": 0, "synced": 0, "pending": 0, "missing": 0}
    
    stats = {"candidates": 0, "synced": 0, "pending": 0, "missing": 0}
    now_utc = datetime.utcnow()
    
    # Query items that need citekeys
    query = "citekey_status = 'pending' AND zotero_item_key IS NOT NULL AND zotero_item_key != ''"
    candidates = list(db["works"].rows_where(query))
    stats["candidates"] = len(candidates)
    
    print(f"Citekey sync candidates: {stats['candidates']}")
    
    if not candidates:
        print("No items pending citekey sync")
        return stats
    
    for row in candidates:
        work_id = row["work_id"]
        item_key = row["zotero_item_key"]
        first_seen = row.get("first_seen_utc", "")
        
        print(f"  Checking work_id={work_id}, zotero_item_key={item_key}")
        
        try:
            # Fetch item from Zotero
            item = zot.item(item_key)
            
            if not item or "data" not in item:
                print(f"    Warning: Could not fetch item {item_key}")
                stats["pending"] += 1
                continue
            
            # FIX 1: Parse Extra field for "Citation Key: <key>" (case-insensitive, whitespace tolerant)
            extra = item["data"].get("extra", "")
            citekey = None
            
            if extra:
                print(f"    Extra field has {len(extra.split(chr(10)))} lines")
                for line in extra.split("\n"):
                    line_stripped = line.strip()
                    # Case-insensitive match for "Citation Key:"
                    if line_stripped.lower().startswith("citation key:"):
                        # Extract citekey after "Citation Key:" (handle whitespace)
                        citekey = line_stripped.split(":", 1)[1].strip()
                        break
            
            if citekey:
                # Citekey found - sync it
                print(f"    ✓ Found citekey: {citekey}")
                
                # Optional: Verify citekey pattern (author+year+optional suffix)
                # Pattern: starts with letters, contains 4-digit year, optional alphanumeric suffix
                if not re.match(r'^[a-zA-Z]+\d{4}[a-zA-Z0-9]*$', citekey):
                    print(f"    ⚠️  Warning: Citekey '{citekey}' does not match expected pattern (author+year+suffix)")
                
                # Update database
                update_data = {
                    "work_id": work_id,
                    "citekey": citekey,
                    "citekey_status": "synced",
                    "citekey_synced_utc": now_utc.isoformat(),
                }
                
                # Store item version if available
                if "version" in item:
                    update_data["zotero_item_version"] = item["version"]
                
                db["works"].update(work_id, update_data)
                stats["synced"] += 1
                
            else:
                # Citekey not found - check if item is old enough to mark as missing
                print(f"    No 'Citation Key:' line found in Extra field")
                if first_seen:
                    try:
                        # Handle ISO 8601 format with or without timezone
                        first_seen_clean = first_seen.replace("Z", "")
                        if "+" in first_seen_clean:
                            # Already has timezone info, remove it
                            first_seen_clean = first_seen_clean.split("+")[0]
                        first_seen_dt = datetime.fromisoformat(first_seen_clean)
                        age_days = (now_utc - first_seen_dt).days
                        
                        if age_days > 30:
                            # Item is old, mark as missing
                            print(f"    Item age: {age_days} days > 30, marking as 'missing'")
                            db["works"].update(work_id, {
                                "citekey_status": "missing",
                                "last_error": f"No Citation Key after {age_days} days. Open Zotero Desktop to trigger BBT."
                            })
                            stats["missing"] += 1
                        else:
                            print(f"    Item age: {age_days} days, keeping as 'pending'")
                            stats["pending"] += 1
                    except (ValueError, TypeError) as e:
                        # Could not parse date, keep as pending
                        print(f"    Could not parse date ({e}), keeping as 'pending'")
                        stats["pending"] += 1
                else:
                    print(f"    No first_seen_utc, keeping as 'pending'")
                    stats["pending"] += 1
        
        except Exception as e:
            print(f"    ❌ Error fetching item {item_key}: {e}")
            # Differentiate between network/API errors and other errors
            if "404" in str(e) or "not found" in str(e).lower():
                print(f"    Item not found in Zotero, may have been deleted")
            elif "401" in str(e) or "403" in str(e):
                print(f"    ⚠️  Authentication error, check API credentials")
            else:
                print(f"    Temporary error, will retry on next run")
            stats["pending"] += 1
            continue
    
    print(f"Citekey sync complete: Synced={stats['synced']}, Still pending={stats['pending']}, Missing={stats['missing']}")
    return stats


# ==================== SQLite Storage ====================

def init_database(db_path: Path) -> Database:
    """Initialize SQLite database with proper schema."""
    db = Database(db_path)
    
    # Create runs table
    db["runs"].create({
        "run_id": str,
        "start_time": str,
        "end_time": str,
        "arxiv_fetched": int,
        "arxiv_gated": int,
        "arxiv_curated": int,
        "openalex_fetched": int,
        "openalex_curated": int,
        "total_imported": int,
        "errors": str,
    }, pk="run_id", if_not_exists=True)
    
    # Create works table
    db["works"].create({
        "work_id": str,
        "citekey": str,
        "citekey_status": str,
        "citekey_synced_utc": str,
        "source": str,
        "doi": str,
        "title": str,
        "abstract": str,
        "authors_json": str,
        "published_date": str,
        "journal": str,
        "venue_id": str,
        "url": str,
        "pdf_url": str,
        "is_open_access": int,
        "relevance_score": float,
        "education_intent_pass": int,
        "first_seen_utc": str,
        "last_seen_utc": str,
        "last_run_id": str,
        "zotero_item_key": str,
        "zotero_item_version": int,
        "zotero_attachment_key": str,
        "zotero_collection_key": str,
        "zotero_collection_name": str,
        "import_status": str,
        "last_error": str,
    }, pk="work_id", if_not_exists=True)
    
    # Add columns to existing database if they don't exist
    try:
        db.execute("ALTER TABLE works ADD COLUMN citekey_status TEXT DEFAULT 'pending'")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    try:
        db.execute("ALTER TABLE works ADD COLUMN citekey_synced_utc TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    try:
        db.execute("ALTER TABLE works ADD COLUMN zotero_item_version INTEGER")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    # Create unique index on citekey where not null
    try:
        db.execute("DROP INDEX IF EXISTS idx_citekey")
        db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_citekey ON works(citekey) WHERE citekey IS NOT NULL AND citekey != ''")
    except sqlite3.OperationalError as e:
        print(f"Warning: Could not create unique index on citekey: {e}")
    
    return db


def store_paper_in_db(paper: Paper, db: Database, run_id: str):
    """Store or update paper in database."""
    now_utc = datetime.utcnow().isoformat()
    
    # Check if work already exists
    existing = list(db["works"].rows_where("work_id = ?", [paper.identifier]))
    
    record = {
        "work_id": paper.identifier,
        "citekey": paper.citekey,
        "citekey_status": paper.citekey_status or "pending",
        "citekey_synced_utc": paper.citekey_synced_utc or "",
        "source": paper.source_type,
        "doi": paper.doi or "",
        "title": paper.title,
        "abstract": paper.abstract,
        "authors_json": json.dumps(paper.authors),
        "published_date": paper.publication_date or "",
        "journal": paper.journal_name or "",
        "venue_id": "",
        "url": paper.url,
        "pdf_url": paper.pdf_url or "",
        "is_open_access": 1 if paper.pdf_url else 0,
        "relevance_score": paper.relevance_score,
        "education_intent_pass": 1 if paper.education_intent_pass else 0,
        "first_seen_utc": existing[0]["first_seen_utc"] if existing else now_utc,
        "last_seen_utc": now_utc,
        "last_run_id": run_id,
        "zotero_item_key": paper.zotero_item_key or "",
        "zotero_item_version": paper.zotero_item_version or 0,
        "zotero_attachment_key": paper.zotero_attachment_key or "",
        "zotero_collection_key": paper.zotero_collection_key or "",
        "zotero_collection_name": paper.zotero_collection_name or "",
        "import_status": paper.import_status or "",
        "last_error": paper.last_error or "",
    }
    
    # Upsert
    db["works"].upsert(record, pk="work_id")


def generate_references_json(db: Database, output_path: Path) -> Dict[str, int]:
    """
    FIX 1: Generate CSL-JSON references from database with protection against overwriting.
    
    Logic:
    - Only includes items with citekey_status='synced' in references.json (stable citations)
    - Creates references_pending.json for items without synced citekeys (transparency)
    - NEVER overwrites references.json with empty file when pending>0 and prior file exists
    - On first-ever run (no prior references.json), writes temporary IDs from work_id
    
    Returns dict with counts: {"synced": N, "pending": N, "file_protected": bool}
    """
    print("\n=== Generating references.json ===")
    
    references = []
    pending_references = []
    
    # Get all works with synced citekeys (for main references.json)
    synced_query = "citekey_status = 'synced' AND citekey IS NOT NULL AND citekey != ''"
    for row in db["works"].rows_where(synced_query, order_by="citekey"):
        csl_item = _build_csl_item(row, use_citekey_as_id=True)
        if csl_item:
            references.append(csl_item)
    
    # Get all works without synced citekeys (for references_pending.json)
    pending_query = "citekey_status != 'synced' OR citekey IS NULL OR citekey = ''"
    for row in db["works"].rows_where(pending_query, order_by="work_id"):
        csl_item = _build_csl_item(row, use_citekey_as_id=False)
        if csl_item:
            pending_references.append(csl_item)
    
    # Sort references by citekey for deterministic output
    references.sort(key=lambda x: x["id"])
    
    stats = {
        "synced": len(references),
        "pending": len(pending_references),
        "file_protected": False
    }
    
    # FIX 1: Protection logic - never overwrite references.json with empty when pending>0
    if len(references) == 0 and len(pending_references) > 0:
        # No synced items yet, but we have pending items
        if output_path.exists():
            # Prior references.json exists - protect it, don't overwrite
            print(f"⚠️  Protection: references.json NOT overwritten (Synced=0, Pending={len(pending_references)})")
            print(f"   Keeping existing references.json unchanged until citekeys sync from Zotero")
            stats["file_protected"] = True
        else:
            # First-ever run, no prior references.json
            # Write temporary references using work_id as ID with annotation
            print(f"⚠️  First run: No prior references.json found")
            print(f"   Writing {len(pending_references)} items with temporary work_id as citation ID")
            print(f"   These will be replaced with proper citekeys after Zotero Desktop syncs")
            
            # Add annotation to each pending reference
            for ref in pending_references:
                if "note" not in ref:
                    ref["note"] = f"Temporary ID: {ref['id']} - will change after citekey sync from Zotero/BetterBibTeX"
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(pending_references, f, indent=2, ensure_ascii=False)
            print(f"Wrote {len(pending_references)} temporary references to {output_path}")
    
    elif len(references) > 0:
        # We have synced items - write them
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(references, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(references)} synced references to {output_path}")
    
    else:
        # No items at all (synced or pending)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        print(f"Wrote 0 references to {output_path} (no items found)")
    
    # Always write references_pending.json for transparency
    pending_path = output_path.parent / "references_pending.json"
    with open(pending_path, "w", encoding="utf-8") as f:
        json.dump(pending_references, f, indent=2, ensure_ascii=False)
    
    print(f"Wrote {len(pending_references)} pending references to {pending_path}")
    
    return stats


def _build_csl_item(row: Dict, use_citekey_as_id: bool = True) -> Optional[Dict]:
    """
    Build a CSL-JSON item from a database row.
    If use_citekey_as_id is True, use citekey as id; otherwise use work_id.
    """
    # Parse authors
    try:
        authors_list = json.loads(row["authors_json"]) if row["authors_json"] else []
    except:
        authors_list = []
    
    # Convert to CSL author format
    csl_authors = []
    for author in authors_list:
        parts = author.strip().split()
        if len(parts) >= 2:
            csl_authors.append({
                "family": parts[-1],
                "given": " ".join(parts[:-1])
            })
        else:
            csl_authors.append({"literal": author})
    
    # Determine type
    if row["source"] == "arxiv":
        csl_type = "report"
    elif row["journal"]:
        csl_type = "article-journal"
    else:
        csl_type = "document"
    
    # Extract year from date
    issued_parts = None
    if row["published_date"]:
        match = re.search(r'(\d{4})-?(\d{2})?-?(\d{2})?', row["published_date"])
        if match:
            parts = [int(match.group(1))]
            if match.group(2):
                parts.append(int(match.group(2)))
            if match.group(3):
                parts.append(int(match.group(3)))
            issued_parts = [parts]
    
    # Determine ID
    if use_citekey_as_id and row.get("citekey"):
        item_id = row["citekey"]
    else:
        item_id = row["work_id"]
    
    # Build CSL-JSON item
    csl_item = {
        "id": item_id,
        "type": csl_type,
        "title": row["title"],
    }
    
    if csl_authors:
        csl_item["author"] = csl_authors
    
    if issued_parts:
        csl_item["issued"] = {"date-parts": issued_parts}
    
    if row["abstract"]:
        csl_item["abstract"] = row["abstract"]
    
    if row["doi"]:
        csl_item["DOI"] = row["doi"]
        csl_item["URL"] = f"https://doi.org/{row['doi']}"
    elif row["url"]:
        csl_item["URL"] = row["url"]
    
    if row["journal"]:
        csl_item["container-title"] = row["journal"]
    
    if row["source"] == "arxiv":
        csl_item["publisher"] = "arXiv"
    
    return csl_item


# ==================== Digest Generation ====================

def generate_digest(papers: List[Paper], output_path: Path, citekey_stats: Dict[str, int]):
    """
    FIX 1: Generate Markdown digest of curated papers with pending citekey hint.
    """
    print("\n=== Generating Digest ===")
    
    lines = [
        "# LLM Education Literature Digest",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}*",
        "",
    ]
    
    # FIX 1: Add user-facing hint if citekeys are pending
    if citekey_stats.get("synced", 0) == 0 and citekey_stats.get("pending", 0) > 0:
        lines.extend([
            "---",
            "",
            "**⚠️  Citation keys are pending:**",
            "",
            f"- {citekey_stats['pending']} items imported to Zotero without citation keys yet",
            "- To generate citation keys: open Zotero Desktop with Better BibTeX plugin enabled and sync",
            "- Better BibTeX will auto-pin citation keys based on your configured format (e.g., `[auth][year]`)",
            "- Next workflow run will populate `references.json` with proper citekeys for Quarto/Pandoc citations",
            "- Current status: `references.json` contains temporary work IDs or is unchanged from last run",
            "",
            "---",
            "",
        ])
    
    if not papers:
        lines.extend([
            "No new items found in this run.",
            ""
        ])
    else:
        lines.extend([
            f"**{len(papers)} new items**",
            "",
        ])
        
        for i, paper in enumerate(papers, 1):
            lines.append(f"## {i}. {paper.title}")
            lines.append("")
            
            # Authors
            if paper.authors:
                author_list = ", ".join(paper.authors[:5])
                if len(paper.authors) > 5:
                    author_list += f" et al. ({len(paper.authors)} authors)"
                lines.append(f"**Authors:** {author_list}")
                lines.append("")
            
            # Source
            source_info = paper.source_type
            if paper.journal_name:
                source_info += f" • {paper.journal_name}"
            if paper.publication_date:
                source_info += f" • {paper.publication_date[:10]}"
            lines.append(f"**Source:** {source_info}")
            lines.append("")
            
            # Short description (truncated abstract)
            if paper.abstract:
                short_desc = paper.abstract[:300]
                if len(paper.abstract) > 300:
                    short_desc += "..."
                lines.append(f"**Summary:** {short_desc}")
                lines.append("")
            
            # Links
            lines.append(f"**Links:**")
            lines.append(f"- Landing page: {paper.url}")
            if paper.pdf_url:
                lines.append(f"- PDF: {paper.pdf_url}")
            if paper.doi:
                lines.append(f"- DOI: https://doi.org/{paper.doi}")
            lines.append("")
            
            # Relevance score
            lines.append(f"*Relevance score: {paper.relevance_score:.1f}*")
            lines.append("")
            lines.append("---")
            lines.append("")
    
    digest_content = "\n".join(lines)
    
    # Write digest
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(digest_content)
    
    print(f"Digest written to {output_path}")


# ==================== Main Pipeline ====================

def main():
    """
    Main entrypoint for the literature watcher.
    FIX 4: Enhanced error handling - exit 0 for expected conditions, only fail on fatal errors.
    """
    print("=" * 60)
    print("Weekly Literature Watcher: LLMs in University Didactics")
    print("=" * 60)
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()
    
    # FIX 4: Track workflow health (success/warning/failed)
    workflow_health = "success"
    warnings = []
    
    # Generate run ID
    run_id = start_time.strftime("%Y%m%d_%H%M%S")
    
    # Validate secrets (FIX 4: non-fatal warnings)
    if not ZOTERO_LIBRARY_ID or not ZOTERO_API_KEY:
        warning_msg = "Zotero credentials not configured. Items will not be imported."
        print(f"⚠️  WARNING: {warning_msg}")
        print("Set ZOTERO_LIBRARY_ID and ZOTERO_API_KEY environment variables.")
        warnings.append(warning_msg)
        workflow_health = "warning"
    
    if not OPEN_ALEX_API_KEY:
        warning_msg = "OpenAlex API key not configured. Requests may be rate-limited."
        print(f"⚠️  WARNING: {warning_msg}")
        warnings.append(warning_msg)
        # Don't downgrade to warning for missing OpenAlex key alone
    
    # Initialize database (FIX 4: fatal if fails)
    print("\n=== Initializing Database ===")
    try:
        db = init_database(DB_PATH)
        print(f"Database initialized: {DB_PATH}")
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not initialize database: {e}")
        sys.exit(1)
    
    # Load configuration and state (FIX 4: fatal if journals.json missing)
    print("\n=== Loading Configuration ===")
    try:
        if not JOURNALS_CONFIG_PATH.exists():
            print(f"❌ FATAL ERROR: journals.json not found at {JOURNALS_CONFIG_PATH}")
            sys.exit(1)
        config = load_config()
        print(f"Loaded configuration from {JOURNALS_CONFIG_PATH}")
        print(f"Journals configured: {len(config.get('journals', []))}")
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not load configuration: {e}")
        sys.exit(1)
    
    # Update lookback days in config
    if "default_rules" not in config:
        config["default_rules"] = {}
    config["default_rules"]["lookback_days"] = OPENALEX_LOOKBACK_DAYS
    
    state = load_state()
    print(f"Loaded state from {STATE_PATH if STATE_PATH.exists() else 'new state'}")
    print(f"Previously seen: {len(state.seen_arxiv_ids)} arXiv, {len(state.seen_openalex_ids)} OpenAlex, {len(state.seen_dois)} DOIs")
    
    # Resolve Zotero collections (FIX 4: non-fatal, only if credentials present)
    zot = None
    collection_keys = {}
    if ZOTERO_LIBRARY_ID and ZOTERO_API_KEY:
        print("\n=== Resolving Zotero Collections ===")
        try:
            zot = zotero.Zotero(ZOTERO_LIBRARY_ID, "user", ZOTERO_API_KEY)
            collection_keys = resolve_zotero_collections(zot)
            print(f"Resolved collections: {collection_keys}")
        except Exception as e:
            error_str = str(e)
            if "401" in error_str or "403" in error_str:
                # FIX 4: Authentication errors are fatal if Zotero is required
                print(f"❌ FATAL ERROR: Zotero authentication failed (401/403): {e}")
                print("Check ZOTERO_LIBRARY_ID and ZOTERO_API_KEY")
                sys.exit(1)
            else:
                # Other Zotero errors are warnings
                warning_msg = f"Error resolving Zotero collections: {e}"
                print(f"⚠️  WARNING: {warning_msg}")
                warnings.append(warning_msg)
                workflow_health = "warning"
    
    # Search pipelines (FIX 4: non-fatal if return 0 results)
    arxiv_papers, arxiv_stats = search_arxiv(config, state)
    openalex_papers, openalex_stats = search_openalex(config, state)
    
    all_papers = arxiv_papers + openalex_papers
    print(f"\n=== Total Papers Collected: {len(all_papers)} ===")
    
    # FIX 4: Log warnings if no results (but continue)
    if arxiv_stats["fetched"] == 0:
        warning_msg = "arXiv returned 0 results"
        print(f"⚠️  WARNING: {warning_msg}")
        warnings.append(warning_msg)
    
    if openalex_stats.get("fetched", 0) == 0:
        if openalex_stats.get("skip_reason"):
            warning_msg = f"OpenAlex skipped: {openalex_stats['skip_reason']}"
        else:
            warning_msg = "OpenAlex returned 0 results"
        print(f"⚠️  WARNING: {warning_msg}")
        warnings.append(warning_msg)
    
    # Curate papers
    curated_papers = curate_papers(all_papers, config)
    
    # Download PDFs and import to Zotero
    print("\n=== Processing Papers ===")
    successfully_imported = []
    
    for i, paper in enumerate(curated_papers, 1):
        print(f"\n[{i}/{len(curated_papers)}] Processing: {paper.title[:60]}...")
        
        # Download PDF
        if paper.pdf_url:
            print(f"  Downloading PDF...")
            download_pdf(paper)
        else:
            print(f"  No PDF URL available")
            paper.attachment_missing = True
        
        # Import to Zotero
        if zot:
            print(f"  Importing to Zotero...")
            success = import_to_zotero(paper, zot, collection_keys)
            
            if success:
                successfully_imported.append(paper)
                # Mark as seen in state
                if paper.source_type == "arxiv":
                    arxiv_id = paper.source_metadata.get("arxiv_id")
                    if arxiv_id:
                        state.seen_arxiv_ids.add(arxiv_id)
                elif paper.source_type == "openalex":
                    openalex_id = paper.source_metadata.get("openalex_id")
                    if openalex_id:
                        state.seen_openalex_ids.add(openalex_id)
                
                if paper.doi:
                    state.seen_dois.add(paper.doi)
            else:
                print(f"  Import failed, will retry on next run")
        else:
            # If no Zotero credentials, just add to digest
            successfully_imported.append(paper)
        
        # Store in database
        store_paper_in_db(paper, db, run_id)
    
    # Sync citekeys from Zotero for all pending items
    sync_stats = sync_citekeys_from_zotero(db, zot)
    
    # FIX 1: Generate references.json with protection logic
    ref_stats = generate_references_json(db, REFERENCES_PATH)
    
    # FIX 1: Generate digest with citekey hint
    generate_digest(successfully_imported, DIGEST_PATH, sync_stats)
    
    # Record run in database
    end_time = datetime.now()
    arxiv_curated = len([p for p in curated_papers if p.source_type == "arxiv"])
    openalex_curated = len([p for p in curated_papers if p.source_type == "openalex"])
    
    db["runs"].insert({
        "run_id": run_id,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "arxiv_fetched": arxiv_stats["fetched"],
        "arxiv_gated": arxiv_stats["excluded_by_gate"] + arxiv_stats["excluded_by_disambiguation"] + arxiv_stats["excluded_by_negative_terms"],
        "arxiv_curated": arxiv_curated,
        "openalex_fetched": openalex_stats.get("fetched", 0),
        "openalex_curated": openalex_curated,
        "total_imported": len(successfully_imported),
        "errors": "; ".join(warnings) if warnings else "",
    })
    
    # Update state
    state.last_run = datetime.now().isoformat()
    save_state(state)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"arXiv: Fetched {arxiv_stats['fetched']}, "
          f"Gated {arxiv_stats['excluded_by_gate']}, "
          f"Disambiguation {arxiv_stats['excluded_by_disambiguation']}, "
          f"Negative terms {arxiv_stats['excluded_by_negative_terms']}, "
          f"Curated {arxiv_curated}")
    print(f"OpenAlex: Fetched {openalex_stats.get('fetched', 0)}, Curated {openalex_curated}")
    if not openalex_stats.get("executed", True):
        print(f"  (OpenAlex skipped: {openalex_stats.get('skip_reason', 'unknown')})")
    print(f"Total collected: {len(all_papers)}")
    print(f"Total curated: {len(curated_papers)}")
    print(f"Successfully processed: {len(successfully_imported)}")
    print(f"Citekey sync: Synced={sync_stats['synced']}, Pending={sync_stats['pending']}, Missing={sync_stats['missing']}")
    if ref_stats.get("file_protected"):
        print(f"⚠️  references.json protected from overwrite (pending citekeys)")
    print(f"Digest written to: {DIGEST_PATH}")
    print(f"Database written to: {DB_PATH}")
    print(f"References written to: {REFERENCES_PATH}")
    print(f"State saved to: {STATE_PATH}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Duration: {(end_time - start_time).total_seconds():.1f} seconds")
    
    # FIX 4: Workflow health summary
    if warnings:
        print(f"\n⚠️  Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")
    
    print(f"\n🔍 Workflow health: {workflow_health}")
    print("=" * 60)
    
    # FIX 4: Exit 0 for success/warning, only fail on truly fatal issues
    # (Fatal issues already exited with sys.exit(1) above)
    sys.exit(0)




if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
