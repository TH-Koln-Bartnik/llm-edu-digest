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

import arxiv
import requests
from pyzotero import zotero


# ==================== Configuration ====================

# Environment variables (secrets)
ZOTERO_LIBRARY_ID = os.environ.get("ZOTERO_LIBRARY_ID", "")
ZOTERO_API_KEY = os.environ.get("ZOTERO_API_KEY", "")
OPEN_ALEX_API_KEY = os.environ.get("OPEN_ALEX_API_KEY", "")

# Contact email for OpenAlex API (polite pool)
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "github-bot@users.noreply.github.com")

# File paths
JOURNALS_CONFIG_PATH = Path(__file__).parent / "journals.json"
STATE_PATH = Path(__file__).parent / "state.json"
DIGEST_PATH = Path(__file__).parent / "digest.md"

# API defaults
ARXIV_MAX_RESULTS = 100
OPENALEX_LOOKBACK_DAYS = 10
OPENALEX_MAX_RESULTS_PER_JOURNAL = 100
MAX_NEW_ITEMS_TOTAL = 50

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
ARXIV_POLITENESS_DELAY = 3  # seconds between arXiv API calls


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
    pdf_downloaded: bool = False
    pdf_path: Optional[Path] = None
    attachment_missing: bool = False


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


def calculate_relevance_score(paper: Paper, config: Dict, journal_config: Optional[Dict] = None) -> float:
    """
    Calculate relevance score based on keyword matches in title and abstract.
    Returns a deterministic, explainable score.
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
    
    # Education terms
    edu_terms = [normalize_text(t) for t in default_rules.get("education_terms", [])]
    for term in edu_terms:
        if term in title_norm:
            score += weights.get("education_term_in_title", 4)
            break
    for term in edu_terms:
        if term in abstract_norm:
            score += weights.get("education_term_in_abstract", 2)
            break
    
    # Bonus phrases in title
    bonus_phrases = [normalize_text(p) for p in scoring.get("bonus_phrases_title", [])]
    for phrase in bonus_phrases:
        if phrase in title_norm:
            score += weights.get("bonus_phrase_in_title", 2)
            break
    
    # Penalty phrases
    penalty_phrases = [normalize_text(p) for p in scoring.get("penalty_phrases_anywhere", [])]
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

def search_arxiv(config: Dict, state: State) -> List[Paper]:
    """
    Search arXiv for papers on LLMs in education.
    Returns list of Paper objects.
    """
    print("\n=== Searching arXiv ===")
    
    default_rules = config.get("default_rules", {})
    llm_terms = default_rules.get("llm_terms", [])
    edu_terms = default_rules.get("education_terms", [])
    
    # Build search query: (LLM terms) AND (education terms)
    llm_query = " OR ".join([f'all:"{term}"' for term in llm_terms[:3]])  # Limit query complexity
    edu_query = " OR ".join([f'all:"{term}"' for term in edu_terms[:5]])
    query = f"({llm_query}) AND ({edu_query})"
    
    print(f"arXiv query: {query}")
    print(f"Max results: {ARXIV_MAX_RESULTS}")
    
    search = arxiv.Search(
        query=query,
        max_results=ARXIV_MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    
    papers = []
    try:
        results = list(search.results())
        print(f"Retrieved {len(results)} results from arXiv")
        
        for result in results:
            arxiv_id = result.entry_id.split("/")[-1]  # Extract ID from URL
            identifier = f"arxiv:{arxiv_id}"
            
            # Skip if already seen
            if arxiv_id in state.seen_arxiv_ids:
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
                source_metadata={"arxiv_id": arxiv_id},
            )
            
            papers.append(paper)
            
        time.sleep(ARXIV_POLITENESS_DELAY)  # Politeness delay
        
    except Exception as e:
        print(f"Error searching arXiv: {e}")
    
    print(f"Found {len(papers)} new arXiv papers (after deduplication)")
    return papers


# ==================== OpenAlex Pipeline ====================

def resolve_openalex_source_id(journal_name: str, issn: List[str], state: State) -> Optional[str]:
    """
    Resolve journal name/ISSN to OpenAlex source ID.
    Uses cached values from state if available.
    """
    # Check cache first
    if journal_name in state.resolved_source_ids:
        cached_id = state.resolved_source_ids[journal_name]
        print(f"  Using cached source ID for '{journal_name}': {cached_id}")
        return cached_id
    
    print(f"  Resolving OpenAlex source ID for: {journal_name}")
    
    # Try ISSN first if available
    if issn:
        for issn_value in issn:
            try:
                url = f"https://api.openalex.org/sources?filter=issn:{issn_value}"
                headers = {"User-Agent": f"mailto:{CONTACT_EMAIL}"}
                if OPEN_ALEX_API_KEY:
                    url += f"&api_key={OPEN_ALEX_API_KEY}"
                
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if data.get("results") and len(data["results"]) > 0:
                    source_id = data["results"][0]["id"].split("/")[-1]
                    print(f"  Resolved via ISSN {issn_value}: {source_id}")
                    state.resolved_source_ids[journal_name] = source_id
                    return source_id
            except Exception as e:
                print(f"  Error resolving via ISSN {issn_value}: {e}")
    
    # Try by name
    try:
        url = f"https://api.openalex.org/sources?search={requests.utils.quote(journal_name)}"
        headers = {"User-Agent": f"mailto:{CONTACT_EMAIL}"}
        if OPEN_ALEX_API_KEY:
            url += f"&api_key={OPEN_ALEX_API_KEY}"
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("results") and len(data["results"]) > 0:
            # Try to match by name similarity
            for result in data["results"]:
                result_name = result.get("display_name", "")
                if normalize_text(result_name) == normalize_text(journal_name):
                    source_id = result["id"].split("/")[-1]
                    print(f"  Resolved via name: {source_id}")
                    state.resolved_source_ids[journal_name] = source_id
                    return source_id
            
            # Use first result as fallback
            source_id = data["results"][0]["id"].split("/")[-1]
            print(f"  Resolved via name (best match): {source_id}")
            state.resolved_source_ids[journal_name] = source_id
            return source_id
    except Exception as e:
        print(f"  Error resolving via name: {e}")
    
    print(f"  Could not resolve source ID for {journal_name}")
    return None


def search_openalex_journal(journal: Dict, config: Dict, state: State, lookback_days: int) -> List[Paper]:
    """
    Search OpenAlex for papers in a specific journal.
    Returns list of Paper objects.
    """
    journal_name = journal["name"]
    print(f"\n  Searching: {journal_name}")
    
    # Resolve source ID
    issn_list = journal.get("identifiers", {}).get("issn", [])
    source_id = resolve_openalex_source_id(journal_name, issn_list, state)
    
    if not source_id:
        print(f"  Skipping {journal_name}: could not resolve source ID")
        return []
    
    # Build search query
    default_rules = config.get("default_rules", {})
    llm_terms = default_rules.get("llm_terms", [])
    edu_terms = default_rules.get("education_terms", [])
    
    # Combine terms for search
    search_terms = llm_terms[:3] + edu_terms[:3]  # Limit for API
    search_query = " OR ".join(search_terms)
    
    # Date filter
    from_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    
    # Build filter
    filters = f"primary_location.source.id:{source_id},from_publication_date:{from_date}"
    
    url = f"https://api.openalex.org/works?filter={filters}&search={requests.utils.quote(search_query)}&per_page={OPENALEX_MAX_RESULTS_PER_JOURNAL}"
    
    headers = {"User-Agent": f"mailto:{CONTACT_EMAIL}"}
    if OPEN_ALEX_API_KEY:
        url += f"&api_key={OPEN_ALEX_API_KEY}"
    
    try:
        response = retry_with_backoff(requests.get, url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        results = data.get("results", [])
        print(f"  Retrieved {len(results)} results")
        
        papers = []
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
            
            papers.append(paper)
        
        return papers
        
    except Exception as e:
        print(f"  Error searching OpenAlex for {journal_name}: {e}")
        return []


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


def search_openalex(config: Dict, state: State) -> List[Paper]:
    """
    Search OpenAlex for papers in whitelisted journals.
    Returns list of Paper objects.
    """
    print("\n=== Searching OpenAlex ===")
    
    journals = config.get("journals", [])
    lookback_days = config.get("default_rules", {}).get("lookback_days", OPENALEX_LOOKBACK_DAYS)
    
    print(f"Searching {len(journals)} journals")
    print(f"Lookback window: {lookback_days} days")
    
    all_papers = []
    for journal in journals:
        papers = search_openalex_journal(journal, config, state, lookback_days)
        all_papers.extend(papers)
        time.sleep(1)  # Politeness delay between journal queries
    
    print(f"\nFound {len(all_papers)} new papers from OpenAlex (after deduplication)")
    return all_papers


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
    
    # Filter by minimum score and education intent
    curated = []
    for paper in papers:
        journal_config = None
        if paper.journal_name and paper.journal_name in journals_by_name:
            journal_config = journals_by_name[paper.journal_name]
        
        # Check minimum score
        if not passes_min_score(paper.relevance_score, config, journal_config):
            continue
        
        # Check education intent for Tier B practice journals
        if journal_config and not check_education_intent_terms(paper, journal_config):
            print(f"  Filtered (education intent): {paper.title[:60]}...")
            continue
        
        curated.append(paper)
    
    # Sort by relevance score (descending)
    curated.sort(key=lambda p: p.relevance_score, reverse=True)
    
    # Limit total items
    max_items = config.get("default_rules", {}).get("max_new_items_total_per_run", MAX_NEW_ITEMS_TOTAL)
    curated = curated[:max_items]
    
    print(f"Papers after curation: {len(curated)}")
    for i, paper in enumerate(curated[:5]):
        print(f"  {i+1}. [{paper.relevance_score:.1f}] {paper.title[:60]}...")
    
    return curated


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

def import_to_zotero(paper: Paper) -> bool:
    """
    Import paper to Zotero with PDF attachment if available.
    Returns True if successful (item created and recorded as seen).
    """
    if not ZOTERO_LIBRARY_ID or not ZOTERO_API_KEY:
        print("  Skipping Zotero import: credentials not configured")
        return False
    
    try:
        # Initialize Zotero client
        zot = zotero.Zotero(ZOTERO_LIBRARY_ID, "user", ZOTERO_API_KEY)
        
        # Determine item type
        item_type = "journalArticle" if paper.journal_name else "document"
        
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
        
        # Add creators (authors)
        template["creators"] = []
        for author in paper.authors[:50]:  # Limit number of authors
            template["creators"].append({
                "creatorType": "author",
                "name": author,
            })
        
        # Add traceability in extra field
        extra_lines = [f"Source: {paper.identifier}"]
        if paper.relevance_score:
            extra_lines.append(f"Relevance Score: {paper.relevance_score:.1f}")
        if paper.attachment_missing:
            extra_lines.append("Note: PDF attachment not available")
        template["extra"] = "\n".join(extra_lines)
        
        # Create item in Zotero
        created = zot.create_items([template])
        
        if not created.get("success"):
            print(f"  Failed to create Zotero item for {paper.identifier}")
            return False
        
        item_key = created["success"]["0"]
        print(f"  Created Zotero item: {item_key}")
        
        # Upload PDF attachment if available
        if paper.pdf_downloaded and paper.pdf_path and paper.pdf_path.exists():
            try:
                zot.attachment_simple([str(paper.pdf_path)], item_key)
                print(f"  Uploaded PDF attachment for {item_key}")
            except Exception as e:
                print(f"  Error uploading PDF attachment: {e}")
                # Don't fail the import if PDF upload fails
        
        return True
        
    except Exception as e:
        print(f"  Error importing to Zotero: {e}")
        return False


# ==================== Digest Generation ====================

def generate_digest(papers: List[Paper], output_path: Path):
    """Generate Markdown digest of curated papers."""
    print("\n=== Generating Digest ===")
    
    if not papers:
        digest_content = f"""# LLM Education Literature Digest

*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}*

No new items found in this run.
"""
    else:
        lines = [
            "# LLM Education Literature Digest",
            "",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}*",
            "",
            f"**{len(papers)} new items**",
            "",
        ]
        
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
    """Main entrypoint for the literature watcher."""
    print("=" * 60)
    print("Weekly Literature Watcher: LLMs in University Didactics")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()
    
    # Validate secrets
    if not ZOTERO_LIBRARY_ID or not ZOTERO_API_KEY:
        print("WARNING: Zotero credentials not configured. Items will not be imported.")
        print("Set ZOTERO_LIBRARY_ID and ZOTERO_API_KEY environment variables.")
    
    if not OPEN_ALEX_API_KEY:
        print("WARNING: OpenAlex API key not configured. Requests may be rate-limited.")
    
    # Load configuration and state
    print("\n=== Loading Configuration ===")
    config = load_config()
    print(f"Loaded configuration from {JOURNALS_CONFIG_PATH}")
    print(f"Journals configured: {len(config.get('journals', []))}")
    
    state = load_state()
    print(f"Loaded state from {STATE_PATH if STATE_PATH.exists() else 'new state'}")
    print(f"Previously seen: {len(state.seen_arxiv_ids)} arXiv, {len(state.seen_openalex_ids)} OpenAlex, {len(state.seen_dois)} DOIs")
    
    # Search pipelines
    arxiv_papers = search_arxiv(config, state)
    openalex_papers = search_openalex(config, state)
    
    all_papers = arxiv_papers + openalex_papers
    print(f"\n=== Total Papers Collected: {len(all_papers)} ===")
    
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
        if ZOTERO_LIBRARY_ID and ZOTERO_API_KEY:
            print(f"  Importing to Zotero...")
            success = import_to_zotero(paper)
            
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
    
    # Generate digest (only successfully imported items)
    generate_digest(successfully_imported, DIGEST_PATH)
    
    # Update state
    state.last_run = datetime.now().isoformat()
    save_state(state)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Papers collected: {len(all_papers)}")
    print(f"Papers curated: {len(curated_papers)}")
    print(f"Papers successfully processed: {len(successfully_imported)}")
    print(f"Digest written to: {DIGEST_PATH}")
    print(f"State saved to: {STATE_PATH}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)


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
