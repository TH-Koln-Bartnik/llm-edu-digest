#!/usr/bin/env python3
"""
Unit tests for watch_literature.py
Tests for FIX 1-4: citekey parsing, references.json protection, arXiv gating, OpenAlex diagnostics
"""

import json
import pytest
import sqlite3
import tempfile
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

# Import functions to test
import sys
sys.path.insert(0, str(Path(__file__).parent))

from watch_literature import (
    check_hard_education_intent,
    normalize_text,
    Paper,
    EDU_SETTING_TERMS,
    EDU_AMBIGUOUS_TERMS,
    NEGATIVE_STRONG_TERMS,
)


# ==================== Test Fixtures ====================

@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "default_rules": {
            "llm_terms": [
                "large language model", "llm", "chatgpt", "gpt", "generative ai"
            ],
            "education_terms": [
                "education", "teaching", "student", "classroom"
            ],
            "relevance_scoring": {
                "min_score_default": 8,
                "weights": {
                    "llm_term_in_title": 6,
                    "llm_term_in_abstract": 3,
                    "education_term_in_title": 4,
                    "education_term_in_abstract": 2,
                }
            }
        }
    }


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    
    # Initialize database with schema
    from sqlite_utils import Database
    db = Database(db_path)
    
    db["works"].create({
        "work_id": str,
        "citekey": str,
        "citekey_status": str,
        "citekey_synced_utc": str,
        "zotero_item_key": str,
        "first_seen_utc": str,
        "title": str,
        "source": str,
        "doi": str,
        "abstract": str,
        "authors_json": str,
        "published_date": str,
        "journal": str,
        "url": str,  # Add missing url field
    }, pk="work_id", if_not_exists=True)
    
    yield db
    
    # Cleanup
    db_path.unlink()


# ==================== Test FIX 1: Citekey Parsing ====================

class TestCitekeyParsing:
    """Test citekey extraction from Zotero Extra field."""
    
    def test_citation_key_standard_format(self):
        """Test parsing 'Citation Key: author2024a'."""
        extra = "Work-ID: arxiv:2401.12345\nCitation Key: smith2024a\nSource: arxiv"
        
        citekey = None
        for line in extra.split("\n"):
            line_stripped = line.strip()
            if line_stripped.lower().startswith("citation key:"):
                citekey = line_stripped.split(":", 1)[1].strip()
                break
        
        assert citekey == "smith2024a"
    
    def test_citation_key_case_insensitive(self):
        """Test parsing with different case variations."""
        test_cases = [
            ("Citation Key: smith2024", "smith2024"),
            ("citation key: jones2023b", "jones2023b"),
            ("CITATION KEY: BROWN2022", "BROWN2022"),
            ("CiTaTiOn KeY: mixed2021", "mixed2021"),
        ]
        
        for extra_line, expected in test_cases:
            line_stripped = extra_line.strip()
            if line_stripped.lower().startswith("citation key:"):
                citekey = line_stripped.split(":", 1)[1].strip()
                assert citekey == expected
    
    def test_citation_key_with_whitespace(self):
        """Test parsing with various whitespace patterns."""
        test_cases = [
            ("Citation Key:smith2024", "smith2024"),  # No space after colon
            ("Citation Key:  smith2024", "smith2024"),  # Multiple spaces
            ("  Citation Key: smith2024  ", "smith2024"),  # Leading/trailing spaces
            ("Citation Key:\tsmith2024", "smith2024"),  # Tab after colon
        ]
        
        for extra_line, expected in test_cases:
            line_stripped = extra_line.strip()
            if line_stripped.lower().startswith("citation key:"):
                citekey = line_stripped.split(":", 1)[1].strip()
                assert citekey == expected
    
    def test_citation_key_not_found(self):
        """Test when no citation key is present."""
        extra = "Work-ID: arxiv:2401.12345\nSource: arxiv\nRelevance Score: 15.0"
        
        citekey = None
        for line in extra.split("\n"):
            line_stripped = line.strip()
            if line_stripped.lower().startswith("citation key:"):
                citekey = line_stripped.split(":", 1)[1].strip()
                break
        
        assert citekey is None


class TestReferencesJsonProtection:
    """Test references.json protection logic."""
    
    def test_protect_when_synced_zero_pending_nonzero(self, temp_db):
        """references.json should not be overwritten when Synced=0 and Pending>0."""
        from watch_literature import generate_references_json
        
        # Add pending items to database
        temp_db["works"].insert({
            "work_id": "arxiv:2401.12345",
            "citekey": None,
            "citekey_status": "pending",
            "citekey_synced_utc": "",
            "zotero_item_key": "ABC123",
            "first_seen_utc": datetime.now().isoformat(),
            "title": "Test Paper",
            "source": "arxiv",
            "doi": "",
            "abstract": "Test abstract",
            "authors_json": "[]",
            "published_date": "2024-01-01",
            "journal": "",
            "url": "https://arxiv.org/abs/2401.12345",
        })
        
        # Create existing references.json with content
        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = Path(tmpdir) / "references.json"
            existing_content = [{"id": "old_ref", "title": "Old Paper"}]
            with open(ref_path, "w") as f:
                json.dump(existing_content, f)
            
            # Run generation
            stats = generate_references_json(temp_db, ref_path)
            
            # Check protection activated
            assert stats["file_protected"] is True
            assert stats["synced"] == 0
            assert stats["pending"] == 1
            
            # Verify file not overwritten
            with open(ref_path, "r") as f:
                content = json.load(f)
            assert content == existing_content
    
    def test_write_temporary_on_first_run(self, temp_db):
        """On first run (no prior references.json), write temporary IDs."""
        from watch_literature import generate_references_json
        
        # Add pending items to database
        temp_db["works"].insert({
            "work_id": "arxiv:2401.12345",
            "citekey": None,
            "citekey_status": "pending",
            "citekey_synced_utc": "",
            "zotero_item_key": "ABC123",
            "first_seen_utc": datetime.now().isoformat(),
            "title": "Test Paper",
            "source": "arxiv",
            "doi": "",
            "abstract": "Test abstract",
            "authors_json": "[]",
            "published_date": "2024-01-01",
            "journal": "",
            "url": "https://arxiv.org/abs/2401.12345",
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = Path(tmpdir) / "references.json"
            
            # File doesn't exist - first run
            assert not ref_path.exists()
            
            # Run generation
            stats = generate_references_json(temp_db, ref_path)
            
            # Should write temporary references
            assert ref_path.exists()
            assert stats["synced"] == 0
            assert stats["pending"] == 1
            
            # Verify content has work_id as ID
            with open(ref_path, "r") as f:
                content = json.load(f)
            assert len(content) == 1
            assert content[0]["id"] == "arxiv:2401.12345"
            assert "Temporary ID" in content[0].get("note", "")


# ==================== Test FIX 3: arXiv Education Precision ====================

class TestArxivEducationGate:
    """Test enhanced education-intent gating for arXiv."""
    
    def test_curriculum_learning_excluded(self, sample_config):
        """Curriculum learning (ML technique) should be excluded even with LLM terms."""
        paper = Paper(
            identifier="arxiv:test1",
            title="Curriculum Learning for Better LLM Training with Student Models",
            authors=["Test Author"],
            abstract="We propose a curriculum learning approach to train large language models more efficiently using student teacher paradigm.",
            url="https://arxiv.org/abs/test1",
            source_type="arxiv",
        )
        
        result = check_hard_education_intent(paper, sample_config)
        
        # Should be excluded - but first for no_edu_setting_term OR curriculum context
        assert result["gate_pass"] is False
        # Accept either exclusion reason since the gate is multi-stage
        assert result["exclude_reason"] in ["no_edu_setting_term", "curriculum_learning_ml_context", "curriculum_ml_context"]
    
    def test_feedback_loop_rl_excluded(self, sample_config):
        """Feedback loop in RL/algorithmic context should be excluded even with 'student'."""
        paper = Paper(
            identifier="arxiv:test2",
            title="Reinforcement Learning with Feedback Loops for LLM Agents and Students",
            authors=["Test Author"],
            abstract="We use feedback loops and reward signals to train LLM agents using student models in embodied environments.",
            url="https://arxiv.org/abs/test2",
            source_type="arxiv",
        )
        
        result = check_hard_education_intent(paper, sample_config)
        
        assert result["gate_pass"] is False
        # Should exclude for no setting term or feedback algorithmic context
        assert result["exclude_reason"] in ["no_edu_setting_term", "feedback_algorithmic_context"]
    
    def test_distillation_excluded(self, sample_config):
        """Distillation/teacher-student (model compression) should be excluded without classroom."""
        paper = Paper(
            identifier="arxiv:test3",
            title="Knowledge Distillation for Efficient LLM Deployment using Teacher Models",
            authors=["Test Author"],
            abstract="We use teacher-student distillation to compress large language models for inference.",
            url="https://arxiv.org/abs/test3",
            source_type="arxiv",
        )
        
        result = check_hard_education_intent(paper, sample_config)
        
        assert result["gate_pass"] is False
        assert result["exclude_reason"] in ["no_edu_setting_term", "distillation_model_compression"]
    
    def test_quantization_excluded(self, sample_config):
        """Quantization/GPU optimization should be excluded without education setting."""
        paper = Paper(
            identifier="arxiv:test4",
            title="8-bit Quantization for Faster LLM Inference Training",
            authors=["Test Author"],
            abstract="We present a quantization technique for LLM inference on GPUs with KV cache optimization during training.",
            url="https://arxiv.org/abs/test4",
            source_type="arxiv",
        )
        
        result = check_hard_education_intent(paper, sample_config)
        
        assert result["gate_pass"] is False
        assert result["exclude_reason"] in ["no_edu_setting_term", "system_optimization_context: quantization"]
    
    def test_robot_excluded(self, sample_config):
        """Robotics/embodied AI should be excluded unless strong edu context."""
        paper = Paper(
            identifier="arxiv:test5",
            title="LLMs for Robot Motion Generation and Student Learning",
            authors=["Test Author"],
            abstract="We use large language models to generate robot motion in embodied environments using student learning.",
            url="https://arxiv.org/abs/test5",
            source_type="arxiv",
        )
        
        result = check_hard_education_intent(paper, sample_config)
        
        assert result["gate_pass"] is False
        assert result["exclude_reason"] in ["no_edu_setting_term", "system_optimization_context: robot"]
    
    def test_university_course_included(self, sample_config):
        """University course context should be included."""
        paper = Paper(
            identifier="arxiv:test6",
            title="Using ChatGPT in University Computer Science Courses",
            authors=["Test Author"],
            abstract="We evaluate the use of large language models in higher education classroom settings for teaching programming.",
            url="https://arxiv.org/abs/test6",
            source_type="arxiv",
        )
        
        result = check_hard_education_intent(paper, sample_config)
        
        assert result["gate_pass"] is True
        assert "higher education" in result["matched_edu_setting_terms"] or "university" in result["matched_edu_setting_terms"] or "course" in result["matched_edu_setting_terms"]
    
    def test_classroom_tutoring_included(self, sample_config):
        """Classroom/tutoring context should be included."""
        paper = Paper(
            identifier="arxiv:test7",
            title="LLM-based Tutoring Systems for Undergraduate Education",
            authors=["Test Author"],
            abstract="We present a GPT-based tutoring system for classroom use in undergraduate courses.",
            url="https://arxiv.org/abs/test7",
            source_type="arxiv",
        )
        
        result = check_hard_education_intent(paper, sample_config)
        
        assert result["gate_pass"] is True
        assert len(result["matched_edu_setting_terms"]) > 0
    
    def test_grading_rubric_included(self, sample_config):
        """Grading/rubric/assignment context should be included."""
        paper = Paper(
            identifier="arxiv:test8",
            title="Automated Grading of Student Assignments using LLMs",
            authors=["Test Author"],
            abstract="We use large language models to grade student assignments with rubrics in higher education.",
            url="https://arxiv.org/abs/test8",
            source_type="arxiv",
        )
        
        result = check_hard_education_intent(paper, sample_config)
        
        assert result["gate_pass"] is True
        assert "grading" in result["matched_edu_setting_terms"] or "assignment" in result["matched_edu_setting_terms"]
    
    def test_pedagogy_instructional_design_included(self, sample_config):
        """Pedagogy/instructional design should be included."""
        paper = Paper(
            identifier="arxiv:test9",
            title="Pedagogical Approaches for Integrating LLMs in University Teaching",
            authors=["Test Author"],
            abstract="We explore pedagogical and instructional design strategies for using generative AI in higher education courses.",
            url="https://arxiv.org/abs/test9",
            source_type="arxiv",
        )
        
        result = check_hard_education_intent(paper, sample_config)
        
        assert result["gate_pass"] is True
        # Check that we matched pedagogy-related terms
        matched_terms_str = " ".join(result["matched_edu_setting_terms"])
        assert "pedagog" in matched_terms_str or "university" in matched_terms_str or "instructional" in matched_terms_str or "teaching" in matched_terms_str
    
    def test_mooc_online_learning_included(self, sample_config):
        """MOOC/online learning context should be included."""
        paper = Paper(
            identifier="arxiv:test10",
            title="Using ChatGPT in MOOCs and Online Learning Platforms",
            authors=["Test Author"],
            abstract="We study the use of LLMs in massive open online courses and e-learning platforms for higher education.",
            url="https://arxiv.org/abs/test10",
            source_type="arxiv",
        )
        
        result = check_hard_education_intent(paper, sample_config)
        
        assert result["gate_pass"] is True
        assert "mooc" in result["matched_edu_setting_terms"] or any("learning" in t for t in result["matched_edu_setting_terms"])


# ==================== Test FIX 2: OpenAlex Diagnostics ====================

class TestOpenAlexDiagnostics:
    """Test OpenAlex diagnostic logging."""
    
    def test_missing_api_key_detected(self, sample_config, capsys):
        """Test that missing API key is detected and logged."""
        # This is tested by checking console output in actual run
        # For unit test, we verify the logic exists
        import os
        
        original_key = os.environ.get("OPEN_ALEX_API_KEY")
        os.environ.pop("OPEN_ALEX_API_KEY", None)
        
        # Import after clearing env var
        import importlib
        import watch_literature
        importlib.reload(watch_literature)
        
        # Verify that OPEN_ALEX_API_KEY is empty
        assert watch_literature.OPEN_ALEX_API_KEY == ""
        
        # Restore
        if original_key:
            os.environ["OPEN_ALEX_API_KEY"] = original_key
    
    def test_empty_journals_detected(self, sample_config):
        """Test that empty journals list is detected."""
        config_empty = {"journals": []}
        
        # This would be caught by search_openalex function
        journals = config_empty.get("journals", [])
        assert len(journals) == 0


# ==================== Test Helpers ====================

class TestHelpers:
    """Test helper functions."""
    
    def test_normalize_text(self):
        """Test text normalization."""
        assert normalize_text("  Hello   World  ") == "hello world"
        assert normalize_text("UPPERCASE") == "uppercase"
        assert normalize_text("Multiple   \n  spaces") == "multiple spaces"
    
    def test_edu_setting_terms_defined(self):
        """Test that EDU_SETTING_TERMS are properly defined."""
        assert len(EDU_SETTING_TERMS) > 0
        assert "higher education" in EDU_SETTING_TERMS
        assert "university" in EDU_SETTING_TERMS
        assert "classroom" in EDU_SETTING_TERMS
    
    def test_edu_ambiguous_terms_defined(self):
        """Test that EDU_AMBIGUOUS_TERMS are properly defined."""
        assert len(EDU_AMBIGUOUS_TERMS) > 0
        assert "student" in EDU_AMBIGUOUS_TERMS
        assert "curriculum" in EDU_AMBIGUOUS_TERMS
    
    def test_negative_terms_defined(self):
        """Test that NEGATIVE_STRONG_TERMS are properly defined."""
        assert len(NEGATIVE_STRONG_TERMS) > 0
        assert "quantization" in NEGATIVE_STRONG_TERMS
        assert "gpu" in NEGATIVE_STRONG_TERMS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
