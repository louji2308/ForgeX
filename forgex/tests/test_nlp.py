import pytest

from forgex.features.nlp import tag_maintenance_text, SEVERITY_KEYWORDS


def test_tag_maintenance_text_normal():
    result = tag_maintenance_text("There is a leak under the sink")
    assert result["severity"] == "moderate"
    assert isinstance(result["sentiment"], float)
    assert result["tag_source"] in {"nlp", "keyword_fallback"}


def test_tag_maintenance_text_none():
    result = tag_maintenance_text(None)
    assert result["severity"] == "unknown"
    assert result["sentiment"] == 0.0
    assert result["tag_source"] == "empty_input_default"


def test_tag_maintenance_text_empty():
    result = tag_maintenance_text("")
    assert result["severity"] == "unknown"
    assert result["tag_source"] == "empty_input_default"


def test_tag_maintenance_text_critical():
    result = tag_maintenance_text("No heat in the apartment for days, mold growing")
    assert result["severity"] in {"critical", "moderate", "unknown"}


def test_tag_maintenance_text_whitespace():
    result = tag_maintenance_text("   ")
    assert result["severity"] == "unknown"
    assert result["tag_source"] == "empty_input_default"


def test_tag_maintenance_text_pathological_length():
    long_text = "broken " * 5000
    result = tag_maintenance_text(long_text)
    assert result["severity"] in {"moderate", "minor", "unknown"}


def test_severity_keywords_structure():
    for level in ["critical", "moderate", "minor"]:
        assert level in SEVERITY_KEYWORDS
        assert len(SEVERITY_KEYWORDS[level]) > 0
        for kw in SEVERITY_KEYWORDS[level]:
            assert isinstance(kw, str)
