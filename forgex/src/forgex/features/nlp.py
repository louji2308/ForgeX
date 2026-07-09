from __future__ import annotations

import functools
import re
import subprocess
import sys
import warnings
from typing import Optional

from forgex.errors import FeatureBuildError, MissingDependencyError
from forgex.logging_setup import get_logger

logger = get_logger(__name__)

SEVERITY_KEYWORDS = {
    "critical": {"mold", "no heat", "no hot water", "gas smell", "flooding", "not cooling"},
    "moderate": {"leak", "broken", "not working", "clogged"},
    "minor": {"cosmetic", "squeak", "loose handle"},
}

_VADER_AVAILABLE = False
_VADER_SENTIMENT = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_SENTIMENT = SentimentIntensityAnalyzer()
    _VADER_AVAILABLE = True
except ImportError:
    logger.warning("vaderSentiment not available — sentiment will default to 0.0")


class MaintenanceTextTaggerError(FeatureBuildError):
    """Raised when spaCy-based tagging fails."""


@functools.lru_cache(maxsize=1)
def _load_spacy_model():
    try:
        import spacy
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            logger.info("spaCy model 'en_core_web_sm' not found — attempting auto-download...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                    check=True, capture_output=True, timeout=120,
                )
                return spacy.load("en_core_web_sm")
            except Exception as dl_err:
                logger.warning(f"spaCy auto-download failed ({dl_err}) — will use keyword fallback")
                return None
    except ImportError:
        logger.warning("spaCy not installed — will use keyword fallback")
        return None


def _vader_sentiment(text: str) -> float:
    if not _VADER_AVAILABLE or _VADER_SENTIMENT is None:
        return 0.0
    try:
        scores = _VADER_SENTIMENT.polarity_scores(text)
        return float(scores.get("compound", 0.0))
    except Exception as e:
        logger.warning(f"VADER sentiment failed: {e}")
        return 0.0


def _keyword_severity(text: str) -> str:
    for level, keywords in SEVERITY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return level
    return "minor"


def tag_maintenance_text(text: Optional[str]) -> dict:
    """Never raises on bad input — a malformed complaint string degrades
    to a safe default instead of taking down the whole feature batch."""
    if not isinstance(text, str) or not text.strip():
        return {"severity": "unknown", "sentiment": 0.0, "tag_source": "empty_input_default"}

    text_lower = text.lower().strip()[:2000]

    try:
        nlp = _load_spacy_model()
        if nlp is not None:
            _ = nlp(text_lower)
        return {
            "severity": _keyword_severity(text_lower),
            "sentiment": _vader_sentiment(text_lower),
            "tag_source": "nlp" if nlp is not None else "keyword_fallback",
        }
    except MaintenanceTextTaggerError as e:
        logger.warning(f"spaCy unavailable ({e}); falling back to keyword-only tagging")
        return {
            "severity": _keyword_severity(text_lower),
            "sentiment": 0.0,
            "tag_source": "keyword_fallback",
        }
    except Exception:
        return {
            "severity": _keyword_severity(text_lower),
            "sentiment": _vader_sentiment(text_lower),
            "tag_source": "keyword_fallback",
        }
