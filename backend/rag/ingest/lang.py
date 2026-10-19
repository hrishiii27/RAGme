"""Cheap script detection. Fast enough to run over all 65k docs during subset selection.

We don't need a full language-ID model yet: the question for this sprint is only
"is this doc mostly Indic script (→ multilingual holdout) or Latin (→ English set)?"
"""

from __future__ import annotations

# Unicode blocks for the major Indic scripts: Devanagari (0900) through Malayalam (0D7F),
# plus Sinhala (0D80–0DFF) to be safe.
_INDIC_START, _INDIC_END = 0x0900, 0x0DFF


def indic_script_ratio(text: str, sample_chars: int = 4000) -> float:
    """Fraction of *letters* in the first `sample_chars` that are in Indic script blocks."""
    sample = text[:sample_chars]
    letters = [c for c in sample if c.isalpha()]
    if not letters:
        return 0.0
    indic = sum(1 for c in letters if _INDIC_START <= ord(c) <= _INDIC_END)
    return indic / len(letters)


def coarse_lang(text: str, threshold: float = 0.3) -> str:
    """'indic' if ≥ threshold of letters are Indic script, else 'en' (Latin-dominant)."""
    return "indic" if indic_script_ratio(text) >= threshold else "en"
