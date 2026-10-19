"""Phase 1b — Cleaning.  🧑‍💻 YOUR ASSIGNMENT

Why this matters: "garbage in, garbage out" is *literally* true for RAG. If nav menus
("Skip to main content / Home / About / Sitemap ...") stay in the text, they get chunked,
embedded, and retrieved — they look similar to EVERY query about that website, so they
pollute the top-k and waste context tokens.

Implement the four functions marked TODO until `uv run pytest tests/test_clean.py` passes.
`clean_document` (the orchestrator at the bottom) is already wired up.

Real examples from the data you should handle:
  - Page text beginning: "High Commission of India...\\nSitemap\\nContact\\nFeedback\\n
    Skip to main content\\nHigh Commission of India\\nWellington...\\nWhat's New\\nSearch\\nHome..."
  - PDF titles like "/pdf/hostel/hostel%20rules%20%20and%20application%20form%20new.pdf"
  - Repeated header lines: "About University\\nAbout University\\nAbout University"
  - OCR garbage: "é\\n( Government.of Punjab ; :\\nvw Department ... ee ५१७४ (Forest Branch)"
"""

from __future__ import annotations

import re, os
from dataclasses import replace
from urllib.parse import unquote, urlparse

# pyrefly: ignore [missing-import]
from rag.ingest.lang import coarse_lang, indic_script_ratio
from rag.types import Document

# Lines that are pure website chrome. Matched case-insensitively against a *whole stripped line*.
# Feel free to extend this after eyeballing more data (that's part of the exercise!).
BOILERPLATE_LINES = {
    "skip to main content", "skip to content", "skip to navigation", "sitemap", "site map",
    "home", "contact", "contact us", "feedback", "search", "menu", "login", "sign in",
    "screen reader access", "a-", "a", "a+", "-a", "+a", "english", "हिन्दी", "हिंदी",
    "what's new", "faq", "faqs", "help", "disclaimer", "privacy policy", "terms of use",
    "terms & conditions", "copyright policy", "hyperlink policy", "accessibility statement",
    "back to top", "print", "share", "close", "read more", "view all", "more",
}

PAGE_N_RE = re.compile(r"^page\s*\d+(\s*(of|/)\s*\d+)?$", re.IGNORECASE)
LATIN_RE = re.compile(r"[A-Za-z]")
INDIC_RE = re.compile(r"[\u0900-\u0DFF]")


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\t", " ")
    text = re.sub(r" {2,}", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
 
 
def strip_boilerplate(text: str) -> str:
    kept = []
    prev = None
    for line in text.split("\n"):
        key = line.strip()
        if key.lower() in BOILERPLATE_LINES:
            continue
        if key and key == prev:
            continue
        kept.append(line)
        prev = key
    return "\n".join(kept)
 
 
def _is_bad_title(title: str) -> bool:
    t = title.strip()
    return (
        not t
        or t.startswith("/")
        or t.lower().endswith((".pdf", ".doc", ".docx"))
        or "%20" in t
    )
 
 
def fix_title(title: str, text: str, url: str) -> str:
    if not _is_bad_title(title):
        return title.strip()
 
    for line in text.split("\n"):
        line = line.strip()
        if 4 <= len(line.split()) and len(line) <= 120 and not PAGE_N_RE.match(line):
            return line
 
    filename = os.path.basename(unquote(urlparse(url).path))
    stem, _ext = os.path.splitext(filename)
    stem = re.sub(r"[_\-]+", " ", stem)
    return re.sub(r"\s+", " ", stem).strip()
 
 
def _is_junk(token: str) -> bool:
    if len(token) == 1 and not token.isalnum():
        return True
    if LATIN_RE.search(token) and INDIC_RE.search(token):
        return True
    if not any(c.isalnum() for c in token):
        return True
    return False
 
 
def ocr_noise_score(text: str, sample_chars: int = 5000) -> float:
    tokens = text[:sample_chars].split()
    if not tokens:
        return 0.0
    return sum(_is_junk(t) for t in tokens) / len(tokens)


# ---------------------------------------------------------------------------
# Orchestrator (already implemented) 🤖
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+")


def clean_document(doc: Document) -> Document:
    text = normalize_whitespace(strip_boilerplate(doc.text))
    meta = {
        **doc.metadata,
        "lang": coarse_lang(text),
        "indic_ratio": round(indic_script_ratio(text), 3),
        "ocr_score": round(ocr_noise_score(text), 3),
        "n_chars_raw": doc.metadata.get("n_chars", len(doc.text)),
        "n_chars": len(text),
    }
    return replace(doc, text=text, title=fix_title(doc.title, text, doc.url), metadata=meta)
