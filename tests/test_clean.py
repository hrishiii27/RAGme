"""Spec for rag.ingest.clean — make these pass. Run: uv run pytest tests/test_clean.py"""

from rag.ingest.clean import (
    clean_document,
    fix_title,
    normalize_whitespace,
    ocr_noise_score,
    strip_boilerplate,
)
from rag.types import Document

# ---------- normalize_whitespace ----------

def test_collapses_spaces_and_nbsp():
    assert normalize_whitespace("a\u00a0\u00a0b \t  c") == "a b c"


def test_strips_lines_and_keeps_paragraphs():
    assert normalize_whitespace("  line one  \n  line two \n\n\n\n\npara two  ") == (
        "line one\nline two\n\npara two"
    )


# ---------- strip_boilerplate ----------

def test_drops_nav_lines():
    raw = "Skip to main content\nHome\nSitemap\nKnow India Programme\nThe KIP is a scheme."
    assert strip_boilerplate(raw) == "Know India Programme\nThe KIP is a scheme."


def test_case_insensitive_nav():
    assert strip_boilerplate("SKIP TO MAIN CONTENT\nReal text") == "Real text"


def test_keeps_sentences_containing_nav_words():
    raw = "Contact the Tehsildar office for details.\nHome guards are recruited yearly."
    assert strip_boilerplate(raw) == raw


def test_collapses_consecutive_duplicates():
    raw = "About University\nAbout University\nAbout University\nBKNMU was established in 2015."
    assert strip_boilerplate(raw) == "About University\nBKNMU was established in 2015."


def test_non_consecutive_duplicates_kept():
    raw = "Eligibility\nAge 18+\nEligibility"
    assert strip_boilerplate(raw) == raw


# ---------- fix_title ----------

def test_good_title_kept():
    assert fix_title("  Know India Programme  ", "text", "https://x.gov.in/a") == "Know India Programme"


def test_pdf_path_title_uses_first_meaningful_line():
    text = "Page 1\n\nApplication for admission to the Govt Engineering College Hostel\n1 Name"
    got = fix_title("/pdf/hostel/hostel%20rules.pdf", text, "https://gecskp.ac.in/pdf/x.pdf")
    assert got == "Application for admission to the Govt Engineering College Hostel"


def test_falls_back_to_url_filename():
    got = fix_title("/media/documents/NotificationBhudaTehHB.9.pdf", "x\ny",
                    "https://forest.punjab.gov.in/media/documents/forest_notification-2019.pdf")
    assert got == "forest notification 2019"


# ---------- ocr_noise_score ----------

def test_clean_english_is_low_noise():
    text = "The Governor of Punjab is pleased to declare the lands as protected forests. " * 20
    assert ocr_noise_score(text) < 0.05


def test_garbled_ocr_is_noisier():
    garbled = "é ( Government.of ; : vw ee ५१७४ कटcy wi Aas i aes १७ — ; : é " * 20
    clean = "The Governor of Punjab is pleased to declare the lands as protected forests. " * 20
    assert ocr_noise_score(garbled) > ocr_noise_score(clean) + 0.2


def test_empty_is_zero():
    assert ocr_noise_score("") == 0.0


# ---------- integration ----------

def test_clean_document_end_to_end():
    doc = Document(
        doc_id="d1",
        title="/pdf/x.pdf",
        text="Skip to main content\nHome\n\n\n\nPradhan Mantri Awas Yojana guidelines\u00a0 2024\nDetails here.",
        url="https://pmay.gov.in/pdf/x.pdf",
        metadata={"n_chars": 90},
    )
    out = clean_document(doc)
    assert out.text.startswith("Pradhan Mantri Awas Yojana guidelines 2024")
    assert out.title == "Pradhan Mantri Awas Yojana guidelines 2024"
    assert out.metadata["lang"] == "en"
    assert out.metadata["n_chars_raw"] == 90
