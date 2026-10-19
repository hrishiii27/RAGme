# RAGme — Project Memory

> Long-lived context for this project. Holds the **what & why**: goals, constraints, decisions, and facts about the data.
> Tasks live in [notes.md](./notes.md). Open questions live in [questions.md](./questions.md).
> Last updated: 2026-10-06

---

## 1. Who / Why
- **Owner:** Hrishikesh. He is learning RAG by building a real project.
- **My role:** Senior Software Developer specialising in AI systems, agents and RAG techniques. I act as mentor and pair programmer.
- **Main goal: learning.** Understand every phase of RAG properly, and compare the options at each phase with evidence. A working product matters less than that.
- **Product framing:** An enterprise-style RAG website for **Indian public / government information**. It should be a **single source of truth**, so users don't have to search across many government websites.

## 2. The RAG pipeline (each phase is studied on its own)
```
Document Ingestion → Chunking → Embedding → Indexing/Search → Retrieval (+ rerank)
      → Prompt Assembly (curated system prompt + context + user query) → LLM → Answer (+ citations)
```
Each phase needs to be **pluggable**, so that different strategies can be swapped in and benchmarked.

## 3. Experimentation goals (the core of the project)
| Phase | What we compare |
|---|---|
| Chunking | Fixed-size (tokens/chars), recursive, sentence/paragraph, semantic, document-structure aware (headings/pages), parent-child / small-to-big, late chunking, contextual chunking (Anthropic-style) |
| Embedding | Open-source vs API models; multilingual vs English-only; dimension and quantization trade-offs |
| Search | Dense (vector), sparse (BM25), hybrid (RRF / weighted), metadata filtering, query rewriting / HyDE / multi-query, reranking (cross-encoder) |
| Retrieval | top-k, MMR (diversity), parent-document expansion, context compression |
| Prompts | Several system-prompt variants, evaluated against each other; the best one is promoted |
| LLM eval | Analyze → Measure → Improve loop. Includes **human evaluation** (details still to be discussed) |

The final output is a **comprehensive comparison report** for each phase.

## 4. Tech direction (high-level, agreed so far)
- **Backend:** Python + **FastAPI**. Handles document processing, the retrieval pipeline and agent orchestration.
- **Frontend:** a simple, modern web UI. *Design will be discussed later.*
- **Vector DB:** Chroma (embedded mode), moving to Docker later. **Sparse:** BM25, hand-rolled or `rank_bm25`.
- **LLMs:** Gemini Flash (main + judge), OpenRouter open models (cross-check), Ollama ~3B (local iteration).
- **Embedders to compare:** MiniLM, bge-small, multilingual-e5-small, Gemini embedding.
- **Eval / observability:** Ragas + custom harness + LLM-judge calibrated against human grades; Langfuse Cloud.
- **Hardware:** MacBook Air M2, **8 GB RAM**. Keep local models small and avoid running too much at once.
- Full plan: [plan.md](./plan.md)

## 5. Dataset facts — `ankitjh4/bharat-government-documents`
Source: https://huggingface.co/datasets/ankitjh4/bharat-government-documents ("Bharat Guide")

**Size**
- **64,964** distinct normalized text bodies (Parquet `train` split), from 77,526 source records.
- About **1.04 GB** of raw text (≈320 MB download). That is roughly **250M+ tokens** in total.
- **16,122** distinct hostnames: 11.6k `.gov.in`, 1.8k `.ac.in`, 2.3k other `.in`.
- There is also a bigger `documents.csv.gz` with `good` + `review` rows. The Parquet split holds only screened `good` rows.

**Parquet schema**
`content_hash, document_id, title, text, url, kind, trust, quality, fetched_at, assessment_version, topics[list], sources[list of {document_id,url,title,kind,trust,quality,fetched_at,assessment_status,assessment_reasons,assessment_version}]`

**Profile from a 360-row sample (my analysis)**
| Field | Observed values |
|---|---|
| `kind` | page 68%, pdf 24%, scheme 7.5%, docx <1% |
| `trust` | official_domain 71%, directory_verified 21%, official_catalog 7.5% |
| `quality` | full_text 89%, **ocr_text 11%** (noisy) |
| text length (chars) | min 573 · p10 ~1k · **median ~3.6k** · mean ~15k · p90 ~17k · **max ~909k** (very long tail) |
| language | about **14% mostly Indic script** (Hindi etc.). Also mixed Hindi+English OCR |
| topics | schemes, universities, legal_judiciary, public_services, policy, public_finance, infra, culture, tourism, banking… (36 labels; a doc can have several) |

**Data-quality issues that drive design decisions**
1. **HTML boilerplate.** Page text starts with navigation menus ("Skip to main content, Sitemap, Home, About…"). This needs cleaning, or chunking will index junk.
2. **OCR noise.** Some text is garbled and mixes Devanagari and Latin, e.g. `ee ५१७४ (Forest Branch)`.
3. **PDF titles are often file paths**, e.g. `/pdf/hostel/...new.pdf`. A better title has to be derived, from the first heading or the URL.
4. **Huge length variance.** One chunking setting won't fit everything; some documents are 900k chars.
5. **Multilingual.** This affects the embedding model choice and BM25 tokenization.
6. **Forms / directories.** Low-information content such as blank application forms and contact lists.
7. **Not fact-checked, and fetch time ≠ publication time.** Answers should cite sources and encourage verification.
8. **Rights:** "source-specific rights", with no blanket licence. Fine for personal learning; we would need to check before any public deployment.

**Useful metadata for filtering and boosting:** `topics`, `kind`, `trust`, `quality`, `url` host (state/ministry), `fetched_at`.

## 6. Decisions log
| Date | Decision | Why |
|---|---|---|
| 2026-10-06 | FastAPI backend, simple modern frontend | User requirement |
| 2026-10-06 | Each pipeline phase is a pluggable module with its own benchmark | Comparison is the main goal |
| 2026-10-06 | Working mode: I scaffold, the user implements the core RAG pieces | Learning goal (A1) |
| 2026-10-06 | Weekend sprint → portfolio piece (GitHub + demo) | A2/A3 |
| 2026-10-06 | Experiments on a stratified 5–8k doc subset; full corpus only with the winning config | B1, plus laptop limits |
| 2026-10-06 | English first; keep an Indic holdout for a later multilingual sprint; all 36 topics | B2–B4 |
| 2026-10-06 | Gemini (API) + OpenRouter (open models) + small local Ollama model | C1–C3 |
| 2026-10-06 | Chroma first (embedded mode), Docker learned later (Chroma server) | D1 |
| 2026-10-06 | From scratch in plain Python; LangChain used selectively for learning; **no LangGraph** | D2, G1 |
| 2026-10-06 | React (Vite) frontend; Ragas for eval; Langfuse for tracing | D3, D4, F3 |
| 2026-10-06 | Single-shot RAG with multi-turn memory (condense step); agents later | E1, E2 |
| 2026-10-06 | General-public audience; refuse + suggest where to look when the answer isn't in the corpus; no auth | E3–E5 |
| 2026-10-06 | Eval starts with 10 hand-graded seed Qs (model vs human critique/outcome, alignment, agreement), then expands with synthetic Qs; ~50 human judgements by the user | F1, F2 |
| 2026-10-06 | Priority: correctness > faithfulness > citation quality > completeness; latency/cost secondary | F3 |
| 2026-10-11 | Model answers should include context-relevant, accurate follow-up questions for the user | UX / Generation feature |
| 2026-10-11 | Human-AI alignment dataset expanded to 100 QA pairs adhering to seed grading rubric | Golden set curation |
| 2026-10-11 | All experimental benchmarks tracked with exact numbers in reports/benchmark_master.json and md | Benchmarking integrity |
| 2026-10-11 | Model upgraded to gemini-3.1-pro-preview with automatic fallback to gemini-2.5-flash on rate limits | Generation model upgrade |
| 2026-10-11 | Corpus expanded from 6,000 to 15,000 documents (159,977 chunks in Chroma A15000-recursive-1000-100) | Corpus scaling (2.5x) |

## 7. Glossary (for learning)
- **Chunking:** splitting documents into retrievable units.
- **Embedding:** turning text into a vector that captures its meaning.
- **BM25:** keyword-based (sparse) ranking.
- **Hybrid search:** dense + sparse search, with results fused (e.g. RRF = Reciprocal Rank Fusion).
- **Reranker:** a cross-encoder that re-scores the top-N candidates for precision.
- **Recall@k / MRR / nDCG:** retrieval metrics.
- **Faithfulness / Answer relevance / Context precision:** generation metrics (e.g. RAGAS).
- **Golden set:** a curated list of question → expected-answer/source pairs used for evaluation.
