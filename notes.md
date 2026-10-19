# RAGme — Notes & Actionables

> Living task tracker. Legend: `[x]` done · `[/]` in progress · `[ ]` todo · `[?]` blocked on a decision · 🧑‍💻 you · 🤖 me
> Context: [memory.md](./memory.md) · Plan: [plan.md](./plan.md) · Answers: [questions.md](./questions.md)

---

## Phase 0 — Discovery & Planning
- [x] Capture context → `memory.md`
- [x] Profile dataset (360-row sample)
- [x] Clarifying questions answered → `questions.md`
- [x] Implementation plan v1 → `plan.md`
- [?] **Resolve brainstorm items B1–B6 in plan.md §7** ← *current blocker*

## Weekend Day 1 — Retrieval
- [ ] 🤖 Repo scaffold (uv, Python 3.10+, pytest, pre-commit, `.env.example`, `.gitignore`, GitHub repo)
- [ ] 🤖 Parquet download → `data/raw/`
- [x] 🧑‍💻 Cleaning rules (boilerplate strip, title fix, lang detect, OCR score) + tests
- [x] 🤖 Stratified ~6k English subset + ~500 Indic holdout + ingestion report
- [x] 🧑‍💻 Chunkers: fixed, recursive, sentence-window, semantic, parent-child, title-header + tests
- [x] 🤖 Embedders (sentence-transformers, Gemini) + disk cache
- [x] 🤖 Chroma store (one collection per config)
- [x] 🧑‍💻 BM25 + RRF hybrid + MMR
- [x] 🤖 Synthetic golden set (doc-level labels) · 🧑‍💻 review ~30
- [x] 🧑‍💻 Metrics: Recall@k, MRR, nDCG
- [x] 🤖 Experiment runner + reports · run Stages A (chunking) → B (embedding) → C (search)

## Weekend Day 2 — Generation, Eval, App
- [x] 🤖 LLM clients: Gemini, OpenRouter, Ollama (with dynamic free model discovery)
- [x] 🧑‍💻 Condense-question step + prompt builder + 3–4 system-prompt variants (include context-relevant follow-up suggestions)
- [x] 🧑‍💻 Write the 10 seed questions + expected behaviour
- [x] 🤖 LLM judge + rubric · 🧑‍💻 human grading · 🤖 agreement / κ calculation
- [x] 🤖 Stage D: Ragas + judge across prompt variants → promote the winner
- [x] 🤖 FastAPI `/chat` (SSE), `/search`, `/feedback` + Langfuse tracing
- [x] 🤖 React (Vite) chat UI with citations (+ America.gov theme, benchmark modal, source drawer)
- [x] 🤖 Comparison report draft · 🧑‍💻 learnings section · README (master benchmarks consolidated)

## Follow-up sprints
- [x] Docker: Chroma server, docker-compose, multi-stage Nginx + uv images
- [x] Full-corpus index scaled to 681,220 vectors across 46,690 official documents synced to Pinecone Serverless Cloud (maximized to Pinecone 2.0M monthly write unit tier limit)
- [x] Multilingual: Hindi queries, multilingual embedder, Indic holdout eval (see reports/multilingual_benchmark.md)
- [x] Advanced chunking: contextual retrieval, late chunking (see reports/contextual_retrieval_benchmark.md)
- [x] LangChain LCEL re-implementation + comparison (see reports/langchain_lcel_comparison.md)
- [x] Agentic retrieval / routing (plain Python) (see reports/agentic_retrieval_benchmark.md)
- [x] Expand human eval to ~50 judgements; failure-mode taxonomy in Langfuse (see reports/human_eval_50_failure_taxonomy.md)
- [x] Frontend design pass + demo deploy (21st.dev components & animations, production build verified)

---

## Log
- **2026-10-06 (1):** Kickoff. Dataset profiled: 65k docs / ~1 GB, heavy-tailed lengths, 11% OCR, ~14% Indic, lots of nav boilerplate.
- **2026-10-06 (2):** Answers received. Locked: weekend sprint, portfolio deliverable, 8 GB M2 Air, Gemini + OpenRouter + Ollama, Chroma, from scratch (no LangGraph), React/Vite, Ragas, Langfuse, multi-turn, English first. Wrote plan.md v1 with a staged-sweep experiment design and a judge-calibration eval design.
