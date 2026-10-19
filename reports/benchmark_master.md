# Master Benchmark Comparisons (Exact Quantitative Metrics)

This master document records all verified benchmarks conducted across the RAG pipeline development.

---

## 1. Stage A: Chunking Strategy Benchmark (6,000 Documents)

| Config | Chunks | Indexing Time (s) | Recall@5 | Recall@10 | Hit@1 | MRR | nDCG@10 | Outcome |
|---|---|---|---|---|---|---|---|---|
| `fixed-256-32` | 70,679 | 371s | 0.7667 | 0.8267 | 0.5600 | 0.6620 | 0.6998 | Baseline |
| `fixed-512-64` | 36,483 | 219s | 0.7267 | 0.7600 | 0.5067 | 0.5964 | 0.6343 | Baseline |
| `recursive-1000-100` | 64,147 | 372s | 0.7733 | 0.8400 | 0.5733 | 0.6723 | 0.7095 | **WINNER** |
| `recursive-1500-150` | 43,532 | 243s | 0.7267 | 0.7867 | 0.5600 | 0.6387 | 0.6713 | Baseline |

---

## 2. Stage C: Search & Retrieval Strategy Benchmark

| Strategy | Search Latency (s) | Recall@5 | Recall@10 | Hit@1 | MRR | nDCG@10 | Outcome |
|---|---|---|---|---|---|---|---|
| `dense-only` | 1.0s | 0.7733 | 0.8400 | 0.5733 | 0.6723 | 0.7095 | Baseline |
| `bm25-only` | 10.6s | 0.9467 | 0.9600 | 0.8067 | 0.8612 | 0.8845 | Baseline |
| `hybrid-rrf` | 10.8s | 0.9333 | 0.9600 | 0.7267 | 0.8190 | 0.8539 | **WINNER (Chosen)** |
| `hybrid-rrf-mmr` | 12.1s | 0.8067 | 0.9667 | 0.5733 | 0.6781 | 0.7460 | Top Recall |

---

## 3. Stage E: Human Seed Grading & Calibration Benchmark

- **Seed Evaluation Set:** `data/golden/seed_grading.csv` (20 rows hand-reviewed by human)
- **Exact Agreement:** 80.0% (16/20)
- **Pass / Not-Pass Agreement:** 80.0% (16/20)
- **Cohen's Kappa (Inter-Rater Reliability):** $\kappa = 0.29$
- **Calibration Impact:** Base LLM judge was too lenient on polite refusals without actionable redirection. Calibrated rubric penalizes vague refusals and mandates specific portal/department names.

---

## 4. Stage F: Prompt Tournament Leaderboard (Human-Aligned Benchmark)

| Technique | Pass Rate | Correctness (/5) | Faithfulness (/5) | Citations (/5) | Redirection (/5) | Follow-ups (/5) | Composite (/25) |
|---|---|---|---|---|---|---|---|
| Zero-Shot | 92.0% | 4.92 | 5.00 | 5.00 | 4.88 | 4.92 | 24.72 |
| Few-Shot (Learned) | 100.0% | 5.00 | 5.00 | 5.00 | 4.88 | 4.96 | 24.84 |
| Hidden CoT | 92.0% | 5.00 | 5.00 | 4.92 | 4.76 | 5.00 | 24.68 |
| Few-Shot + CoT | 96.0% | 4.88 | 4.92 | 5.00 | 4.92 | 4.92 | 24.64 |

---

## Auditability & Granular Verification Files
- Stage A Raw Data: `reports/stage_A.json`
- Stage C Raw Data: `reports/stage_C.json`
- Seed Human Grading: `data/golden/seed_grading.csv`
- Aligned 100-QA Dataset: `data/golden/aligned_100_dataset.csv`
- Tournament Summary: `reports/prompt_tournament.json`
- Tournament Per-Question Log: `reports/prompt_tournament_details.json`