# Multilingual Retrieval Benchmark

Corpus: golden EN docs + 1000 EN distractors + 1,000 Indic docs → **20,541 chunks** (recursive 1000/100). Labels are doc-level.

Query sets: `en` (150), `hi` (150), `hinglish` (150), `hi_native` (100)

## recall@10

| retriever | en | hi | hinglish | hi_native |
|---|---:|---:|---:|---:|
| bm25 | 0.980 | 0.247 | 0.980 | 0.740 |
| minilm | 0.927 | 0.133 | 0.787 | 0.000 |
| e5 | 0.980 | 0.800 | 0.933 | 0.950 |
| hybrid_minilm | 0.973 | 0.247 | 0.967 | 0.690 |
| hybrid_e5 | 0.980 | 0.760 | 0.980 | 0.890 |
| tr_hybrid | 0.973 | 0.967 | 0.973 | 0.220 |

## mrr

| retriever | en | hi | hinglish | hi_native |
|---|---:|---:|---:|---:|
| bm25 | 0.922 | 0.181 | 0.919 | 0.646 |
| minilm | 0.801 | 0.098 | 0.609 | 0.002 |
| e5 | 0.914 | 0.653 | 0.830 | 0.815 |
| hybrid_minilm | 0.902 | 0.199 | 0.841 | 0.627 |
| hybrid_e5 | 0.930 | 0.386 | 0.919 | 0.719 |
| tr_hybrid | 0.902 | 0.893 | 0.901 | 0.130 |

## hit@1

| retriever | en | hi | hinglish | hi_native |
|---|---:|---:|---:|---:|
| bm25 | 0.880 | 0.147 | 0.880 | 0.600 |
| minilm | 0.727 | 0.087 | 0.493 | 0.000 |
| e5 | 0.880 | 0.567 | 0.773 | 0.750 |
| hybrid_minilm | 0.853 | 0.173 | 0.767 | 0.600 |
| hybrid_e5 | 0.900 | 0.207 | 0.880 | 0.630 |
| tr_hybrid | 0.853 | 0.847 | 0.860 | 0.100 |

## indic_share@10

| retriever | en | hi | hinglish | hi_native |
|---|---:|---:|---:|---:|
| bm25 | 0.067 | 0.925 | 0.080 | 0.984 |
| minilm | 0.112 | 0.895 | 0.182 | 0.977 |
| e5 | 0.083 | 0.769 | 0.447 | 0.952 |
| hybrid_minilm | 0.090 | 0.904 | 0.127 | 0.987 |
| hybrid_e5 | 0.072 | 0.824 | 0.258 | 0.976 |
| tr_hybrid | 0.090 | 0.074 | 0.086 | 0.137 |

`indic_share@10` = fraction of the top-10 docs that come from the Indic holdout. For `en`/`hi`/`hinglish` the right answer is always English, so a high share means the retriever is matching on *script*, not *meaning*.
