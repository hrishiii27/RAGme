"""Multilingual sprint — cross-lingual retrieval benchmark.

Question: when a citizen asks in Hindi or Hinglish, does our English-tuned pipeline still find
the right document? And what is the cheapest fix?

Corpus (local, does NOT touch the production Pinecone index):
  - every English doc referenced by the golden set
  - N random English distractors
  - all 1,000 Indic holdout docs (so script/language confusion can show up)
  chunked with the production chunker (recursive 1000/100)

Retrievers compared:
  bm25            lexical (tokenizer already keeps Devanagari matras intact)
  minilm          all-MiniLM-L6-v2 dense — what production uses today (English-only model)
  e5              intfloat/multilingual-e5-small dense — same 384 dims, multilingual
  hybrid_minilm   BM25 + MiniLM via RRF — the production retriever
  hybrid_e5       BM25 + e5 via RRF
  tr_hybrid       translate query → English with the LLM, then production hybrid

    uv run python -m rag.eval.multilingual_run --distractors 1000

Writes reports/multilingual_benchmark.md and reports/multilingual_benchmark.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))
load_dotenv(ROOT / ".env")

from rag.chunking.strategies import RecursiveCharacterChunker  # noqa: E402
from rag.eval.metrics import hit_at_k, mrr, recall_at_k  # noqa: E402
from rag.eval.multilingual_qa import call_with_retry  # noqa: E402
from rag.search.hybrid import rrf_combine  # noqa: E402
from rag.search.lexical import BM25Retriever  # noqa: E402
from rag.types import Chunk, Document  # noqa: E402

QA = ROOT / "data/golden/multilingual_qa.jsonl"
EN_PARQUET = ROOT / "data/subset/english.parquet"
INDIC_PARQUET = ROOT / "data/subset/indic.parquet"
CACHE = ROOT / ".cache/multilingual"
TRANSLATIONS = CACHE / "query_translations_en.json"
REPORT_MD = ROOT / "reports/multilingual_benchmark.md"
REPORT_JSON = ROOT / "reports/multilingual_benchmark.json"

SETS = ["en", "hi", "hinglish", "hi_native"]
RETRIEVERS = ["bm25", "minilm", "e5", "hybrid_minilm", "hybrid_e5", "tr_hybrid"]
DEPTH = 40  # chunks retrieved per query before doc-level dedupe

# e5 models are trained with these prefixes; leaving them out costs several recall points.
MODELS = {
    "minilm": ("all-MiniLM-L6-v2", "", ""),
    "e5": ("intfloat/multilingual-e5-small", "query: ", "passage: "),
}


# ── corpus ────────────────────────────────────────────────────────────────
def build_corpus(qa: list[dict], n_distractors: int, seed: int) -> tuple[list[Chunk], set[str]]:
    needed = {d for r in qa if r["set"] != "hi_native" for d in r["relevant_doc_ids"]}
    cols = ["doc_id", "title", "text", "url"]
    en = pq.read_table(EN_PARQUET, columns=cols).to_pandas()
    gold = en[en.doc_id.isin(needed)]
    rest = en[~en.doc_id.isin(needed)].sample(n=n_distractors, random_state=seed)
    indic = pq.read_table(INDIC_PARQUET, columns=cols).to_pandas()
    missing = needed - set(gold.doc_id)
    if missing:
        print(f"WARNING: {len(missing)} golden docs not found in english.parquet")

    chunker = RecursiveCharacterChunker(1000, 100)
    chunks: list[Chunk] = []
    for df in (gold, rest, indic):
        for row in df.itertuples(index=False):
            doc = Document(doc_id=row.doc_id, title=row.title or "", text=row.text or "", url=row.url or "")
            chunks.extend(chunker.chunk(doc))
    print(f"Corpus: {len(gold)} gold + {len(rest)} distractor EN docs, {len(indic)} Indic docs "
          f"→ {len(chunks)} chunks")
    return chunks, set(indic.doc_id)


def embed_corpus(key: str, chunks: list[Chunk]) -> np.ndarray:
    """Embed every chunk once and cache to disk (keyed by model + chunk ids)."""
    from sentence_transformers import SentenceTransformer

    model_name, _, doc_prefix = MODELS[key]
    sig = hashlib.sha1("|".join(c.chunk_id for c in chunks).encode()).hexdigest()[:12]
    path = CACHE / f"{key}_{sig}.npy"
    if path.exists():
        return np.load(path)
    CACHE.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name)
    t0 = time.time()
    vecs = model.encode([doc_prefix + c.text for c in chunks], batch_size=64,
                        normalize_embeddings=True, show_progress_bar=True)
    print(f"  embedded {len(chunks)} chunks with {model_name} in {time.time() - t0:.0f}s")
    np.save(path, vecs.astype(np.float32))
    return vecs


# ── query translation (for tr_hybrid) ─────────────────────────────────────
def translate_queries(questions: list[str]) -> dict[str, str]:
    cache: dict[str, str] = json.loads(TRANSLATIONS.read_text()) if TRANSLATIONS.exists() else {}
    todo = [q for q in dict.fromkeys(questions) if q not in cache]
    if todo:
        from rag.llm.client import create_llm_client

        llm = create_llm_client()
        print(f"Translating {len(todo)} queries to English with {llm.model_name}")
        for i in range(0, len(todo), 10):
            batch = todo[i : i + 10]
            prompt = ("Translate each search query into natural English. Keep proper nouns, acronyms "
                      "and numbers. Return ONLY a JSON array of strings of the same length "
                      f"({len(batch)}).\n\nINPUT:\n" + json.dumps(batch, ensure_ascii=False, indent=1))
            for q, t in zip(batch, call_with_retry(llm, prompt, len(batch))):
                cache[q] = str(t).strip()
            CACHE.mkdir(parents=True, exist_ok=True)
            TRANSLATIONS.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
    return cache


# ── evaluation ────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distractors", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    qa = [json.loads(line) for line in QA.read_text().splitlines() if line.strip()]
    qa = [r for r in qa if r["set"] in SETS]
    chunks, indic_ids = build_corpus(qa, args.distractors, args.seed)

    bm25 = BM25Retriever(chunks)
    vecs = {k: embed_corpus(k, chunks) for k in MODELS}

    from sentence_transformers import SentenceTransformer

    encoders = {k: SentenceTransformer(MODELS[k][0]) for k in MODELS}

    def dense(key: str, query: str) -> list[tuple[Chunk, float]]:
        qv = encoders[key].encode(MODELS[key][1] + query, normalize_embeddings=True)
        scores = vecs[key] @ qv
        top = np.argpartition(-scores, DEPTH)[:DEPTH]
        top = top[np.argsort(-scores[top])]
        return [(chunks[i], float(scores[i])) for i in top]

    translations = translate_queries([r["question"] for r in qa if r["set"] != "en"])

    def run(name: str, q: str, set_name: str) -> list[tuple[Chunk, float]]:
        if name == "bm25":
            return bm25.search(q, k=DEPTH)
        if name in MODELS:
            return dense(name, q)
        if name == "hybrid_minilm":
            return rrf_combine(bm25.search(q, k=DEPTH), dense("minilm", q), k=DEPTH)
        if name == "hybrid_e5":
            return rrf_combine(bm25.search(q, k=DEPTH), dense("e5", q), k=DEPTH)
        if name == "tr_hybrid":
            q_en = q if set_name == "en" else translations.get(q, q)
            return rrf_combine(bm25.search(q_en, k=DEPTH), dense("minilm", q_en), k=DEPTH)
        raise ValueError(name)

    agg: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    examples: list[dict] = []
    t0 = time.time()
    for r in qa:
        rel = set(r["relevant_doc_ids"])
        for name in RETRIEVERS:
            docs = list(dict.fromkeys(c.doc_id for c, _ in run(name, r["question"], r["set"])))
            m = agg[(r["set"], name)]
            m["hit@1"].append(hit_at_k(docs, rel, 1))
            m["recall@10"].append(recall_at_k(docs, rel, 10))
            m["mrr"].append(mrr(docs, rel))
            top10 = docs[:10]
            m["indic_share@10"].append(sum(d in indic_ids for d in top10) / max(len(top10), 1))
            if name == "hybrid_minilm" and r["set"] == "hi" and len(examples) < 5:
                examples.append({"q": r["question"], "hit@10": recall_at_k(docs, rel, 10),
                                 "indic_in_top10": sum(d in indic_ids for d in top10)})
    print(f"Evaluated {len(qa)} queries × {len(RETRIEVERS)} retrievers in {time.time() - t0:.0f}s")

    results = {f"{s}|{n}": {k: round(float(np.mean(v)), 4) for k, v in agg[(s, n)].items()}
               for s in SETS for n in RETRIEVERS if (s, n) in agg}
    counts = {s: sum(1 for r in qa if r["set"] == s) for s in SETS}
    REPORT_JSON.parent.mkdir(exist_ok=True)
    REPORT_JSON.write_text(json.dumps({"counts": counts, "n_chunks": len(chunks),
                                       "results": results, "examples": examples}, ensure_ascii=False, indent=1))
    write_markdown(results, counts, len(chunks), args.distractors)
    print(f"Wrote {REPORT_MD.relative_to(ROOT)}")


def write_markdown(results: dict, counts: dict, n_chunks: int, n_distractors: int) -> None:
    lines = [
        "# Multilingual Retrieval Benchmark",
        "",
        f"Corpus: golden EN docs + {n_distractors} EN distractors + 1,000 Indic docs → **{n_chunks:,} chunks** "
        "(recursive 1000/100). Labels are doc-level.",
        "",
        "Query sets: " + ", ".join(f"`{s}` ({counts[s]})" for s in SETS),
        "",
    ]
    for metric in ["recall@10", "mrr", "hit@1", "indic_share@10"]:
        lines += [f"## {metric}", "", "| retriever | " + " | ".join(SETS) + " |",
                  "|---|" + "---:|" * len(SETS)]
        for n in RETRIEVERS:
            cells = [f"{results.get(f'{s}|{n}', {}).get(metric, float('nan')):.3f}" for s in SETS]
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["`indic_share@10` = fraction of the top-10 docs that come from the Indic holdout. "
              "For `en`/`hi`/`hinglish` the right answer is always English, so a high share means the "
              "retriever is matching on *script*, not *meaning*.", ""]
    REPORT_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
