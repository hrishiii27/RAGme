"""Seed grading sheet: model answer + AI-judge critique, ready for human grading.

    uv run python -m rag.eval.seed_csv

Pipeline per question: retrieve top-5 chunks (recursive-1000-100 index, dense) ->
Gemini answers from the context only -> a SEPARATE Gemini judge call critiques the answer
against the reference answer and returns PASS / PARTIAL / FAIL.

Writes:
  data/golden/seed_grading.csv   <- you edit Human Critique / Human Outcome
  data/golden/seed_grading_meta.json  <- S.No -> qid, doc ids, retrieval-hit flag (not for editing)
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import pyarrow.parquet as pq

from rag.embed.base import CachedEmbedder
from rag.embed.local import LocalEmbedder
from rag.store.chroma import ChromaStore

GOLDEN = Path("data/golden/synthetic_qa.jsonl")
SUBSET = Path("data/subset/english.parquet")
OUT_CSV = Path("data/golden/seed_grading.csv")
OUT_META = Path("data/golden/seed_grading_meta.json")
COLLECTION = "A6000-recursive-1000-100"
GEN_MODEL = "gemini-2.5-flash"
TOP_K = 5

# 16 clean, specific questions + 4 deliberate edge cases (marked below) so the judge
# is calibrated on hard / ambiguous / failing cases too, not just easy passes.
QIDS = [
    "syn-0004", "syn-0007", "syn-0020", "syn-0022", "syn-0024", "syn-0027",
    "syn-0040", "syn-0045", "syn-0048", "syn-0055", "syn-0060", "syn-0070",
    "syn-0086", "syn-0036", "syn-0065", "syn-0043",
    # edge cases:
    "syn-0000",  # generic website disclaimer: many sites say the same thing
    "syn-0012",  # same, different site
    "syn-0010",  # time-sensitive ("currently open")
    "syn-0107",  # Hindi question vs. English-only embedder
]

ANSWER_PROMPT = """You answer questions about Indian government information using ONLY the context below.
Rules:
- If the context does not contain the answer, say you could not find it in the provided
  documents and suggest what kind of official source to check. Do not guess.
- Be concise (1-3 sentences). Cite sources like [1], [2].

CONTEXT:
{context}

QUESTION: {question}
ANSWER:"""

JUDGE_PROMPT = """You are grading a RAG system's answer. Priorities, in order:
correctness > faithfulness (supported by the retrieved context) > citation quality > completeness.

QUESTION: {question}
REFERENCE ANSWER (from the source document): {reference}
RETRIEVED CONTEXT (what the system saw):
{context}
SYSTEM ANSWER: {answer}

Decide:
- PASS: correct and supported.
- PARTIAL: partly correct, incomplete, or right but poorly supported/cited.
- FAIL: wrong, unsupported, hallucinated, or an unnecessary refusal when the context had the answer.
Note: a refusal is acceptable (PASS) only if the context truly lacks the answer.

Return JSON: {{"critique": "2-3 sentences explaining the verdict", "outcome": "PASS|PARTIAL|FAIL"}}"""


def _generate(client, contents: str, config, tries: int = 6) -> str:
    """generate_content with exponential backoff on transient API errors (503/429/timeouts)."""
    for attempt in range(tries):
        try:
            return client.models.generate_content(
                model=GEN_MODEL, contents=contents, config=config
            ).text
        except Exception as e:  # noqa: BLE001 - google-genai raises several error types
            if attempt == tries - 1:
                raise
            wait = 2 ** attempt * 2
            print(f"   retry in {wait}s ({type(e).__name__})")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def main() -> None:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    golden = {
        r["qid"]: r
        for r in (json.loads(l) for l in GOLDEN.read_text().splitlines() if l)
    }
    urls = {
        r["doc_id"]: (r["title"], r["url"])
        for r in pq.read_table(SUBSET, columns=["doc_id", "title", "url"]).to_pylist()
    }

    embedder = CachedEmbedder(LocalEmbedder("all-MiniLM-L6-v2"), cache_dir=".cache/embeddings")
    store = ChromaStore(persist_dir=".chroma", collection_name=COLLECTION)

    rows, meta = [], []
    for n, qid in enumerate(QIDS, start=1):
        g = golden[qid]
        hits = store.search(embedder.embed_query(g["question"]), n_results=TOP_K)
        ctx_parts = []
        for i, h in enumerate(hits, start=1):
            title, url = urls.get(h["metadata"]["doc_id"], ("?", "?"))
            ctx_parts.append(f"[{i}] {title} ({url})\n{h['document']}")
        context = "\n\n".join(ctx_parts)

        answer = _generate(
            client,
            ANSWER_PROMPT.format(context=context, question=g["question"]),
            types.GenerateContentConfig(temperature=0.0),
        ).strip()

        verdict = json.loads(_generate(
            client,
            JUDGE_PROMPT.format(
                question=g["question"], reference=g["answer"], context=context, answer=answer
            ),
            types.GenerateContentConfig(
                response_mime_type="application/json", temperature=0.0
            ),
        ))

        rows.append({
            "S.No": n,
            "Sample Question": g["question"],
            "Model Answer": answer,
            "Model Critique": verdict["critique"],
            "Model Outcome": verdict["outcome"],
            "Human Critique": "",
            "Human Outcome": "",
            "Alignment": "",
            "Agreement": "",
        })
        gold = set(g["relevant_doc_ids"])
        meta.append({
            "s_no": n, "qid": qid, "reference_answer": g["answer"],
            "gold_doc_ids": list(gold),
            "gold_in_top5": any(h["metadata"]["doc_id"] in gold for h in hits),
            "edge_case": qid in QIDS[16:],
        })
        print(f"{n:2d}/{len(QIDS)} {qid} gold_in_top5={meta[-1]['gold_in_top5']} -> {verdict['outcome']}")
        time.sleep(1)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:  # BOM so Excel reads Hindi
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
