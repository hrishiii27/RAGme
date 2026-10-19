"""Phase 5 — Synthetic golden set generator.

For each sampled document, ask Gemini to write one question that ONLY that document can
answer. The label is doc-level (`doc_id`), so it works for every chunking strategy.

    uv run python -m rag.eval.synth --n 150

Output: data/golden/synthetic_qa.jsonl  (review ~30 by hand and set "reviewed": true)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import pyarrow.parquet as pq

SUBSET = Path("data/subset/english.parquet")
OUT = Path("data/golden/synthetic_qa.jsonl")
MODEL = "gemini-2.5-flash"

PROMPT = """You are building a retrieval benchmark over Indian government documents.
Read the document and write ONE realistic question a citizen might ask that this document
(and ideally only this one) answers. Rules:
- The question must be self-contained: do NOT say "the document", "this page" or "the text".
- Include a distinguishing detail (scheme name, department, place, year) when available.
- Prefer a concrete fact over a vague topic.
Return JSON: {{"question": "...", "answer": "short answer from the document"}}

TITLE: {title}
DOCUMENT:
{text}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-chars", type=int, default=800)
    args = ap.parse_args()

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    rows = pq.read_table(SUBSET, columns=["doc_id", "title", "text", "url", "kind"]).to_pylist()
    rows = [r for r in rows if len(r["text"]) >= args.min_chars]
    random.Random(args.seed).shuffle(rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if OUT.exists():  # resumable
        done = {d for line in OUT.read_text().splitlines() if line
                for d in json.loads(line)["relevant_doc_ids"]}

    written = len(done)
    with OUT.open("a") as f:
        for r in rows:
            if written >= args.n:
                break
            if r["doc_id"] in done:
                continue
            try:
                resp = client.models.generate_content(
                    model=MODEL,
                    contents=PROMPT.format(title=r["title"], text=r["text"][:6000]),
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json", temperature=0.4
                    ),
                )
                qa = json.loads(resp.text)
            except Exception as e:  # skip bad generations, keep going
                print(f"skip {r['doc_id']}: {e}")
                time.sleep(2)
                continue
            rec = {
                "qid": f"syn-{written:04d}",
                "question": qa["question"],
                "answer": qa.get("answer", ""),
                "relevant_doc_ids": [r["doc_id"]],
                "source_url": r["url"],
                "kind": r["kind"],
                "reviewed": False,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            if written % 10 == 0:
                print(f"{written}/{args.n}")
            time.sleep(0.5)
    print(f"Done: {written} questions -> {OUT}")


if __name__ == "__main__":
    main()
