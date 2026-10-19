"""Expand Golden QA dataset with 100 high-quality entries modelled after the human-graded seed CSV.

Criteria based on human feedback:
- Explicit details: numbers, dates, circular codes, person designations, conditions.
- Direct ground-truth answer.
- Grounded in documents from data/subset/english.parquet.
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
OUT_JSONL = Path("data/golden/expanded_100_qa.jsonl")
GEN_MODEL = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = """You are an expert AI creating a high-rigor benchmark for Indian Government Public Information RAG.
Read the provided document and produce ONE realistic, highly specific citizen query that this document answers.

Style Guidelines (based on human feedback):
1. Precision: Focus on concrete entities, amounts, timelines, percentages, designations, rules, or circular numbers.
2. Grounded Reference: Provide the exact, untampered factual answer from the text.
3. Realistic Tone: Phrase it naturally as an Indian citizen, RTI filer, or beneficiary asking about their eligibility, rules, or details.
4. Avoid generic questions (e.g. do not ask "What is this document about?" or "What are the disclaimers?").

Output JSON:
{
  "question": "<specific citizen question>",
  "reference_answer": "<exact factual answer strictly from the document>",
  "key_entity": "<the key scheme, rule, department, or individual mentioned>"
}
"""

PROMPT_TEMPLATE = """TITLE: {title}
URL: {url}
TEXT:
{text}
"""


def _generate(client, contents: str, config, tries: int = 5) -> str:
    for attempt in range(tries):
        try:
            resp = client.models.generate_content(
                model=GEN_MODEL,
                contents=contents,
                config=config,
            )
            return resp.text or ""
        except Exception as e:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt * 2)
    raise RuntimeError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # Load English subset documents
    print(f"Loading documents from {SUBSET}...")
    rows = pq.read_table(SUBSET, columns=["doc_id", "title", "text", "url", "kind"]).to_pylist()
    # Filter for substantive documents
    rows = [r for r in rows if 800 <= len(r["text"]) <= 35000]
    random.Random(args.seed).shuffle(rows)

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    existing_ids: set[str] = set()
    if OUT_JSONL.exists():
        for line in OUT_JSONL.read_text().splitlines():
            if line.strip():
                existing_ids.add(json.loads(line)["doc_id"])

    print(f"Found {len(existing_ids)} already generated entries.")
    written = len(existing_ids)

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        temperature=0.3,
    )

    with OUT_JSONL.open("a", encoding="utf-8") as f:
        for r in rows:
            if written >= args.count:
                break
            if r["doc_id"] in existing_ids:
                continue

            content = PROMPT_TEMPLATE.format(
                title=r["title"],
                url=r["url"],
                text=r["text"][:7000],
            )

            try:
                raw_json = _generate(client, content, config)
                data = json.loads(raw_json)
                q = data.get("question", "").strip()
                ref = data.get("reference_answer", "").strip()
                if not q or not ref:
                    continue

                record = {
                    "qid": f"exp-{written + 1:04d}",
                    "doc_id": r["doc_id"],
                    "question": q,
                    "reference_answer": ref,
                    "key_entity": data.get("key_entity", ""),
                    "source_url": r["url"],
                    "source_title": r["title"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                written += 1
                if written % 10 == 0:
                    print(f"Progress: {written}/{args.count} QA pairs created...")
                time.sleep(0.4)
            except Exception as ex:
                print(f"Skipping {r['doc_id']}: {ex}")
                time.sleep(2)

    print(f"Completed! {written} benchmark entries written to {OUT_JSONL}")


if __name__ == "__main__":
    main()
