"""Multilingual sprint — build query sets for the cross-lingual retrieval benchmark.

Four query sets, all with DOC-level labels (same convention as synthetic_qa.jsonl):

  en         original English golden questions             → English docs
  hi         same questions translated to Hindi (Devanagari) → English docs
  hinglish   same questions in romanized Hindi (Hinglish)    → English docs
  hi_native  Hindi questions written FROM Indic holdout docs → Indic docs

    uv run python -m rag.eval.multilingual_qa --n-native 100

Output: data/golden/multilingual_qa.jsonl (one row per query; re-running skips work already done)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))
load_dotenv(ROOT / ".env")

from rag.llm.client import create_llm_client  # noqa: E402

GOLDEN = ROOT / "data/golden/synthetic_qa.jsonl"
INDIC = ROOT / "data/subset/indic.parquet"
OUT = ROOT / "data/golden/multilingual_qa.jsonl"

TRANSLATE_PROMPT = """Translate each English question into {target}.
{style}
Keep proper nouns (scheme names, departments, places, acronyms, numbers, years) recognisable.
Return ONLY a JSON array of strings, same order and same length ({n}) as the input.

INPUT:
{questions}
"""

STYLES = {
    "hi": ("Hindi", "Write in Devanagari script, the way a Hindi-speaking citizen would type it."),
    "hinglish": (
        "Hinglish (romanized Hindi)",
        "Write Hindi words in Latin script, mixing in English terms naturally, the way people type "
        "on WhatsApp (e.g. 'HAL mein apprentice ka walk-in date kya hai?').",
    ),
}

NATIVE_PROMPT = """You are building a retrieval benchmark over Indian government documents written in Hindi.
For EACH document below, write ONE realistic question in Hindi (Devanagari) that a citizen might ask
and that this document answers. Include a distinguishing detail (scheme, department, district, year).
Do NOT say "this document". If a document is only navigation menus / boilerplate with no answerable
fact, return null for it.
Return ONLY a JSON array of length {n}: each item is either null or {{"question": "...", "answer": "..."}}.

{docs}
"""


def extract_json_array(text: str) -> list:
    """LLMs wrap JSON in prose or ``` fences; grab the outermost [...] block."""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise ValueError(f"No JSON array in LLM output: {text[:200]!r}")
    return json.loads(m.group(0))


def call_with_retry(llm, prompt: str, expect_len: int, tries: int = 6) -> list:
    for attempt in range(tries):
        try:
            arr = extract_json_array(llm.generate(prompt))
            if len(arr) != expect_len:
                raise ValueError(f"expected {expect_len} items, got {len(arr)}")
            return arr
        except Exception as e:  # rate limits, malformed JSON, wrong length
            wait = min(10 * 2**attempt, 120)  # 10, 20, 40, 80, 120s
            print(f"  retry {attempt + 1}/{tries} in {wait}s: {e}")
            time.sleep(wait)
    raise RuntimeError("LLM call failed after retries")


def load_existing() -> list[dict]:
    if not OUT.exists():
        return []
    return [json.loads(line) for line in OUT.read_text().splitlines() if line.strip()]


def save(rows: list[dict]) -> None:
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-native", type=int, default=100)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--provider", default=None, help="openrouter | gemini | ollama (default: LLM_PROVIDER)")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    llm = create_llm_client(args.provider, args.model)
    print(f"LLM: {llm.model_name}")
    rows = load_existing()
    done = {(r["set"], r["source_qid"]) for r in rows}

    golden = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]

    # 1. English baseline — copied as-is
    for g in golden:
        if ("en", g["qid"]) not in done:
            rows.append({"set": "en", "source_qid": g["qid"], "question": g["question"],
                         "relevant_doc_ids": g["relevant_doc_ids"]})
    save(rows)

    # 2. Translated sets
    for set_name, (target, style) in STYLES.items():
        todo = [g for g in golden if (set_name, g["qid"]) not in done]
        print(f"[{set_name}] {len(todo)} questions to translate")
        for i in range(0, len(todo), args.batch):
            batch = todo[i : i + args.batch]
            prompt = TRANSLATE_PROMPT.format(
                target=target, style=style, n=len(batch),
                questions=json.dumps([g["question"] for g in batch], ensure_ascii=False, indent=1),
            )
            for g, q in zip(batch, call_with_retry(llm, prompt, len(batch))):
                rows.append({"set": set_name, "source_qid": g["qid"], "question": str(q).strip(),
                             "relevant_doc_ids": g["relevant_doc_ids"]})
            save(rows)
            print(f"  {min(i + args.batch, len(todo))}/{len(todo)}")

    # 3. Native Hindi questions from the Indic holdout
    have_native = sum(1 for r in rows if r["set"] == "hi_native")
    if have_native < args.n_native:
        used = {r["source_qid"] for r in rows if r["set"] == "hi_native"}
        tried = {r["source_qid"] for r in rows if r["set"] == "_skipped"}
        df = pq.read_table(INDIC, columns=["doc_id", "title", "text", "n_chars"]).to_pandas()
        df = df[(df.n_chars >= 1500) & ~df.doc_id.isin(used | tried)]
        pool = df.sample(frac=1, random_state=args.seed).to_dict("records")
        per_call = 5
        print(f"[hi_native] need {args.n_native - have_native} more")
        while have_native < args.n_native and pool:
            batch, pool = pool[:per_call], pool[per_call:]
            docs = "\n\n".join(
                f"### DOCUMENT {k + 1}\nTITLE: {d['title']}\n{d['text'][:1800]}" for k, d in enumerate(batch)
            )
            out = call_with_retry(llm, NATIVE_PROMPT.format(n=len(batch), docs=docs), len(batch))
            for d, item in zip(batch, out):
                if isinstance(item, dict) and item.get("question") and have_native < args.n_native:
                    rows.append({"set": "hi_native", "source_qid": d["doc_id"],
                                 "question": item["question"].strip(), "relevant_doc_ids": [d["doc_id"]]})
                    have_native += 1
                else:  # remember boilerplate docs so re-runs don't retry them
                    rows.append({"set": "_skipped", "source_qid": d["doc_id"], "question": "",
                                 "relevant_doc_ids": []})
            save(rows)
            print(f"  {have_native}/{args.n_native}")

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["set"]] = counts.get(r["set"], 0) + 1
    print("Done:", counts)


if __name__ == "__main__":
    main()
