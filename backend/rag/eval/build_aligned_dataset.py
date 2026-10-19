"""Generate full 100-row aligned dataset CSV embodying human-graded feedback from seed_grading.csv.

Columns:
- S.No
- Sample Question
- Reference Answer
- Model Answer (Aligned with exact citations, bonus specs, actionable redirection, follow-ups)
- Model Critique (Calibrated rubric evaluating grounding, precision, and redirection)
- Model Outcome (PASS / PARTIAL / FAIL)
- Follow-up Factual
- Follow-up Action
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from pathlib import Path

import pyarrow.parquet as pq

from rag.chunking.strategies import RecursiveCharacterChunker
from rag.embed.base import CachedEmbedder
from rag.embed.local import LocalEmbedder
from rag.generate.prompt import format_context
from rag.search.hybrid import rrf_combine
from rag.search.lexical import BM25Retriever
from rag.store.chroma import ChromaStore
from rag.types import Chunk, Document

EXPANDED_JSONL = Path("data/golden/expanded_100_qa.jsonl")
OUT_CSV = Path("data/golden/aligned_100_dataset.csv")
SUBSET = Path("data/subset/english.parquet")
COLLECTION = "A6000-recursive-1000-100"
GEN_MODEL = "gemini-2.5-flash"

SYSTEM_INSTRUCTION_GENERATOR = """You are an official Indian Government Information Assistant producing high-standard RAG answers aligned with human reviewer guidelines.

Human Reviewer Standards:
1. Exact Precision & Citations: Base answers strictly on context snippets. Cite with [1], [2].
2. Comprehensive "Cherries on Top": Include specific standard codes (e.g. IS standards), gazette numbers, or authority designations if present in snippets.
3. No Irrelevant Padding: If an asked entity or year is missing, NEVER pad with facts about different years or different ministers.
4. Actionable Redirection: If an answer is missing or partial, do NOT give generic boilerplate ("check online"). Name the specific department, state portal, or gazette indicated by the context/entity.
5. Hybrid Follow-up Questions: Conclude with:
### Suggested Follow-up Questions
- [Factual]: 1 specific factual clarification question.
- [Actionable]: 1 practical next-step guidance question for a citizen.
"""

CALIBRATED_JUDGE_PROMPT = """You are a rigorous judge evaluating a RAG answer against human-aligned standards.

QUESTION: {question}
REFERENCE FACT: {reference}
RETRIEVED CONTEXT:
{context}
GENERATED ANSWER:
{answer}

Human-Aligned Rubric:
- PASS: Accurately delivers the factual answer with citations [1], OR performs an honest refusal that explicitly directs to the specific official portal/department. Includes 2 valid follow-up questions.
- PARTIAL: Mostly correct but missing key conditions, or refuses without giving actionable specific redirection.
- FAIL: Factually wrong, hallucinates unsupported facts/ministers, or contradicts context.

Return JSON only:
{{
  "critique": "<2-3 sentences explaining evaluation according to the human-aligned rubric>",
  "outcome": "PASS" | "PARTIAL" | "FAIL"
}}
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
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt * 1.5)
    raise RuntimeError("unreachable")


def parse_followups(text: str) -> tuple[str, str]:
    factual, action = "", ""
    if "### Suggested Follow-up Questions" in text:
        parts = text.split("### Suggested Follow-up Questions")[-1].strip().split("\n")
        lines = [re.sub(r"^[-*0-9.]+\s*", "", line).strip() for line in parts if line.strip()]
        if len(lines) >= 1:
            factual = lines[0]
        if len(lines) >= 2:
            action = lines[1]
    return factual, action


def main() -> None:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # Load questions
    with EXPANDED_JSONL.open("r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(items)} items from {EXPANDED_JSONL}")

    # Load retriever
    print("Loading retriever index...")
    rows = pq.read_table(SUBSET).to_pylist()
    docs = [Document(doc_id=r["doc_id"], title=r["title"], text=r["text"], url=r["url"]) for r in rows]
    chunker = RecursiveCharacterChunker(1000, 100)
    embedder = CachedEmbedder(LocalEmbedder("all-MiniLM-L6-v2"), cache_dir=".cache/embeddings")
    store = ChromaStore(persist_dir=".chroma", collection_name=COLLECTION)

    all_chunks = []
    for d in docs:
        all_chunks.extend(chunker.chunk(d))
    chunk_by_id = {c.chunk_id: c for c in all_chunks}
    bm25 = BM25Retriever(all_chunks)

    def retrieve(query: str, top_k: int = 4) -> list[Chunk]:
        q_vec = embedder.embed_query(query)
        d_res = store.search(q_vec, n_results=16)
        dense_hits = []
        for r in d_res:
            c = chunk_by_id.get(r["id"])
            if c:
                dense_hits.append((c, 1.0 - (r["distance"] or 0.0)))
        bm25_hits = bm25.search(query, k=16)
        fused = rrf_combine(bm25_hits, dense_hits, k=top_k, rank_constant=60)
        return [c for c, _ in fused]

    # Resume capability if CSV already has rows
    existing_s_nos = set()
    csv_rows = []
    if OUT_CSV.exists():
        with OUT_CSV.open("r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_s_nos.add(int(row["S.No"]))
                csv_rows.append(row)
        print(f"Resuming: found {len(existing_s_nos)} already generated entries.")

    fieldnames = [
        "S.No",
        "Sample Question",
        "Reference Answer",
        "Model Answer",
        "Model Critique",
        "Model Outcome",
        "Follow-up Factual",
        "Follow-up Action",
    ]

    for idx, item in enumerate(items, 1):
        if idx in existing_s_nos:
            continue

        q = item["question"]
        ref = item["reference_answer"]
        chunks = retrieve(q, top_k=4)
        ctx_str = format_context([(c, 1.0) for c in chunks])

        # 1. Generate Aligned Model Answer
        prompt_content = f"Context:\n{ctx_str}\n\nUser Question: {q}\n\nAnswer:"
        ans_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION_GENERATOR,
            temperature=0.2,
        )
        answer = _generate(client, prompt_content, ans_config).strip()

        # 2. Evaluate with Calibrated Judge
        judge_content = CALIBRATED_JUDGE_PROMPT.format(
            question=q, reference=ref, context=ctx_str, answer=answer
        )
        judge_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
        )
        verdict_raw = _generate(client, judge_content, judge_config)
        verdict = json.loads(verdict_raw)

        factual_fu, action_fu = parse_followups(answer)

        entry = {
            "S.No": idx,
            "Sample Question": q,
            "Reference Answer": ref,
            "Model Answer": answer,
            "Model Critique": verdict.get("critique", ""),
            "Model Outcome": verdict.get("outcome", "PASS"),
            "Follow-up Factual": factual_fu,
            "Follow-up Action": action_fu,
        }
        csv_rows.append(entry)

        # Write incrementally so progress is preserved
        with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        if idx % 10 == 0 or idx == len(items):
            print(f"[{idx}/{len(items)}] Outcome: {entry['Model Outcome']} | Q: {q[:65]}...")

        time.sleep(0.3)

    print(f"Successfully generated all {len(csv_rows)} rows in {OUT_CSV}")


if __name__ == "__main__":
    main()
