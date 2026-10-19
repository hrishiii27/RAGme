"""Phase 6c — Prompt Tournament Engine.

Evaluates 4 prompting techniques over the golden/expanded benchmark:
1. Zero-Shot Direct
2. Few-Shot In-Context (Learned from human seed grading)
3. Hidden Chain-of-Thought (CoT with <reasoning> stripped)
4. Few-Shot + Hidden CoT (Hybrid Champion)

All techniques include the hybrid follow-up questions requirement (1 factual + 1 actionable next-step guidance).
Scoring is performed by a Calibrated LLM Judge measuring:
- correctness
- faithfulness
- citation_quality
- follow_up_quality
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from rag.chunking.strategies import RecursiveCharacterChunker
from rag.embed.base import CachedEmbedder
from rag.embed.local import LocalEmbedder
from rag.generate.prompt import format_context
from rag.llm.client import GeminiClient
from rag.search.hybrid import rrf_combine
from rag.search.lexical import BM25Retriever
from rag.store.chroma import ChromaStore
from rag.types import Chunk, Document

def find_project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "data").exists() and (parent / "reports").exists():
            return parent
    return p.parents[3]


from dotenv import load_dotenv

PROJECT_ROOT = find_project_root()
load_dotenv(PROJECT_ROOT / ".env")

EXPANDED_QA = PROJECT_ROOT / "data" / "golden" / "expanded_100_qa.jsonl"
SUBSET = PROJECT_ROOT / "data" / "subset" / "english.parquet"
REPORTS = PROJECT_ROOT / "reports"
COLLECTION = "A6000-recursive-1000-100"

# --- Prompt Templates ---

# 1. Zero-Shot Baseline
PROMPT_ZERO_SHOT = """You are an official Indian Government Information Assistant.
Answer the user's question faithfully using ONLY the provided context snippets.

Rules:
- Cite sources using [1], [2].
- If the answer is not present, state that clearly and suggest which official portal or ministry to check.
- Include a section '### Suggested Follow-up Questions' with 1 factual follow-up and 1 next-step practical action.

Context:
{context}

User Question: {question}

Answer:"""

# 2. Few-Shot In-Context (Built from your seed grading feedback!)
PROMPT_FEW_SHOT = """You are an official Indian Government Information Assistant.
Answer the user's question using ONLY the provided context snippets.

Rules:
1. Cite facts using [1], [2].
2. If exact details/specs are present (e.g. standards, names, amounts), include them clearly.
3. If the answer is not present or partial, do NOT guess or substitute facts from other years/ministers. State the refusal clearly and guide the citizen to the exact relevant department/portal.
4. Conclude with '### Suggested Follow-up Questions' (1 factual clarification + 1 actionable next-step guidance).

--- EXAMPLE 1 (Factual & Comprehensive) ---
Context:
[1] Central Motor Vehicles Rules (transport.gov.in)
Rule 100 specifies that windscreen and rear window of motor vehicles must have at least 70% visibility, and side windows minimum 50% visibility, conforming to Indian Standard IS: 2553-Part 2-1992.

User Question: What are the legal visibility requirements for vehicle windows?

Answer:
As per Rule 100 of the Central Motor Vehicles Rules 1989, the glass of the windscreen and rear window must maintain at least 70% visibility, while side windows require a minimum of 50% visibility [1]. The glass specifications must strictly conform to Indian Standards [IS: 2553 - Part 2 - 1992] [1].

### Suggested Follow-up Questions
- What are the penalties for non-compliance with Rule 100 visibility standards?
- How can a citizen obtain an official certification for approved automotive glass tinting?

--- EXAMPLE 2 (Actionable Refusal with Redirection) ---
Context:
[1] Tribal Welfare Scholarships Notice
Lists Pre-Matric and Post-Matric scholarships for ST students for academic year 2024.

User Question: Are registrations currently open for the ST Law Graduate scheme under the ST Advocates Stipend?

Answer:
I cannot find information regarding the ST Law Graduate scheme or the ST Advocates Stipend in the provided documents [1]. The available records only cover Pre-Matric and Post-Matric scholarships. 

To check the current registration status for this specific scheme, please consult the official Tribal Welfare Department portal (e.g. Telangana ePASS or your respective state tribal welfare website).

### Suggested Follow-up Questions
- What are the general eligibility requirements for the ST Post-Matric Scholarship scheme?
- Which state government department handles advocacy stipends for ST law graduates?
-------------------------------------------

Context:
{context}

User Question: {question}

Answer:"""

# 3. Hidden Chain-of-Thought
PROMPT_COT = """You are an official Indian Government Information Assistant.
Analyze the question step-by-step in an internal <reasoning> block before giving your answer.

Inside <reasoning>:
1. Target: Identify the exact entity, date, rule, or amount requested.
2. Context Evidence: What snippets mention this directly? Are there multi-step conditions or math additions?
3. Actionable Path: Is the answer fully present? If partial/absent, identify which exact department or portal should be named.
4. Follow-ups: Formulate 1 factual inquiry and 1 next-step citizen guidance question.

After </reasoning>, provide the polished, citizen-facing Answer with citations [1], [2], followed by:
### Suggested Follow-up Questions
- [Factual follow-up]
- [Actionable guidance follow-up]

Context:
{context}

User Question: {question}"""

# 4. Few-Shot + Hidden CoT (Hybrid)
PROMPT_HYBRID = PROMPT_FEW_SHOT.replace(
    "Answer the user's question using ONLY the provided context snippets.",
    "Think step-by-step inside a hidden <reasoning> ... </reasoning> block first to verify math, dates, and grounding, then output the final citizen answer with citations [1] and '### Suggested Follow-up Questions'."
)


# --- Calibrated LLM Judge Prompt ---
JUDGE_EVAL_PROMPT = """You are an expert evaluator grading a RAG answer against a ground-truth reference answer and human-calibrated standards.

QUESTION: {question}
REFERENCE ANSWER: {reference}
RETRIEVED CONTEXT: {context}
GENERATED ANSWER: {answer}

Scoring Rules:
1. correctness (1-5): If facts are in the context, does the answer accurately convey them? If the retrieved context genuinely lacks the answer, an honest refusal that does not hallucinate should receive 5/5.
2. faithfulness (1-5): Are all claims strictly supported by the context snippets? Zero hallucination, no unmentioned ministers/dates.
3. citation_quality (1-5): Are claims cited cleanly with [1], [2]?
4. redirection_quality (1-5): If information was absent/partial, did it guide the user to an exact specific department/portal/email instead of generic boilerplate? If the answer was fully found in context, rate 5/5.
5. follow_up_quality (1-5): Are there 2 relevant follow-up questions (hybrid: 1 factual clarification + 1 actionable next-step guidance)?

Verdict Criteria:
- PASS: Accurately delivers the factual answer from context with citations [1], OR performs an honest refusal with specific departmental redirection when the fact is absent from context. Includes 2 valid follow-up questions.
- PARTIAL: Mostly correct but missing key conditions, or refuses without giving actionable specific redirection.
- FAIL: Hallucinates unsupported facts/dates, makes math errors, or contradicts context.

Return JSON only:
{{
  "correctness": <int 1-5>,
  "faithfulness": <int 1-5>,
  "citation_quality": <int 1-5>,
  "redirection_quality": <int 1-5>,
  "follow_up_quality": <int 1-5>,
  "verdict": "PASS" | "PARTIAL" | "FAIL",
  "critique": "<2 sentences explaining the evaluation according to the rubric>"
}}"""


def generate_with_retry(client, model, contents, config, max_retries=5):
    """Helper with exponential backoff for reliable evaluation runs."""
    delay = 1.0
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(delay)
            delay *= 2.0


def clean_cot(text: str) -> str:
    """Strips the <reasoning> block for clean user presentation."""
    return re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL).strip()


def run_tournament(sample_size: int = 25) -> None:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # Load benchmark items
    with EXPANDED_QA.open("r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()][:sample_size]

    print(f"Loaded {len(items)} benchmark test questions from {EXPANDED_QA}.")

    # Initialize Retrieval
    print("Initializing RRF Hybrid Retriever...")
    rows = pq.read_table(SUBSET).to_pylist()
    docs = [Document(doc_id=r["doc_id"], title=r["title"], text=r["text"], url=r["url"]) for r in rows]
    chunker = RecursiveCharacterChunker(1000, 100)
    embedder = CachedEmbedder(LocalEmbedder("all-MiniLM-L6-v2"), cache_dir=str(PROJECT_ROOT / ".cache" / "embeddings"))
    store = ChromaStore(persist_dir=str(PROJECT_ROOT / ".chroma"), collection_name=COLLECTION)

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

    techniques = {
        "Zero-Shot": PROMPT_ZERO_SHOT,
        "Few-Shot (Learned)": PROMPT_FEW_SHOT,
        "Hidden CoT": PROMPT_COT,
        "Few-Shot + CoT": PROMPT_HYBRID,
    }

    results_by_technique = defaultdict(lambda: defaultdict(list))
    item_details = []

    print(f"\nStarting Prompt Tournament across 4 techniques on {len(items)} questions...")

    for idx, item in enumerate(items, 1):
        q = item["question"]
        ref = item["reference_answer"]
        chunks = retrieve(q, top_k=4)
        ctx_str = format_context([(c, 1.0) for c in chunks])

        print(f"\n[{idx}/{len(items)}] Q: {q[:75]}...")

        item_record = {
            "index": idx,
            "qid": item.get("qid", f"q-{idx:04d}"),
            "question": q,
            "reference_answer": ref,
            "techniques": {}
        }

        for name, tmpl in techniques.items():
            prompt = tmpl.format(context=ctx_str, question=q)
            resp = generate_with_retry(
                client=client,
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2),
            )
            raw_answer = resp.text or ""
            clean_answer = clean_cot(raw_answer)

            # Judge evaluation
            j_prompt = JUDGE_EVAL_PROMPT.format(
                question=q, reference=ref, context=ctx_str, answer=clean_answer
            )
            j_resp = generate_with_retry(
                client=client,
                model="gemini-2.5-flash",
                contents=j_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", temperature=0.0
                ),
            )
            try:
                scores = json.loads(j_resp.text)
            except Exception:
                scores = {
                    "correctness": 3,
                    "faithfulness": 3,
                    "citation_quality": 3,
                    "redirection_quality": 3,
                    "follow_up_quality": 3,
                    "verdict": "PARTIAL",
                    "critique": "Failed to parse JSON judge output."
                }

            for k in ["correctness", "faithfulness", "citation_quality", "redirection_quality", "follow_up_quality"]:
                results_by_technique[name][k].append(scores.get(k, 3))
            results_by_technique[name]["pass_count"].append(1 if scores.get("verdict") == "PASS" else 0)

            item_record["techniques"][name] = {
                "generated_answer": clean_answer,
                "scores": {k: scores.get(k, 3) for k in ["correctness", "faithfulness", "citation_quality", "redirection_quality", "follow_up_quality"]},
                "verdict": scores.get("verdict", "PARTIAL"),
                "critique": scores.get("critique", "")
            }

            time.sleep(0.3)

        item_details.append(item_record)

    # Compute Summary
    tournament_summary = []
    print("\n" + "=" * 60)
    print("PROMPT TOURNAMENT LEADERBOARD")
    print("=" * 60)

    for name, metrics in results_by_technique.items():
        pass_rate_val = (sum(metrics['pass_count']) / len(items)) * 100
        corr_val = sum(metrics['correctness']) / len(items)
        faith_val = sum(metrics['faithfulness']) / len(items)
        cite_val = sum(metrics['citation_quality']) / len(items)
        redir_val = sum(metrics['redirection_quality']) / len(items)
        fol_val = sum(metrics['follow_up_quality']) / len(items)
        composite_score = corr_val + faith_val + cite_val + redir_val + fol_val

        summary_row = {
            "Technique": name,
            "Pass Rate": f"{pass_rate_val:.1f}%",
            "Correctness (/5)": f"{corr_val:.2f}",
            "Faithfulness (/5)": f"{faith_val:.2f}",
            "Citations (/5)": f"{cite_val:.2f}",
            "Redirection (/5)": f"{redir_val:.2f}",
            "Follow-ups (/5)": f"{fol_val:.2f}",
            "Composite (/25)": f"{composite_score:.2f}",
        }
        tournament_summary.append(summary_row)

    REPORTS.mkdir(exist_ok=True)
    
    # Save exact aggregated summary
    (REPORTS / "prompt_tournament.json").write_text(json.dumps(tournament_summary, indent=2))
    
    # Save exact question-by-question granular comparison logs
    (REPORTS / "prompt_tournament_details.json").write_text(json.dumps(item_details, indent=2))

    cols = list(tournament_summary[0].keys())
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in tournament_summary]
    md_str = "\n".join(md)
    (REPORTS / "prompt_tournament.md").write_text(md_str)
    print(md_str)
    print(f"\nSaved summary to {REPORTS / 'prompt_tournament.json'}")
    print(f"Saved granular item comparisons to {REPORTS / 'prompt_tournament_details.json'}")

    # Compile unified master report
    try:
        from rag.eval.compile_benchmarks import compile_master_report
        compile_master_report()
    except Exception as e:
        print(f"Warning: Failed to compile master report: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=25)
    args = parser.parse_args()
    run_tournament(sample_size=args.samples)

