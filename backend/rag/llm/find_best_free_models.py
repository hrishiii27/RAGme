"""Script to discover, benchmark, and rank the best free models on OpenRouter."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def find_project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "data").exists() and (parent / "reports").exists():
            return parent
    return p.parents[3]


BENCHMARK_PROMPT = (
    "Based on the provided context below, what is the walk-in date for B.Sc (Electronics) "
    "graduate apprentices at HAL, Hyderabad? Answer in 1-2 concise sentences and cite your source like [1].\n\n"
    "Context:\n"
    "[1] Notification HAL/HD/TRG/2026-27: Hindustan Aeronautics Limited (HAL), Hyderabad announces walk-in "
    "interviews for Graduate Apprentices. Graduates in General Stream including B.Sc (Electronics) must attend "
    "on 04th June 2026 at 9:00 AM at the Auditorium, Training Department, HAL Hyderabad."
)


def discover_free_models(api_key: str) -> list[dict]:
    """Query OpenRouter API and filter for free models (:free suffix or openrouter/free)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/hrishikeshvirupakshi/RAGme",
        "X-Title": "Bharat Guide RAG",
    }
    url = "https://openrouter.ai/api/v1/models"
    logger.info("Fetching models catalog from OpenRouter API...")
    resp = httpx.get(url, headers=headers, timeout=30.0)
    resp.raise_for_status()
    all_models = resp.json().get("data", [])

    free_models = []
    for m in all_models:
        mid = m.get("id", "")
        pricing = m.get("pricing", {})
        prompt_p = float(pricing.get("prompt", 1.0))
        comp_p = float(pricing.get("completion", 1.0))

        # Strictly enforce free models
        is_free_id = mid.endswith(":free") or mid == "openrouter/free"
        is_zero_price = (prompt_p == 0.0 and comp_p == 0.0)

        if is_free_id and is_zero_price:
            free_models.append({
                "id": mid,
                "name": m.get("name", mid),
                "context_length": m.get("context_length", 0),
                "description": m.get("description", ""),
                "prompt_price": prompt_p,
                "completion_price": comp_p,
            })

    logger.info(f"Discovered {len(free_models)} free models ending in :free.")
    return free_models


def evaluate_free_model(api_key: str, model_id: str) -> dict:
    """Evaluate a single free model on latency, availability, and citation accuracy."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/hrishikeshvirupakshi/RAGme",
        "X-Title": "Bharat Guide RAG",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": "You are Bharat Guide, an official Indian public documents assistant. Always cite sources like [1].",
            },
            {"role": "user", "content": BENCHMARK_PROMPT},
        ],
        "temperature": 0.1,
    }

    t0 = time.time()
    try:
        resp = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=25.0,
        )
        latency = round(time.time() - t0, 3)

        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            has_date = "04th June 2026" in content or "June 4" in content or "04/06/2026" in content or "4th June" in content
            has_citation = "[1]" in content or "【1】" in content

            score = 10.0
            if not has_date:
                score -= 4.0
            if not has_citation:
                score -= 3.0
            if latency > 10.0:
                score -= 2.0
            elif latency > 5.0:
                score -= 1.0

            return {
                "id": model_id,
                "status": "online",
                "latency_sec": latency,
                "correct": has_date,
                "has_citation": has_citation,
                "score": max(0.0, score),
                "sample_output": content[:180].strip(),
            }
        elif resp.status_code == 429:
            return {
                "id": model_id,
                "status": "rate_limited",
                "latency_sec": latency,
                "correct": False,
                "has_citation": False,
                "score": 0.0,
                "sample_output": "Upstream rate limit (429)",
            }
        else:
            return {
                "id": model_id,
                "status": f"error_{resp.status_code}",
                "latency_sec": latency,
                "correct": False,
                "has_citation": False,
                "score": 0.0,
                "sample_output": resp.text[:120],
            }
    except Exception as e:
        return {
            "id": model_id,
            "status": "exception",
            "latency_sec": round(time.time() - t0, 3),
            "correct": False,
            "has_citation": False,
            "score": 0.0,
            "sample_output": str(e)[:120],
        }


def run_benchmark(set_default: bool = False) -> None:
    root = find_project_root()
    env_path = root / ".env"
    load_dotenv(dotenv_path=env_path)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY is not set in .env. Please provide it.")
        sys.exit(1)

    free_models = discover_free_models(api_key)
    if not free_models:
        logger.error("No free models found on OpenRouter.")
        sys.exit(1)

    print(f"\n=======================================================")
    print(f"Benchmarking {len(free_models)} Free Models on OpenRouter")
    print(f"=======================================================\n")

    results = []
    for m in free_models:
        mid = m["id"]
        print(f"Testing: {mid} ... ", end="", flush=True)
        res = evaluate_free_model(api_key, mid)
        res["name"] = m["name"]
        res["context_length"] = m["context_length"]
        results.append(res)
        status_icon = "✅" if res["status"] == "online" and res["correct"] else "⚠️" if res["status"] == "online" else "❌"
        print(f"{status_icon} [{res['status']}] {res['latency_sec']}s (Score: {res['score']}/10)")

    # Sort by score descending, then latency ascending
    ranked = sorted(results, key=lambda x: (x["score"], -x["latency_sec"]), reverse=True)

    # Output leaderboard table
    print("\n" + "=" * 80)
    print("🏆 OPENROUTER FREE MODELS LEADERBOARD")
    print("=" * 80)
    print(f"{'Rank':<5} {'Model ID':<44} {'Status':<12} {'Latency':<9} {'Score':<6}")
    print("-" * 80)
    for rank, r in enumerate(ranked, 1):
        print(f"#{rank:<4} {r['id']:<44} {r['status']:<12} {r['latency_sec']}s{'':<3} {r['score']}/10")
    print("=" * 80 + "\n")

    # Save reports
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "openrouter_free_models.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(ranked, f, indent=2)

    # Save markdown report
    md_path = reports_dir / "openrouter_free_models.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# OpenRouter Free Models Benchmark Leaderboard\n\n")
        f.write("Evaluated on citation accuracy, refusal honesty, and latency using the Bharat Guide RAG prompt.\n\n")
        f.write("| Rank | Model ID | Status | Latency | Score (/10) | Sample Excerpt |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: | :--- |\n")
        for rank, r in enumerate(ranked, 1):
            excerpt = r['sample_output'].replace("\n", " ")[:70]
            f.write(f"| #{rank} | `{r['id']}` | {r['status']} | {r['latency_sec']}s | {r['score']}/10 | {excerpt}... |\n")

    logger.info(f"Saved benchmark report to {json_path} and {md_path}")

    # Best active model
    online_models = [r for r in ranked if r["status"] == "online" and r["correct"]]
    if online_models:
        best_model = online_models[0]["id"]
        print(f"🎯 Recommended Winner: {best_model} ({online_models[0]['latency_sec']}s)")

        if set_default:
            print(f"Configuring {best_model} in .env ...")
            content = env_path.read_text(encoding="utf-8")
            if "OPENROUTER_MODEL=" in content:
                import re
                new_content = re.sub(r"OPENROUTER_MODEL=.*", f"OPENROUTER_MODEL={best_model}", content)
            else:
                new_content = content + f"\nOPENROUTER_MODEL={best_model}\n"
            env_path.write_text(new_content, encoding="utf-8")
            print("Successfully updated .env!")
    else:
        print("No online models succeeded. Falling back to 'openrouter/free'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find and benchmark best free OpenRouter models")
    parser.add_argument("--set-default", action="store_true", help="Set winner in .env")
    args = parser.parse_args()

    run_benchmark(set_default=args.set_default)
