"""Judge-vs-human agreement on the seed grading sheet.

    uv run python -m rag.eval.agreement

Reads data/golden/seed_grading.csv, fills the `Alignment` and `Agreement` columns in place
(your Human Critique / Human Outcome are never modified) and prints summary statistics.

  Alignment : Aligned | Misaligned   (exact match of Model Outcome vs Human Outcome)
  Agreement : Agree | Judge too lenient | Judge too strict   (direction of the disagreement)
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

CSV_PATH = Path("data/golden/seed_grading.csv")
ORDER = {"FAIL": 0, "PARTIAL": 1, "PASS": 2}


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(a) | set(b))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def main() -> None:
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames)
        rows = list(reader)

    graded = [r for r in rows if r["Human Outcome"].strip()]
    if not graded:
        raise SystemExit("No Human Outcome values filled in yet.")

    model, human = [], []
    for r in rows:
        h = r["Human Outcome"].strip().upper()
        m = r["Model Outcome"].strip().upper()
        if not h:
            continue
        if h not in ORDER:
            raise SystemExit(f"S.No {r['S.No']}: Human Outcome must be PASS/PARTIAL/FAIL, got {h!r}")
        r["Human Outcome"] = h
        model.append(m)
        human.append(h)
        if m == h:
            r["Alignment"], r["Agreement"] = "Aligned", "Agree"
        else:
            r["Alignment"] = "Misaligned"
            r["Agreement"] = "Judge too lenient" if ORDER[m] > ORDER[h] else "Judge too strict"

    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    n = len(model)
    exact = sum(m == h for m, h in zip(model, human))
    bin_ok = sum((m == "PASS") == (h == "PASS") for m, h in zip(model, human))
    print(f"graded rows        : {n}")
    print(f"exact agreement    : {exact}/{n} = {exact / n:.0%}")
    print(f"pass/not-pass agree: {bin_ok}/{n} = {bin_ok / n:.0%}")
    print(f"Cohen's kappa      : {cohen_kappa(model, human):.2f}")
    print(f"judge outcomes     : {dict(Counter(model))}")
    print(f"human outcomes     : {dict(Counter(human))}")
    print("\nDisagreements:")
    for r in rows:
        if r["Alignment"] == "Misaligned":
            print(f"  #{r['S.No']:>2} judge={r['Model Outcome']:<7} human={r['Human Outcome']:<7} "
                  f"({r['Agreement']}) {r['Sample Question'][:70]}")


if __name__ == "__main__":
    main()
