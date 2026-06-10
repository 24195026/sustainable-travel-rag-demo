"""Regression benchmark — uses the retrieval module instead of direct ChromaDB.

If this produces the same 0.833 precision-at-5 as run_retrieval_benchmark.py,
the retrieval module is a faithful wrapper around ChromaDB.

Author: Bonna Bambilla, MSc Data Science, Arden University
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import defaultdict

# Add src/ to path so we can import the retrieval module
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval import retrieve

BENCHMARK_PATH = Path("eval/retrieval_benchmark.json")
TOP_K = 5
THRESHOLD = 0.75


def main():
    print(f"Regression benchmark using src/retrieval/retrieve.py")
    print(f"Loading benchmark from {BENCHMARK_PATH}")
    with BENCHMARK_PATH.open() as f:
        benchmark = json.load(f)
    queries = benchmark["queries"]
    print(f"Loaded {len(queries)} queries")
    print()

    hits = 0
    by_tier = defaultdict(lambda: {"hits": 0, "total": 0})
    by_source = defaultdict(lambda: {"hits": 0, "total": 0})

    for q in queries:
        chunks = retrieve(q["query_text"], k=TOP_K)
        retrieved_source_ids = [c.source_id for c in chunks]
        is_hit = q["expected_source_id"] in retrieved_source_ids

        if is_hit:
            hits += 1
            by_tier[q["tier"]]["hits"] += 1
            by_source[q["expected_source_id"]]["hits"] += 1
        by_tier[q["tier"]]["total"] += 1
        by_source[q["expected_source_id"]]["total"] += 1

        marker = "HIT " if is_hit else "MISS"
        print(f"{q['query_id']}  {marker}  (expected Src {q['expected_source_id']})  retrieved: {retrieved_source_ids}")

    overall = hits / len(queries)
    verdict = "PASS" if overall >= THRESHOLD else "FAIL"

    print()
    print("=" * 60)
    print("REGRESSION RESULT")
    print("=" * 60)
    print(f"Overall: {hits}/{len(queries)} = {overall:.3f}  [{verdict}]")
    print()
    print("Compare to the standalone benchmark in eval/results/retrieval_benchmark_results.json")
    print("If overall matches 0.833 exactly, the retrieval module is a faithful wrapper.")


if __name__ == "__main__":
    main()
