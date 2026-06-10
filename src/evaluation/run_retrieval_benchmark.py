"""
Retrieval benchmark for the Sustainable Travel RAG corpus.

Loads 30 queries from eval/retrieval_benchmark.json, runs each through
the persistent ChromaDB index, and computes precision-at-5 against the
expected_source_id labels. Saves per-query results plus aggregate stats.

Author: Bonna Bambilla, MSc Data Science, Arden University
Project: COM7014 Advanced Computing Project
Output:  eval/results/retrieval_benchmark_results.json
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import chromadb
from chromadb.utils import embedding_functions

BENCHMARK_PATH = Path("eval/retrieval_benchmark.json")
RESULTS_PATH = Path("eval/results/retrieval_benchmark_results.json")
DB_PATH = Path("data/chroma_db")
COLLECTION_NAME = "sustainable_travel_v1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
THRESHOLD = 0.75


def main():
    print(f"Loading benchmark from {BENCHMARK_PATH}")
    with BENCHMARK_PATH.open() as f:
        benchmark = json.load(f)
    queries = benchmark["queries"]
    print(f"Loaded {len(queries)} queries")
    print()

    print(f"Connecting to ChromaDB at {DB_PATH}")
    client = chromadb.PersistentClient(path=str(DB_PATH))
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
    )
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedder,
    )
    print(f"Collection: {COLLECTION_NAME} ({collection.count()} chunks)")
    print()

    per_query = []
    hits = 0
    by_tier = defaultdict(lambda: {"hits": 0, "total": 0})
    by_source = defaultdict(lambda: {"hits": 0, "total": 0})

    for q in queries:
        results = collection.query(
            query_texts=[q["query_text"]],
            n_results=TOP_K,
        )
        retrieved_source_ids = [m["source_id"] for m in results["metadatas"][0]]
        retrieved_chunk_ids = list(results["ids"][0])
        is_hit = q["expected_source_id"] in retrieved_source_ids

        per_query.append({
            "query_id": q["query_id"],
            "query_text": q["query_text"],
            "expected_source_id": q["expected_source_id"],
            "expected_source_name": q["expected_source_name"],
            "tier": q["tier"],
            "retrieved_source_ids": retrieved_source_ids,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "hit": is_hit,
        })

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
    print("PRECISION-AT-5 RESULTS")
    print("=" * 60)
    print(f"Overall:  {hits}/{len(queries)} = {overall:.3f}   [{verdict} - threshold {THRESHOLD}]")
    print()
    print("By tier:")
    for tier in sorted(by_tier):
        t = by_tier[tier]
        score = t["hits"] / t["total"] if t["total"] else 0
        print(f"  Tier {tier}: {t['hits']}/{t['total']} = {score:.3f}")
    print()
    print("By source:")
    for src in sorted(by_source):
        s = by_source[src]
        score = s["hits"] / s["total"] if s["total"] else 0
        print(f"  Source {src}: {s['hits']}/{s['total']} = {score:.3f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "summary": {
            "total_queries": len(queries),
            "hits": hits,
            "precision_at_5": round(overall, 4),
            "threshold": THRESHOLD,
            "verdict": verdict,
        },
        "by_tier": {str(k): {**v, "precision": round(v["hits"]/v["total"], 4) if v["total"] else 0} for k, v in by_tier.items()},
        "by_source": {str(k): {**v, "precision": round(v["hits"]/v["total"], 4) if v["total"] else 0} for k, v in by_source.items()},
        "per_query": per_query,
    }
    with RESULTS_PATH.open("w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Per-query results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
