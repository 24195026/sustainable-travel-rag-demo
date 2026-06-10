"""Reranker orchestrator for the Sustainable Travel RAG chatbot.

Combines the four reranker components into a single rerank() call:
    geo.haversine_km        - great-circle distance
    geocode.geocode         - name -> coordinates
    normalise.*             - terms to a common [0,1] scale
    score.compute_score     - alpha*GSTC + beta*relevance - gamma*carbon

Design (decided Phase 4.4.5): the reranker reorders retrieved CHUNKS.
  - GSTC and relevance are natural chunk properties (always applied).
  - Carbon is an origin->destination property. It is computed ONCE per
    query (the same trip applies to every chunk) and is only active when
    both origin and destination are supplied and geocodable. When either
    is missing (e.g. verification queries with no destination), the
    carbon term is 0 and the reranker reduces to GSTC + relevance.

Documented limitation: because carbon is uniform across all chunks in a
single query, it shifts every chunk's score equally and does NOT reorder
chunks within that query. Its measurable effect is at the destination/
evaluation level (carbon savings, regional disparity - RQ2), not within-
query chunk ordering. This is by design for a chunk-level reranker.

Public API:
    rerank(chunks, origin=None, destination=None, weights=Weights(),
           return_scores=False) -> list[RetrievedChunk]
                                   | list[tuple[RetrievedChunk, float]]

Author: Bonna Bambilla, MSc Data Science, Arden University
Project: COM7014 Advanced Computing Project
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval import RetrievedChunk
from reranking.geo import haversine_km
from reranking.geocode import geocode
from reranking.normalise import (
    normalise_gstc,
    normalise_relevance,
    normalise_carbon,
)
from reranking.score import Weights, compute_score


def _carbon_norm_for_trip(
    origin: Optional[str],
    destination: Optional[str],
) -> float:
    """Compute the normalised carbon term for an origin->destination trip.

    Returns 0.0 (no penalty) if either endpoint is missing or cannot be
    geocoded. Computed once per query and reused for every chunk.
    """
    if not origin or not destination:
        return 0.0
    o = geocode(origin)
    d = geocode(destination)
    if o is None or d is None:
        return 0.0
    km = haversine_km(o[0], o[1], d[0], d[1])
    return normalise_carbon(km)


def rerank(
    chunks: list[RetrievedChunk],
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    weights: Weights = Weights(),
    return_scores: bool = False,
) -> Union[list[RetrievedChunk], list[tuple[RetrievedChunk, float]]]:
    """Rerank retrieved chunks by the multi-objective score.

    Args:
        chunks: retrieved chunks (each has tier_weight and distance).
        origin: user's departure location name (optional).
        destination: trip destination name (optional).
        weights: Weights(alpha, beta, gamma).
        return_scores: if True, return (chunk, score) pairs; else chunks.

    Returns:
        Chunks sorted by descending score (best first), or (chunk, score)
        pairs if return_scores is True.
    """
    if not chunks:
        return []

    # Carbon is the same for the whole trip - compute once.
    carbon_norm = _carbon_norm_for_trip(origin, destination)

    scored: list[tuple[RetrievedChunk, float]] = []
    for c in chunks:
        gstc_norm = normalise_gstc(c.tier_weight)
        relevance_norm = normalise_relevance(1.0 - c.distance)
        s = compute_score(gstc_norm, relevance_norm, carbon_norm, weights)
        scored.append((c, s))

    # Stable sort by score descending (ties keep original retrieval order)
    scored.sort(key=lambda pair: pair[1], reverse=True)

    if return_scores:
        return scored
    return [c for c, _ in scored]


if __name__ == "__main__":
    from dataclasses import dataclass

    # Lightweight fake chunk matching the fields rerank() reads.
    @dataclass
    class FakeChunk:
        chunk_id: str
        tier_weight: float
        distance: float
        source_name: str = "test"

    print("Reranker unit tests")
    print("=" * 60)
    results = []

    # --- Test 1: GSTC + relevance ordering (no carbon) ---
    # Chunk A: Tier 1 (1.5), close match (distance 0.1) -> should win
    # Chunk B: Tier 4 (0.4), weak match (distance 0.5) -> should lose
    a = FakeChunk("A", tier_weight=1.5, distance=0.1)
    b = FakeChunk("B", tier_weight=0.4, distance=0.5)
    order = rerank([b, a])  # deliberately pass B first
    t1 = (order[0].chunk_id == "A")
    results.append(t1)
    print(f"  [{'PASS' if t1 else 'FAIL'}] Tier1+close ranks above Tier4+weak "
          f"(got order: {[c.chunk_id for c in order]})")

    # --- Test 2: empty input ---
    t2 = (rerank([]) == [])
    results.append(t2)
    print(f"  [{'PASS' if t2 else 'FAIL'}] empty chunk list returns []")

    # --- Test 3: carbon is uniform per query (does NOT reorder) ---
    # With a real origin+destination, both chunks get the same carbon term,
    # so the GSTC+relevance ordering from Test 1 must be preserved.
    order_carbon = rerank([b, a], origin="Frankfurt", destination="Singapore")
    t3 = (order_carbon[0].chunk_id == "A")
    results.append(t3)
    print(f"  [{'PASS' if t3 else 'FAIL'}] carbon uniform: ordering preserved "
          f"with origin/destination (got: {[c.chunk_id for c in order_carbon]})")

    # --- Test 4: carbon actually computed for a known trip ---
    cn = _carbon_norm_for_trip("Frankfurt", "Singapore")
    t4 = abs(cn - 0.5129) < 0.005
    results.append(t4)
    print(f"  [{'PASS' if t4 else 'FAIL'}] Frankfurt->Singapore carbon_norm "
          f"= {cn:.4f} (expected ~0.5129)")

    # --- Test 5: carbon off when destination missing ---
    cn_off = _carbon_norm_for_trip("Frankfurt", None)
    t5 = (cn_off == 0.0)
    results.append(t5)
    print(f"  [{'PASS' if t5 else 'FAIL'}] carbon=0 when destination missing "
          f"(got {cn_off})")

    # --- Test 6: carbon off when destination ungeocodable ---
    cn_unknown = _carbon_norm_for_trip("Frankfurt", "Atlantis")
    t6 = (cn_unknown == 0.0)
    results.append(t6)
    print(f"  [{'PASS' if t6 else 'FAIL'}] carbon=0 for ungeocodable destination "
          f"(got {cn_unknown})")

    # --- Test 7: return_scores returns pairs ---
    pairs = rerank([a, b], return_scores=True)
    t7 = (len(pairs) == 2 and isinstance(pairs[0], tuple) and pairs[0][1] >= pairs[1][1])
    results.append(t7)
    print(f"  [{'PASS' if t7 else 'FAIL'}] return_scores gives sorted (chunk, score) pairs")

    print()
    print("=" * 60)
    print(f"OVERALL: {'ALL TESTS PASS' if all(results) else 'SOME TESTS FAILED'}")
