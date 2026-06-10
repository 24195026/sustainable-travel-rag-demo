"""Multi-objective score function for the reranker.

Combines three ALREADY-NORMALISED terms (each in [0,1]) into a single
ranking score:

    score = alpha * gstc_norm + beta * relevance_norm - gamma * carbon_norm

This module is the pure formula only. It does NOT geocode, compute
distance, or normalise - those are handled upstream (geo.py, geocode.py,
normalise.py) and orchestrated in rerank() (reranker.py). Keeping the
formula pure makes it trivially testable and keeps the alpha/beta/gamma
weights as clean parameters for the Phase 6 sensitivity grid.

Public API:
    Weights(alpha, beta, gamma)            # immutable weight bundle
    compute_score(gstc, relevance, carbon, weights) -> float

Author: Bonna Bambilla, MSc Data Science, Arden University
Project: COM7014 Advanced Computing Project
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Weights:
    """Immutable bundle of reranker weights.

    alpha: weight on GSTC credibility (tier weight, normalised)
    beta:  weight on retrieval relevance (normalised)
    gamma: weight on the carbon penalty (subtracted)

    Defaults are relevance-and-credibility-led with a gentle carbon nudge;
    the Phase 6 sensitivity grid explores alternatives empirically. Weights
    are not constrained to sum to 1 (the carbon term is subtracted, so a
    sum-to-one constraint has no clean meaning here).
    """
    alpha: float = 0.5
    beta: float = 0.3
    gamma: float = 0.2


def compute_score(
    gstc: float,
    relevance: float,
    carbon: float,
    weights: Weights = Weights(),
) -> float:
    """Compute the reranker score from normalised terms.

    Args:
        gstc:      normalised GSTC credibility in [0, 1]
        relevance: normalised retrieval relevance in [0, 1]
        carbon:    normalised carbon penalty in [0, 1] (0 = no penalty)
        weights:   a Weights instance (defaults if omitted)

    Returns:
        A float score. Higher is better. The carbon term is subtracted,
        so the score can in principle be negative if gamma*carbon exceeds
        the positive terms; relative ordering is what matters for ranking.
    """
    return (
        weights.alpha * gstc
        + weights.beta * relevance
        - weights.gamma * carbon
    )


if __name__ == "__main__":
    print("Score function unit tests")
    print("=" * 60)
    results = []
    w = Weights()  # defaults 0.5 / 0.3 / 0.2

    def check(label, got, expected, tol=0.0001):
        ok = abs(got - expected) <= tol
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: got {got:.4f}, expected {expected:.4f}")

    print(f"Default weights: alpha={w.alpha}, beta={w.beta}, gamma={w.gamma}")
    print()

    # Arithmetic checks
    print("Arithmetic:")
    check("all 1.0 -> 0.5+0.3-0.2", compute_score(1.0, 1.0, 1.0, w), 0.6)
    check("all 0.0", compute_score(0.0, 0.0, 0.0, w), 0.0)
    check("gstc 1, rel 0, carbon 0", compute_score(1.0, 0.0, 0.0, w), 0.5)
    check("gstc 0, rel 1, carbon 0", compute_score(0.0, 1.0, 0.0, w), 0.3)
    check("gstc 0, rel 0, carbon 1", compute_score(0.0, 0.0, 1.0, w), -0.2)

    # Behaviour checks - the carbon penalty must prefer nearer destinations
    print()
    print("Behaviour (carbon penalty prefers nearer, all else equal):")
    near = compute_score(1.0, 0.9, 0.05, w)   # Tier1, relevant, ~1000 km
    far  = compute_score(1.0, 0.9, 0.85, w)   # Tier1, relevant, ~17000 km
    ok_near_beats_far = near > far
    results.append(ok_near_beats_far)
    print(f"  [{'PASS' if ok_near_beats_far else 'FAIL'}] near ({near:.4f}) > far ({far:.4f})")

    # Credibility check - higher tier wins, all else equal
    print()
    print("Behaviour (higher GSTC tier wins, all else equal):")
    t1 = compute_score(1.0, 0.8, 0.3, w)      # Tier 1 normalised
    t4 = compute_score(0.2667, 0.8, 0.3, w)   # Tier 4 normalised
    ok_t1_beats_t4 = t1 > t4
    results.append(ok_t1_beats_t4)
    print(f"  [{'PASS' if ok_t1_beats_t4 else 'FAIL'}] Tier1 ({t1:.4f}) > Tier4 ({t4:.4f})")

    # Custom weights work
    print()
    print("Custom weights:")
    w2 = Weights(alpha=0.2, beta=0.2, gamma=0.6)
    check("carbon-heavy weights", compute_score(1.0, 1.0, 1.0, w2), -0.2)

    # Frozen dataclass cannot be mutated
    print()
    print("Immutability:")
    try:
        w.alpha = 0.9  # type: ignore
        results.append(False)
        print("  [FAIL] Weights was mutable (should be frozen)")
    except Exception:
        results.append(True)
        print("  [PASS] Weights is frozen (immutable)")

    print()
    print("=" * 60)
    print(f"OVERALL: {'ALL TESTS PASS' if all(results) else 'SOME TESTS FAILED'}")
