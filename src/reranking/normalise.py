"""Score-term normalisation for the reranker (fixed-reference).

normalise_gstc(tier_weight) -> [0,1] via /1.5
normalise_relevance(relevance) -> [0,1] via clamp
normalise_carbon(distance_km) -> [0,1] via /20000 then clamp

A normalised carbon value of 0.5 always means 10000 km, in every query,
keeping alpha/beta/gamma weights stable for the Phase 6 sensitivity grid.

Author: Bonna Bambilla, MSc Data Science, Arden University
Project: COM7014 Advanced Computing Project
"""
from __future__ import annotations

MAX_TIER_WEIGHT = 1.5
MAX_CARBON_KM = 20000.0


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def normalise_gstc(tier_weight: float) -> float:
    return _clamp01(tier_weight / MAX_TIER_WEIGHT)


def normalise_relevance(relevance: float) -> float:
    return _clamp01(relevance)


def normalise_carbon(distance_km: float) -> float:
    return _clamp01(distance_km / MAX_CARBON_KM)


if __name__ == "__main__":
    print("Normalisation unit tests (fixed-reference)")
    print("=" * 60)
    results = []

    def check(label, got, expected, tol=0.005):
        ok = abs(got - expected) <= tol
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: got {got:.4f}, expected {expected:.4f}")

    print("GSTC (/1.5):")
    check("Tier 1 (1.5)", normalise_gstc(1.5), 1.0)
    check("Tier 2 (1.0)", normalise_gstc(1.0), 0.6667)
    check("Tier 3 (0.7)", normalise_gstc(0.7), 0.4667)
    check("Tier 4 (0.4)", normalise_gstc(0.4), 0.2667)
    print("Relevance (clamp):")
    check("0.85 stays", normalise_relevance(0.85), 0.85)
    check("1.2 -> 1", normalise_relevance(1.2), 1.0)
    check("-0.1 -> 0", normalise_relevance(-0.1), 0.0)
    print("Carbon (/20000, clamp):")
    check("0 km", normalise_carbon(0), 0.0)
    check("478 km", normalise_carbon(478), 0.0239)
    check("10258 km", normalise_carbon(10258), 0.5129)
    check("10000 km == 0.5", normalise_carbon(10000), 0.5)
    check("16994 km", normalise_carbon(16994), 0.8497)
    check("25000 -> 1", normalise_carbon(25000), 1.0)
    print()
    print("=" * 60)
    print(f"OVERALL: {'ALL TESTS PASS' if all(results) else 'SOME TESTS FAILED'}")
