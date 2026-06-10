"""Geographic distance utilities for the reranker carbon penalty.

The reranker penalises destinations that are far from the user's origin,
as a proxy for travel carbon cost. This module provides the great-circle
distance between two latitude/longitude points using the Haversine
formula.

Public API:
    haversine_km(lat1, lon1, lat2, lon2) -> float

Author: Bonna Bambilla, MSc Data Science, Arden University
Project: COM7014 Advanced Computing Project
"""
from __future__ import annotations

import math

# Mean radius of the Earth in kilometres (IUGG standard).
_EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometres.

    Uses the Haversine formula. Inputs are decimal degrees
    (e.g., London = 51.5074, -0.1278).

    Args:
        lat1, lon1: latitude and longitude of point 1, in degrees.
        lat2, lon2: latitude and longitude of point 2, in degrees.

    Returns:
        Distance in kilometres (float, >= 0).
    """
    # Convert decimal degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return _EARTH_RADIUS_KM * c


if __name__ == "__main__":
    # Unit test against known city-pair distances (published great-circle
    # figures, tolerance +/- 2% to allow for coordinate rounding).
    cities = {
        "London": (51.5074, -0.1278),
        "Paris": (48.8566, 2.3522),
        "Frankfurt": (50.1109, 8.6821),
        "Singapore": (1.3521, 103.8198),
        "New York": (40.7128, -74.0060),
        "Sydney": (-33.8688, 151.2093),
    }

    # (city_a, city_b, expected_km) - expected from public great-circle data
    test_cases = [
        ("London", "Paris", 344),
        ("Frankfurt", "Singapore", 10279),
        ("London", "New York", 5570),
        ("Sydney", "London", 16988),
        ("Frankfurt", "Paris", 479),
    ]

    print("Haversine unit test (tolerance +/- 2%)")
    print("=" * 60)
    all_pass = True
    for a, b, expected in test_cases:
        lat1, lon1 = cities[a]
        lat2, lon2 = cities[b]
        got = haversine_km(lat1, lon1, lat2, lon2)
        tolerance = expected * 0.02
        ok = abs(got - expected) <= tolerance
        all_pass = all_pass and ok
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {a} -> {b}: got {got:.0f} km, expected ~{expected} km")

    # Sanity checks
    print()
    print("Sanity checks:")
    same = haversine_km(51.5074, -0.1278, 51.5074, -0.1278)
    print(f"  Same point -> {same:.4f} km (expected 0.0)")
    all_pass = all_pass and (same < 0.001)

    print()
    print("=" * 60)
    print(f"OVERALL: {'ALL TESTS PASS' if all_pass else 'SOME TESTS FAILED'}")
