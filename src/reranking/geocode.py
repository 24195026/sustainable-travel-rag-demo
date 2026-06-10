"""Geocoding for the reranker carbon penalty.

Turns a destination name into (latitude, longitude) coordinates so the
reranker can compute Haversine distance from the user's origin.

Layered design:
  1. Curated coordinate table (offline, reproducible) - the workhorse.
     Covers ~200 major global destinations organised by region, matching
     the Phase 5 guided-form regions.
  2. Alias map for common name variations (Bali, Saigon, Mumbai, etc.).
  3. Optional live-geocoding fallback (geopy/Nominatim) for destinations
     not in the table. Network-gated: fires only where outbound HTTPS is
     allowed (deployment), dormant in the restricted Codespace.
  4. Graceful miss: returns None if all layers fail. Callers must treat
     None as "skip the carbon term" rather than crashing.

Public API:
    geocode(name) -> (lat, lon) | None
    geocode_detail(name) -> dict | None   # lat, lon, display_name, region, source

Author: Bonna Bambilla, MSc Data Science, Arden University
Project: COM7014 Advanced Computing Project
"""
from __future__ import annotations

from typing import Optional

# Coordinate table: normalised key -> (lat, lon, display_name, region)
# Coordinates are decimal degrees. Values are well-known city centroids.
_TABLE: dict[str, tuple[float, float, str, str]] = {
    # --- Europe ---
    "london": (51.5074, -0.1278, "London", "Europe"),
    "paris": (48.8566, 2.3522, "Paris", "Europe"),
    "rome": (41.9028, 12.4964, "Rome", "Europe"),
    "madrid": (40.4168, -3.7038, "Madrid", "Europe"),
    "barcelona": (41.3874, 2.1686, "Barcelona", "Europe"),
    "lisbon": (38.7223, -9.1393, "Lisbon", "Europe"),
    "amsterdam": (52.3676, 4.9041, "Amsterdam", "Europe"),
    "berlin": (52.5200, 13.4050, "Berlin", "Europe"),
    "munich": (48.1351, 11.5820, "Munich", "Europe"),
    "frankfurt": (50.1109, 8.6821, "Frankfurt", "Europe"),
    "vienna": (48.2082, 16.3738, "Vienna", "Europe"),
    "prague": (50.0755, 14.4378, "Prague", "Europe"),
    "budapest": (47.4979, 19.0402, "Budapest", "Europe"),
    "athens": (37.9838, 23.7275, "Athens", "Europe"),
    "venice": (45.4408, 12.3155, "Venice", "Europe"),
    "florence": (43.7696, 11.2558, "Florence", "Europe"),
    "milan": (45.4642, 9.1900, "Milan", "Europe"),
    "zurich": (47.3769, 8.5417, "Zurich", "Europe"),
    "geneva": (46.2044, 6.1432, "Geneva", "Europe"),
    "brussels": (50.8503, 4.3517, "Brussels", "Europe"),
    "copenhagen": (55.6761, 12.5683, "Copenhagen", "Europe"),
    "stockholm": (59.3293, 18.0686, "Stockholm", "Europe"),
    "oslo": (59.9139, 10.7522, "Oslo", "Europe"),
    "helsinki": (60.1699, 24.9384, "Helsinki", "Europe"),
    "reykjavik": (64.1466, -21.9426, "Reykjavik", "Europe"),
    "dublin": (53.3498, -6.2603, "Dublin", "Europe"),
    "edinburgh": (55.9533, -3.1883, "Edinburgh", "Europe"),
    "warsaw": (52.2297, 21.0122, "Warsaw", "Europe"),
    "krakow": (50.0647, 19.9450, "Krakow", "Europe"),
    "porto": (41.1579, -8.6291, "Porto", "Europe"),
    "seville": (37.3891, -5.9845, "Seville", "Europe"),
    "nice": (43.7102, 7.2620, "Nice", "Europe"),
    "zagreb": (45.8150, 15.9819, "Zagreb", "Europe"),
    "dubrovnik": (42.6507, 18.0944, "Dubrovnik", "Europe"),
    "split": (43.5081, 16.4402, "Split", "Europe"),
    "ljubljana": (46.0569, 14.5058, "Ljubljana", "Europe"),
    "tallinn": (59.4370, 24.7536, "Tallinn", "Europe"),
    "riga": (56.9496, 24.1052, "Riga", "Europe"),
    "vilnius": (54.6872, 25.2797, "Vilnius", "Europe"),
    "bucharest": (44.4268, 26.1025, "Bucharest", "Europe"),
    "sofia": (42.6977, 23.3219, "Sofia", "Europe"),
    "belgrade": (44.7866, 20.4489, "Belgrade", "Europe"),
    "istanbul": (41.0082, 28.9784, "Istanbul", "Europe"),
    "santorini": (36.3932, 25.4615, "Santorini", "Europe"),
    "mykonos": (37.4467, 25.3289, "Mykonos", "Europe"),
    "malaga": (36.7213, -4.4214, "Malaga", "Europe"),
    "valencia": (39.4699, -0.3763, "Valencia", "Europe"),
    "naples": (40.8518, 14.2681, "Naples", "Europe"),
    "hamburg": (53.5511, 9.9937, "Hamburg", "Europe"),

    # --- Asia (heavy SE Asia coverage for FAQ focus) ---
    "singapore": (1.3521, 103.8198, "Singapore", "Asia"),
    "bangkok": (13.7563, 100.5018, "Bangkok", "Asia"),
    "chiang mai": (18.7883, 98.9853, "Chiang Mai", "Asia"),
    "phuket": (7.8804, 98.3923, "Phuket", "Asia"),
    "denpasar": (-8.6705, 115.2126, "Bali (Denpasar)", "Asia"),
    "ubud": (-8.5069, 115.2625, "Ubud, Bali", "Asia"),
    "lombok": (-8.6500, 116.3249, "Lombok", "Asia"),
    "flores": (-8.6573, 121.0794, "Flores", "Asia"),
    "jakarta": (-6.2088, 106.8456, "Jakarta", "Asia"),
    "yogyakarta": (-7.7956, 110.3695, "Yogyakarta", "Asia"),
    "hanoi": (21.0278, 105.8342, "Hanoi", "Asia"),
    "ho chi minh city": (10.8231, 106.6297, "Ho Chi Minh City", "Asia"),
    "hoi an": (15.8801, 108.3380, "Hoi An", "Asia"),
    "kuala lumpur": (3.1390, 101.6869, "Kuala Lumpur", "Asia"),
    "penang": (5.4141, 100.3288, "Penang", "Asia"),
    "manila": (14.5995, 120.9842, "Manila", "Asia"),
    "cebu": (10.3157, 123.8854, "Cebu", "Asia"),
    "phnom penh": (11.5564, 104.9282, "Phnom Penh", "Asia"),
    "siem reap": (13.3671, 103.8448, "Siem Reap", "Asia"),
    "vientiane": (17.9757, 102.6331, "Vientiane", "Asia"),
    "luang prabang": (19.8834, 102.1347, "Luang Prabang", "Asia"),
    "yangon": (16.8409, 96.1735, "Yangon", "Asia"),
    "tokyo": (35.6762, 139.6503, "Tokyo", "Asia"),
    "kyoto": (35.0116, 135.7681, "Kyoto", "Asia"),
    "osaka": (34.6937, 135.5023, "Osaka", "Asia"),
    "seoul": (37.5665, 126.9780, "Seoul", "Asia"),
    "busan": (35.1796, 129.0756, "Busan", "Asia"),
    "taipei": (25.0330, 121.5654, "Taipei", "Asia"),
    "hong kong": (22.3193, 114.1694, "Hong Kong", "Asia"),
    "beijing": (39.9042, 116.4074, "Beijing", "Asia"),
    "shanghai": (31.2304, 121.4737, "Shanghai", "Asia"),
    "delhi": (28.7041, 77.1025, "Delhi", "Asia"),
    "mumbai": (19.0760, 72.8777, "Mumbai", "Asia"),
    "jaipur": (26.9124, 75.7873, "Jaipur", "Asia"),
    "goa": (15.2993, 74.1240, "Goa", "Asia"),
    "bengaluru": (12.9716, 77.5946, "Bengaluru", "Asia"),
    "kathmandu": (27.7172, 85.3240, "Kathmandu", "Asia"),
    "colombo": (6.9271, 79.8612, "Colombo", "Asia"),
    "kandy": (7.2906, 80.6337, "Kandy", "Asia"),
    "male": (4.1755, 73.5093, "Male, Maldives", "Asia"),
    "dubai": (25.2048, 55.2708, "Dubai", "Asia"),
    "abu dhabi": (24.4539, 54.3773, "Abu Dhabi", "Asia"),
    "doha": (25.2854, 51.5310, "Doha", "Asia"),
    "amman": (31.9454, 35.9284, "Amman", "Asia"),
    "tel aviv": (32.0853, 34.7818, "Tel Aviv", "Asia"),
    "muscat": (23.5880, 58.3829, "Muscat", "Asia"),

    # --- Africa ---
    "cape town": (-33.9249, 18.4241, "Cape Town", "Africa"),
    "johannesburg": (-26.2041, 28.0473, "Johannesburg", "Africa"),
    "marrakech": (31.6295, -7.9811, "Marrakech", "Africa"),
    "casablanca": (33.5731, -7.5898, "Casablanca", "Africa"),
    "fes": (34.0181, -5.0078, "Fes", "Africa"),
    "cairo": (30.0444, 31.2357, "Cairo", "Africa"),
    "nairobi": (-1.2921, 36.8219, "Nairobi", "Africa"),
    "zanzibar": (-6.1659, 39.2026, "Zanzibar", "Africa"),
    "dar es salaam": (-6.7924, 39.2083, "Dar es Salaam", "Africa"),
    "addis ababa": (9.0320, 38.7469, "Addis Ababa", "Africa"),
    "kigali": (-1.9706, 30.1044, "Kigali", "Africa"),
    "accra": (5.6037, -0.1870, "Accra", "Africa"),
    "lagos": (6.5244, 3.3792, "Lagos", "Africa"),
    "dakar": (14.7167, -17.4677, "Dakar", "Africa"),
    "tunis": (36.8065, 10.1815, "Tunis", "Africa"),
    "windhoek": (-22.5597, 17.0832, "Windhoek", "Africa"),
    "victoria falls": (-17.9243, 25.8572, "Victoria Falls", "Africa"),
    "maputo": (-25.9692, 32.5732, "Maputo", "Africa"),
    "kampala": (0.3476, 32.5825, "Kampala", "Africa"),
    "mombasa": (-4.0435, 39.6682, "Mombasa", "Africa"),
    "port louis": (-20.1609, 57.5012, "Port Louis, Mauritius", "Africa"),
    "antananarivo": (-18.8792, 47.5079, "Antananarivo", "Africa"),
    "gaborone": (-24.6282, 25.9231, "Gaborone", "Africa"),
    "luxor": (25.6872, 32.6396, "Luxor", "Africa"),
    "essaouira": (31.5085, -9.7595, "Essaouira", "Africa"),

    # --- Latin America ---
    "mexico city": (19.4326, -99.1332, "Mexico City", "Latin America"),
    "cancun": (21.1619, -86.8515, "Cancun", "Latin America"),
    "oaxaca": (17.0732, -96.7266, "Oaxaca", "Latin America"),
    "guadalajara": (20.6597, -103.3496, "Guadalajara", "Latin America"),
    "lima": (-12.0464, -77.0428, "Lima", "Latin America"),
    "cusco": (-13.5320, -71.9675, "Cusco", "Latin America"),
    "la paz": (-16.4897, -68.1193, "La Paz", "Latin America"),
    "buenos aires": (-34.6037, -58.3816, "Buenos Aires", "Latin America"),
    "santiago": (-33.4489, -70.6693, "Santiago", "Latin America"),
    "rio de janeiro": (-22.9068, -43.1729, "Rio de Janeiro", "Latin America"),
    "sao paulo": (-23.5505, -46.6333, "Sao Paulo", "Latin America"),
    "bogota": (4.7110, -74.0721, "Bogota", "Latin America"),
    "cartagena": (10.3910, -75.4794, "Cartagena", "Latin America"),
    "medellin": (6.2476, -75.5658, "Medellin", "Latin America"),
    "quito": (-0.1807, -78.4678, "Quito", "Latin America"),
    "san jose": (9.9281, -84.0907, "San Jose, Costa Rica", "Latin America"),
    "panama city": (8.9824, -79.5199, "Panama City", "Latin America"),
    "guatemala city": (14.6349, -90.5069, "Guatemala City", "Latin America"),
    "antigua": (14.5586, -90.7295, "Antigua, Guatemala", "Latin America"),
    "havana": (23.1136, -82.3666, "Havana", "Latin America"),
    "montevideo": (-34.9011, -56.1645, "Montevideo", "Latin America"),
    "galapagos": (-0.7393, -90.3163, "Galapagos", "Latin America"),
    "tulum": (20.2114, -87.4654, "Tulum", "Latin America"),
    "merida": (20.9674, -89.5926, "Merida", "Latin America"),
    "valparaiso": (-33.0472, -71.6127, "Valparaiso", "Latin America"),

    # --- Oceania & Other ---
    "sydney": (-33.8688, 151.2093, "Sydney", "Oceania"),
    "melbourne": (-37.8136, 144.9631, "Melbourne", "Oceania"),
    "brisbane": (-27.4698, 153.0251, "Brisbane", "Oceania"),
    "perth": (-31.9505, 115.8605, "Perth", "Oceania"),
    "auckland": (-36.8485, 174.7633, "Auckland", "Oceania"),
    "wellington": (-41.2865, 174.7762, "Wellington", "Oceania"),
    "queenstown": (-45.0312, 168.6626, "Queenstown", "Oceania"),
    "nadi": (-17.7765, 177.4356, "Nadi, Fiji", "Oceania"),
    "honolulu": (21.3069, -157.8583, "Honolulu", "Oceania"),
    "papeete": (-17.5516, -149.5585, "Papeete, Tahiti", "Oceania"),

    # --- North America (common origins/destinations) ---
    "new york": (40.7128, -74.0060, "New York", "North America"),
    "los angeles": (34.0522, -118.2437, "Los Angeles", "North America"),
    "san francisco": (37.7749, -122.4194, "San Francisco", "North America"),
    "toronto": (43.6532, -79.3832, "Toronto", "North America"),
    "vancouver": (49.2827, -123.1207, "Vancouver", "North America"),
    "chicago": (41.8781, -87.6298, "Chicago", "North America"),
    "miami": (25.7617, -80.1918, "Miami", "North America"),
    "montreal": (45.5017, -73.5673, "Montreal", "North America"),
}

# Alias map: common alternative spellings/names -> canonical key
_ALIASES: dict[str, str] = {
    "bali": "denpasar",
    "saigon": "ho chi minh city",
    "hcmc": "ho chi minh city",
    "bombay": "mumbai",
    "bangalore": "bengaluru",
    "maldives": "male",
    "fez": "fes",
    "nyc": "new york",
    "sf": "san francisco",
    "la": "los angeles",
    "rio": "rio de janeiro",
    "costa rica": "san jose",
    "fiji": "nadi",
    "tahiti": "papeete",
    "mauritius": "port louis",
    "krung thep": "bangkok",
    "peking": "beijing",
}


def _normalise(name: str) -> str:
    return name.strip().lower()


def geocode_detail(name: str) -> Optional[dict]:
    """Resolve a destination name to full detail, or None if not found.

    Lookup order: table -> alias -> live fallback (if network) -> None.
    """
    if not name or not name.strip():
        return None
    key = _normalise(name)

    # Layer 1: direct table hit
    if key in _TABLE:
        lat, lon, disp, region = _TABLE[key]
        return {"lat": lat, "lon": lon, "display_name": disp,
                "region": region, "source": "table"}

    # Layer 2: alias -> table
    if key in _ALIASES:
        canon = _ALIASES[key]
        lat, lon, disp, region = _TABLE[canon]
        return {"lat": lat, "lon": lon, "display_name": disp,
                "region": region, "source": "alias"}

    # Layer 3: live geocoding fallback (network-gated, dormant if blocked)
    live = _live_geocode(name)
    if live is not None:
        return live

    # Layer 4: graceful miss
    return None


def geocode(name: str) -> Optional[tuple[float, float]]:
    """Resolve a destination name to (lat, lon), or None if not found."""
    detail = geocode_detail(name)
    if detail is None:
        return None
    return (detail["lat"], detail["lon"])


def _live_geocode(name: str) -> Optional[dict]:
    """Optional live fallback via geopy/Nominatim. Returns None on any
    failure (no network, not installed, no result, rate limit). This is
    dormant in the restricted Codespace (egress blocked) and active in
    deployment environments that permit outbound HTTPS.
    """
    try:
        from geopy.geocoders import Nominatim  # type: ignore
        geocoder = Nominatim(user_agent="sustainable-travel-rag")
        loc = geocoder.geocode(name, timeout=5)
        if loc is None:
            return None
        return {"lat": loc.latitude, "lon": loc.longitude,
                "display_name": name.strip().title(),
                "region": "Unknown", "source": "live"}
    except Exception:
        return None


if __name__ == "__main__":
    print(f"Coordinate table size: {len(_TABLE)} destinations")
    print(f"Alias entries: {len(_ALIASES)}")
    print("=" * 60)

    # Lookup tests across all layers
    tests = [
        ("Singapore", "table"),
        ("frankfurt", "table"),
        ("  London  ", "table"),     # whitespace + case
        ("Bali", "alias"),           # alias -> denpasar
        ("Saigon", "alias"),         # alias -> ho chi minh city
        ("Bombay", "alias"),         # alias -> mumbai
        ("Atlantis", None),          # genuine miss (live blocked here)
    ]
    print("Lookup tests:")
    for name, expected_source in tests:
        d = geocode_detail(name)
        if d is None:
            got = None
            line = f"  '{name}' -> None"
        else:
            got = d["source"]
            line = f"  '{name}' -> ({d['lat']}, {d['lon']}) {d['display_name']} [{d['source']}]"
        ok = (got == expected_source)
        print(f"  [{'PASS' if ok else 'FAIL'}]{line}")

    # Cross-check against proven Haversine: Frankfurt -> Singapore
    print()
    print("Cross-check vs proven Haversine (Frankfurt -> Singapore):")
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parent))
    from geo import haversine_km
    f = geocode("Frankfurt")
    s = geocode("Singapore")
    dist = haversine_km(f[0], f[1], s[0], s[1])
    print(f"  Table coords -> {dist:.0f} km (unit test earlier gave 10258 km)")
    print(f"  Match: {'YES' if abs(dist - 10258) < 50 else 'NO - INVESTIGATE'}")
