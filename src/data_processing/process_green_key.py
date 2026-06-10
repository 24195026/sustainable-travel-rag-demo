"""
Process the Green Key Certified Sites database into three regional CSVs.

Source: data/supplementary/Green Key Certified Sites.xlsx (8,500+ properties)
Output: data/curated/green_key_sites_<region>.csv

Per the project's source ingestion strategy (see docs/sources_registry.md
Asset A and research_log.md 2026-04-25 entry), the full database is NOT
ingested into ChromaDB. These regional subsets exist to give the chatbot
concrete named examples for FAQ_011 (budget accommodation, Southeast Asia)
and similar regional queries, without flooding retrieval.
"""
from pathlib import Path
import pandas as pd

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_FILE = REPO_ROOT / "data" / "supplementary" / "Green Key Certified Sites.xlsx"
OUTPUT_DIR = REPO_ROOT / "data" / "curated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Columns to keep (in the order they will appear in the CSV)
COLUMNS_TO_KEEP = [
    "Establishment Name",
    "Country",
    "City",
    "Establishment Category",
    "Certified until",
    "Website",
]

# Region definitions — using ISO 3166-1 alpha-3 country codes
# (Green Key data uses codes, not full country names)
REGIONS = {
    "southeast_asia": {
        "BRN", "KHM", "IDN", "LAO", "MYS", "MMR",
        "PHL", "SGP", "THA", "TLS", "VNM",
    },
    "africa": {
        "DZA", "AGO", "BEN", "BWA", "BFA", "BDI", "CPV", "CMR", "CAF", "TCD",
        "COM", "COG", "COD", "CIV", "DJI", "EGY", "GNQ", "ERI", "SWZ", "ETH",
        "GAB", "GMB", "GHA", "GIN", "GNB", "KEN", "LSO", "LBR", "LBY", "MDG",
        "MWI", "MLI", "MRT", "MUS", "MAR", "MOZ", "NAM", "NER", "NGA", "RWA",
        "STP", "SEN", "SYC", "SLE", "SOM", "ZAF", "SSD", "SDN", "TZA", "TGO",
        "TUN", "UGA", "ZMB", "ZWE",
    },
    "latin_america": {
        "ARG", "BLZ", "BOL", "BRA", "CHL", "COL", "CRI", "CUB", "DOM", "ECU",
        "SLV", "GTM", "HTI", "HND", "MEX", "NIC", "PAN", "PRY", "PER", "URY",
        "VEN",
        "ABW", "BMU", "CUW", "CYM", "KNA", "SXM", "TCA", "TTO", "PRI",
    },
}


def main() -> None:
    print(f"Reading: {SOURCE_FILE}")
    if not SOURCE_FILE.exists():
        print("ERROR: Source file not found.")
        print(f"Expected at: {SOURCE_FILE}")
        return

    df = pd.read_excel(SOURCE_FILE, engine="openpyxl")

    # Source data has whitespace issues in column names (e.g., 'Country '
    # has a trailing space, 'Establishment  Name' has a double space).
    # Normalise to single-spaced, stripped versions before any column lookup.
    df.columns = [" ".join(c.split()) for c in df.columns]

    print(f"Total rows in source: {len(df):,}")

    # Strip whitespace from Country column for safe matching
    df["Country"] = df["Country"].astype(str).str.strip()

    # Verify columns we need are present
    missing = [c for c in COLUMNS_TO_KEEP if c not in df.columns]
    if missing:
        print(f"ERROR: Missing expected columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        return

    # Process each region
    for region_key, country_set in REGIONS.items():
        subset = df[df["Country"].isin(country_set)][COLUMNS_TO_KEEP].copy()
        subset = subset.sort_values(["Country", "City", "Establishment Name"])

        output_path = OUTPUT_DIR / f"green_key_sites_{region_key}.csv"
        subset.to_csv(output_path, index=False)

        unique_countries = subset["Country"].nunique()
        print(
            f"  {region_key:<16} -> {len(subset):>5} properties "
            f"across {unique_countries:>2} countries -> {output_path.name}"
        )

    # Sanity check: countries in source not matched to any region
    all_known = set().union(*REGIONS.values())
    in_source = set(df["Country"].dropna().unique())
    europe_and_other = sorted(in_source - all_known)
    print(
        f"\nFor reference: {len(europe_and_other)} countries in source "
        f"are outside our 3 target regions (mostly Europe + N. America + Oceania)."
    )


if __name__ == "__main__":
    main()