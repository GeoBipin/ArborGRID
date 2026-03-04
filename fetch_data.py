# =============================================================================
# ArborGrid 2.0 | Phase 1: Multi-Source Data Ingestion
# File: fetch_data.py
# Run:  & $PY fetch_data.py
# Outputs:
#   data/powerlines.geojson   — OSM power lines  (saved as WGS84)
#   data/facilities.geojson   — OSM hospitals + schools (saved as WGS84)
#   data/trees_raw.geojson    — Vancouver public trees  (saved as WGS84)
# =============================================================================

import sys
import time
import requests
import geopandas as gpd
import osmnx as ox
from pathlib import Path
from io import BytesIO

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR   = Path("C:/ArborGRID")
DATA_DIR   = BASE_DIR / "data"
TARGET_CRS = "EPSG:26910"      # UTM Zone 10N — metric
WGS84      = "EPSG:4326"
PLACE      = "Vancouver, BC, Canada"

TREES_EXPORT_URL = (
    "https://opendata.vancouver.ca/api/explore/v2.1/catalog/datasets/"
    "public-trees/exports/geojson"
    "?timezone=America%2FVancouver&use_labels=false"
)

# =============================================================================
# HELPERS
# =============================================================================

def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[✔] Output directory ready: {DATA_DIR}")


def save_geojson(gdf: gpd.GeoDataFrame, filename: str) -> Path:
    """Save as WGS84 GeoJSON (compute_risk.py reprojects on load)."""
    out = gdf.to_crs(WGS84) if gdf.crs.to_epsg() != 4326 else gdf
    out_path = DATA_DIR / filename
    out.to_file(out_path, driver="GeoJSON")
    mb = out_path.stat().st_size / 1_048_576
    print(f"    → Saved: {filename}  ({len(gdf):,} features | {mb:.2f} MB)")
    return out_path


def print_banner(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def download_file(url: str, label: str, timeout: int = 180) -> bytes:
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            total  = int(r.headers.get("content-length", 0))
            done   = 0
            chunks = []
            for chunk in r.iter_content(chunk_size=262144):
                if chunk:
                    chunks.append(chunk)
                    done += len(chunk)
                    mb   = done / 1_048_576
                    pct  = f"{done/total*100:.0f}%" if total else "..."
                    print(f"    {label}: {mb:.1f} MB ({pct})",
                          end="\r", flush=True)
            print()
            return b"".join(chunks)
    except requests.exceptions.RequestException as e:
        print(f"\n[✘] Download failed for {label}: {e}")
        sys.exit(1)

# =============================================================================
# STEP 1 — OSM POWERLINES
# =============================================================================

def fetch_powerlines() -> gpd.GeoDataFrame:
    print_banner("STEP 1/3 — OSM Powerlines")
    print(f"[→] Querying Overpass for power lines in: {PLACE}")
    try:
        gdf = ox.features_from_place(
            PLACE, tags={"power": ["line", "minor_line"]})
    except Exception as e:
        print(f"[✘] OSMnx query failed: {e}")
        sys.exit(1)

    gdf = gdf[gdf.geometry.geom_type.isin(
        ["LineString", "MultiLineString"])].copy()
    keep = ["geometry", "power", "voltage", "operator", "name"]
    gdf  = gdf[[c for c in keep if c in gdf.columns]].copy()
    gdf  = gdf.set_crs(WGS84, allow_override=True).to_crs(TARGET_CRS)
    gdf["length_m"] = gdf.geometry.length.round(2)
    gdf  = gdf.reset_index(drop=True)

    print(f"[✔] {len(gdf):,} segments | "
          f"{gdf['length_m'].sum()/1000:.2f} km total")
    save_geojson(gdf, "powerlines.geojson")
    return gdf

# =============================================================================
# STEP 2 — OSM CRITICAL FACILITIES
# =============================================================================

def fetch_facilities() -> gpd.GeoDataFrame:
    print_banner("STEP 2/3 — OSM Critical Facilities")
    print(f"[→] Querying Overpass for hospitals & schools in: {PLACE}")
    try:
        gdf = ox.features_from_place(
            PLACE, tags={"amenity": ["hospital", "school"]})
    except Exception as e:
        print(f"[✘] OSMnx query failed: {e}")
        sys.exit(1)

    keep = ["geometry", "amenity", "name", "addr:street"]
    gdf  = gdf[[c for c in keep if c in gdf.columns]].copy()
    gdf  = gdf.set_crs(WGS84, allow_override=True).to_crs(TARGET_CRS)
    gdf  = gdf.reset_index(drop=True)

    print(f"[✔] Hospitals: {(gdf['amenity']=='hospital').sum()} | "
          f"Schools: {(gdf['amenity']=='school').sum()}")
    save_geojson(gdf, "facilities.geojson")
    return gdf

# =============================================================================
# STEP 3 — VANCOUVER PUBLIC TREES
# =============================================================================

def fetch_vancouver_trees() -> gpd.GeoDataFrame:
    print_banner("STEP 3/3 — Vancouver Public Trees (Bulk Export)")
    print("[→] Downloading ~52 MB GeoJSON export...")
    raw = download_file(TREES_EXPORT_URL, "Trees GeoJSON")
    print(f"[✔] Downloaded: {len(raw)/1_048_576:.2f} MB")
    print("[→] Parsing (~30s)...")
    try:
        gdf = gpd.read_file(BytesIO(raw))
    except Exception as e:
        print(f"[✘] Parse failed: {e}")
        sys.exit(1)

    # Normalise column names
    gdf.columns = [c.lower().strip() for c in gdf.columns]

    rename = {
        "treeid":             "tree_id",
        "asset_id":           "tree_id",
        "civicnumber":        "civic_number",
        "stdstreet":          "std_street",
        "address":            "std_street",
        "genusname":          "genus_name",
        "speciesname":        "species_name",
        "commonname":         "common_name",
        "cultivar_name":      "cultivar_name",
        "heightrangeid":      "height_range",
        "height_range_id":    "height_range",
        "height_m":           "height_range",
        "diameter_cm":        "diameter",
        "diametre":           "diameter",
        "neighbourhoodname":  "neighbourhood",
        "neighbourhood_name": "neighbourhood",
        "date_planted":       "date_planted",
    }
    gdf = gdf.rename(
        columns={k: v for k, v in rename.items() if k in gdf.columns})

    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84)
    gdf = gdf.to_crs(TARGET_CRS)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf = gdf.reset_index(drop=True)

    print(f"[✔] {len(gdf):,} trees | Reprojected to {TARGET_CRS}")
    print(f"[i] Columns: {list(gdf.columns)}")

    if "height_range" in gdf.columns:
        print(f"[i] height_range range: "
              f"{gdf['height_range'].min()} – {gdf['height_range'].max()}")
    if "diameter" in gdf.columns:
        print(f"[i] diameter range   : "
              f"{gdf['diameter'].min()} – {gdf['diameter'].max()}")

    if "common_name" in gdf.columns:
        for sp, n in gdf["common_name"].value_counts().head(3).items():
            print(f"    {str(sp):<40} {n:>6,} trees")

    save_geojson(gdf, "trees_raw.geojson")
    return gdf

# =============================================================================
# MAIN
# =============================================================================

def main():
    start = time.time()
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║       ArborGrid 2.0 — Phase 1: Data Ingestion            ║")
    print("╚══════════════════════════════════════════════════════════╝")

    ensure_dirs()
    powerlines = fetch_powerlines()
    facilities = fetch_facilities()
    trees      = fetch_vancouver_trees()

    m, s = divmod(int(time.time() - start), 60)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║              ✅ PHASE 1 COMPLETE                          ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Powerline segments : {len(powerlines):>6,}                          ║")
    print(f"║  Critical facilities: {len(facilities):>6,}                          ║")
    print(f"║  Public trees       : {len(trees):>6,}                          ║")
    print(f"║  Elapsed time       : {m}m {s:02d}s                           ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  Files saved → C:/ArborGRID/data/                        ║")
    print("║   • powerlines.geojson                                   ║")
    print("║   • facilities.geojson                                   ║")
    print("║   • trees_raw.geojson                                    ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  Next → & $PY compute_risk.py                            ║")
    print("╚══════════════════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    main()