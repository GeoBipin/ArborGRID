# =============================================================================
# ArborGrid 2.0 | Tree Data Summary Extractor
# File: extract_summary.py
# Run: & $PY extract_summary.py
#
# Purpose: Reads trees_raw.geojson and prints a full schema + statistical
#          summary so we know exactly what data is available for the dashboard.
#          Also saves a compact summary CSV to data/tree_summary.csv.
# =============================================================================

import warnings
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

DATA_DIR  = Path("C:/ArborGRID/data")
TREES_F   = DATA_DIR / "trees_raw.geojson"
OUT_CSV   = DATA_DIR / "tree_summary.csv"

print("\n╔══════════════════════════════════════════════════════════╗")
print("║   ArborGrid 2.0 — Tree Data Summary Extractor           ║")
print("╚══════════════════════════════════════════════════════════╝\n")

if not TREES_F.exists():
    print(f"[✘] Not found: {TREES_F} — run fetch_data.py first.")
    raise SystemExit(1)

gdf = gpd.read_file(TREES_F)
print(f"[✔] Loaded {len(gdf):,} trees  |  CRS: {gdf.crs}")

# =============================================================================
# 1. SCHEMA — every column, its dtype, non-null count, null count
# =============================================================================
print("\n" + "="*62)
print("  SCHEMA — Columns & Null Analysis")
print("="*62)

schema_rows = []
for col in gdf.columns:
    if col == "geometry":
        continue
    dtype     = str(gdf[col].dtype)
    non_null  = int(gdf[col].notna().sum())
    null_cnt  = int(gdf[col].isna().sum())
    null_pct  = null_cnt / len(gdf) * 100
    schema_rows.append({
        "column":    col,
        "dtype":     dtype,
        "non_null":  non_null,
        "nulls":     null_cnt,
        "null_%":    round(null_pct, 1),
    })
    print(f"  {col:<25} {dtype:<12} {non_null:>7,} non-null  |  "
          f"{null_cnt:>7,} nulls ({null_pct:.1f}%)")

# =============================================================================
# 2. HEIGHT (height_range / height_m) STATS
# =============================================================================
ht_col = next((c for c in ["height_range", "height_m"] if c in gdf.columns), None)

print("\n" + "="*62)
print("  HEIGHT STATISTICS")
print("="*62)

if ht_col:
    h = pd.to_numeric(gdf[ht_col], errors="coerce").dropna()
    print(f"  Column used   : '{ht_col}'")
    print(f"  Count (non-null): {len(h):,}")
    print(f"  Min           : {h.min():.1f} m")
    print(f"  Max           : {h.max():.1f} m")
    print(f"  Mean          : {h.mean():.1f} m")
    print(f"  Median        : {h.median():.1f} m")
    print(f"  Std dev       : {h.std():.1f} m")
    print(f"\n  Height distribution:")
    bins   = [0, 5, 10, 15, 20, 25, 9999]
    labels = ["<5m (shrub/young)", "5–10m (small)", "10–15m (medium)",
              "15–20m (tall ⚠️)", "20–25m (very tall ⚠️)", "25m+ (critical ⚠️)"]
    cut = pd.cut(h, bins=bins, labels=labels, right=False)
    for lab, cnt in cut.value_counts().sort_index().items():
        pct = cnt / len(h) * 100
        bar = "█" * int(pct / 2)
        print(f"    {str(lab):<26} {cnt:>7,}  ({pct:5.1f}%)  {bar}")
else:
    print("  [!] No height column found.")

# =============================================================================
# 3. DIAMETER STATS
# =============================================================================
dm_col = next((c for c in ["diameter", "diameter_cm"] if c in gdf.columns), None)

print("\n" + "="*62)
print("  DIAMETER STATISTICS (cm)")
print("="*62)

if dm_col:
    d = pd.to_numeric(gdf[dm_col], errors="coerce").dropna()
    print(f"  Column used   : '{dm_col}'")
    print(f"  Count (non-null): {len(d):,}")
    print(f"  Min           : {d.min():.1f} cm")
    print(f"  Max           : {d.max():.1f} cm")
    print(f"  Mean          : {d.mean():.1f} cm")
    print(f"  Median        : {d.median():.1f} cm")
else:
    print("  [!] No diameter column found.")

# =============================================================================
# 4. TOP SPECIES
# =============================================================================
print("\n" + "="*62)
print("  TOP 20 SPECIES (common_name)")
print("="*62)

if "common_name" in gdf.columns:
    for name, cnt in gdf["common_name"].value_counts().head(20).items():
        pct = cnt / len(gdf) * 100
        print(f"  {str(name):<40} {cnt:>7,}  ({pct:.1f}%)")
else:
    print("  [!] No common_name column found.")

# =============================================================================
# 5. GENUS BREAKDOWN
# =============================================================================
print("\n" + "="*62)
print("  TOP 15 GENERA (genus_name)")
print("="*62)

if "genus_name" in gdf.columns:
    for name, cnt in gdf["genus_name"].value_counts().head(15).items():
        pct = cnt / len(gdf) * 100
        print(f"  {str(name):<30} {cnt:>7,}  ({pct:.1f}%)")

# =============================================================================
# 6. NEIGHBOURHOOD BREAKDOWN
# =============================================================================
print("\n" + "="*62)
print("  TREES BY NEIGHBOURHOOD")
print("="*62)

nb_col = next((c for c in ["neighbourhood", "neighbourhood_name"] if c in gdf.columns), None)
if nb_col:
    for name, cnt in gdf[nb_col].value_counts().items():
        pct = cnt / len(gdf) * 100
        bar = "█" * int(pct * 2)
        print(f"  {str(name):<35} {cnt:>6,}  ({pct:4.1f}%)  {bar}")
else:
    print("  [!] No neighbourhood column found.")

# =============================================================================
# 7. DATE PLANTED COVERAGE
# =============================================================================
print("\n" + "="*62)
print("  DATE PLANTED COVERAGE")
print("="*62)

if "date_planted" in gdf.columns:
    dates = pd.to_datetime(gdf["date_planted"], errors="coerce").dropna()
    print(f"  Trees with planting date: {len(dates):,} / {len(gdf):,} "
          f"({len(dates)/len(gdf)*100:.1f}%)")
    if len(dates) > 0:
        print(f"  Earliest: {dates.min().date()}")
        print(f"  Latest  : {dates.max().date()}")
        print(f"\n  Trees planted per decade:")
        decades = (dates.dt.year // 10 * 10).value_counts().sort_index()
        for decade, cnt in decades.items():
            print(f"    {decade}s  {cnt:>6,}")

# =============================================================================
# 8. SAVE SUMMARY CSV
# =============================================================================
print("\n" + "="*62)
print("  SAVING SUMMARY CSV")
print("="*62)

summary_rows = []

# Species counts
if "common_name" in gdf.columns:
    sp = gdf["common_name"].value_counts().reset_index()
    sp.columns = ["label", "count"]
    sp["category"] = "species"
    summary_rows.append(sp)

# Neighbourhood counts
if nb_col:
    nb = gdf[nb_col].value_counts().reset_index()
    nb.columns = ["label", "count"]
    nb["category"] = "neighbourhood"
    summary_rows.append(nb)

# Height bin counts
if ht_col:
    hb = cut.value_counts().sort_index().reset_index()
    hb.columns = ["label", "count"]
    hb["category"] = "height_bin"
    summary_rows.append(hb)

if summary_rows:
    summary_df = pd.concat(summary_rows, ignore_index=True)
    summary_df.to_csv(OUT_CSV, index=False)
    print(f"  [✔] Saved: {OUT_CSV}")

print(f"""
╔══════════════════════════════════════════════════════════╗
║   ✅ EXTRACTION COMPLETE                                 ║
╠══════════════════════════════════════════════════════════╣
║  Total trees         : {len(gdf):>7,}                        ║
║  Columns available   : {len(gdf.columns)-1:>7,}                        ║
║  Height data present : {"Yes" if ht_col else "No":>7}                        ║
║  Diameter present    : {"Yes" if dm_col else "No":>7}                        ║
║  Summary CSV         : tree_summary.csv                  ║
╚══════════════════════════════════════════════════════════╝
""")


