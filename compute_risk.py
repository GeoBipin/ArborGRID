# =============================================================================
# ArborGrid 2.0 | Phase 2: Vector Sentinel ETL
# File: compute_risk.py
# Run: & $PY compute_risk.py
#
# Unique key  : seg_id  (integer row index 0..N)
#               Replaces FID — guaranteed unique even when two segments share
#               identical length_m values.  length_m is kept as a display
#               attribute; it is never used as a join key.
#
# Height note : Vancouver public trees (API v2) store height as actual metres
#               in the 'height_m' field (e.g. 13.7 m).  fetch_data.py renames
#               this to 'height_range' in trees_raw.geojson.
#               HEIGHT_ALERT_THRESHOLD = 15 m  — segments whose tallest buffer
#               tree meets or exceeds this are flagged as height alerts.
#
# Output columns added to enriched_powerlines.geojson:
#   seg_id, tree_count, max_height, max_height_label,
#   height_alert, risk_level, dist_to_critical,
#   nearest_facility, nearest_fac_type
# =============================================================================

import sys
import time
import warnings
import geopandas as gpd
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR               = Path("C:/ArborGRID/data")
TARGET_CRS             = "EPSG:26910"
WGS84                  = "EPSG:4326"
BUFFER_M               = 15
OUT_FILE               = DATA_DIR / "enriched_powerlines.geojson"
EXPECTED_SEGMENTS      = 581

# Vancouver public trees schema (API v2) uses height_m — actual metres.
# fetch_data.py renames it to 'height_range' in trees_raw.geojson.
# Threshold: tallest tree in buffer >= 15 m triggers a height alert.
HEIGHT_ALERT_THRESHOLD = 15   # metres

# =============================================================================
# HELPERS
# =============================================================================

def banner(t): print(f"\n{'='*62}\n  {t}\n{'='*62}")
def ok(m):     print(f"  [✔] {m}")
def info(m):   print(f"  [i] {m}")
def warn(m):   print(f"  [!] {m}")


def load_layer(path: Path, name: str) -> gpd.GeoDataFrame:
    if not path.exists():
        print(f"  [✘] Not found: {path} — run fetch_data.py first.")
        sys.exit(1)
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(TARGET_CRS)
    elif gdf.crs.to_epsg() != 26910:
        gdf = gdf.to_crs(TARGET_CRS)
    before = len(gdf)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf = gdf.reset_index(drop=True)
    if before != len(gdf):
        warn(f"{name}: dropped {before - len(gdf)} null/empty geometries")
    ok(f"{name}: {len(gdf):,} features")
    return gdf


def assign_risk_level(count: int) -> str:
    if   count == 0:   return "Safe"
    elif count <=  5:  return "Low"
    elif count <= 15:  return "Moderate"
    elif count <= 30:  return "High"
    else:              return "Critical"


# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================
banner("STEP 1/5 — Load Data")

powerlines = load_layer(DATA_DIR / "powerlines.geojson", "powerlines")
facilities = load_layer(DATA_DIR / "facilities.geojson", "facilities")
trees      = load_layer(DATA_DIR / "trees_raw.geojson",  "trees")

info(f"Powerlines columns : {list(powerlines.columns)}")
info(f"Trees columns      : {list(trees.columns)}")

# =============================================================================
# STEP 2 — ASSIGN seg_id (unique integer key, float-collision-safe)
# =============================================================================
banner("STEP 2/5 — Assign Unique Segment ID (seg_id)")

powerlines = powerlines.reset_index(drop=True)
powerlines["seg_id"] = powerlines.index   # integer 0, 1, 2 … N

ok(f"seg_id assigned: 0 → {powerlines['seg_id'].max()}  ({len(powerlines)} segments)")

# Validate expected segment count
if len(powerlines) != EXPECTED_SEGMENTS:
    warn(
        f"Expected {EXPECTED_SEGMENTS} segments but found {len(powerlines)}. "
        f"Check source data — all {len(powerlines)} will still be processed."
    )
else:
    ok(f"Segment count confirmed: {EXPECTED_SEGMENTS} ✓")

# Inform user about length_m
if "length_m" in powerlines.columns:
    dupes = powerlines["length_m"].duplicated().sum()
    info(
        f"length_m range: {powerlines['length_m'].min():.2f} – "
        f"{powerlines['length_m'].max():.2f} m"
    )
    if dupes > 0:
        info(
            f"{dupes} segments share a length_m value with another segment — "
            f"seg_id avoids all collisions (length_m is display-only, NOT the key)."
        )

# =============================================================================
# STEP 3 — BUILD 15m BUFFERS (one per segment, independently)
# =============================================================================
banner("STEP 3/5 — Build 15m Buffers")

buffers = gpd.GeoDataFrame(
    {
        "seg_id":   powerlines["seg_id"].values,
        "geometry": powerlines.geometry.buffer(BUFFER_M),
    },
    crs=TARGET_CRS,
)

ok(f"{len(buffers):,} buffers  |  corridor area: {buffers.geometry.area.sum() / 10_000:.1f} ha")
info(
    f"Each buffer is independent.  A tree between two adjacent powerlines "
    f"is counted for BOTH — correct, each line is assessed on its own risk."
)

# =============================================================================
# STEP 4 — COUNT TREES + MAX HEIGHT per segment
# =============================================================================
banner("STEP 4/5 — Count Trees & Max Height per Segment")

# ── Detect height column ────────────────────────────────────────────────────
ht_col = None
# fetch_data.py renames 'height_m' → 'height_range' when saving trees_raw.geojson.
# We check for that renamed column first, then fall back to the original name.
for candidate in ["height_range", "height_m"]:
    if candidate in trees.columns:
        ht_col = candidate
        info(f"Height column found: '{ht_col}' (actual metres)")
        break

if ht_col is None:
    warn("No height column found in tree data → height alerts DISABLED")

# ── Ensure tree geometries are points ───────────────────────────────────────
tr = trees.copy()
if not tr.geometry.geom_type.eq("Point").all():
    info("Converting tree geometries to centroids ...")
    tr.geometry = tr.geometry.centroid

# ── Spatial join trees → buffers ────────────────────────────────────────────
t0 = time.time()
info(f"Joining {len(tr):,} trees → {len(buffers):,} buffers ...")

keep_tree_cols = ["geometry"] + ([ht_col] if ht_col else [])

joined = gpd.sjoin(
    tr[keep_tree_cols],
    buffers[["seg_id", "geometry"]],
    how="inner",
    predicate="intersects",   # tree point inside 15m buffer polygon
)

ok(f"Join: {len(joined):,} tree-buffer hits  |  {time.time() - t0:.1f}s")

# ── Aggregate per segment ────────────────────────────────────────────────────
if ht_col and ht_col in joined.columns:
    joined[ht_col] = pd.to_numeric(joined[ht_col], errors="coerce")
    # ── tree_count: count ALL rows (every tree in the buffer),
    #    completely independent of whether height data is present.
    #    Using .size() instead of .count() prevents NaN rows being silently
    #    dropped — which was the previous undercounting bug.
    count_agg  = joined.groupby("seg_id").size().reset_index(name="tree_count")
    # ── max_height: max ordinal among trees that DO have height data
    height_agg = (
        joined.groupby("seg_id")[ht_col]
        .max()
        .reset_index()
        .rename(columns={ht_col: "max_height"})
    )
    agg = count_agg.merge(height_agg, on="seg_id", how="left")
else:
    agg = joined.groupby("seg_id").size().reset_index(name="tree_count")
    agg["max_height"] = pd.NA   # no height data available

# ── Merge back to powerlines ─────────────────────────────────────────────────
result = powerlines.copy()
result = result.merge(agg, on="seg_id", how="left")
result["tree_count"] = result["tree_count"].fillna(0).astype(int)

# ── Height alert & label (only when height data is present) ──────────────────
if ht_col:
    result["max_height"] = result["max_height"].fillna(0)

    # height_alert: True only when the tallest tree in the buffer
    # has height_range_id >= HEIGHT_ALERT_THRESHOLD (i.e. > ~15 m)
    # AND there is at least one tree in the buffer
    result["height_alert"] = (
        (result["max_height"] >= HEIGHT_ALERT_THRESHOLD) &
        (result["tree_count"] > 0)
    )

    # Human-readable height label: show actual metres to 1 decimal place
    result["max_height_label"] = result.apply(
        lambda row: (
            f"{row['max_height']:.1f} m"
            + (" ⚠️ >15m" if row["max_height"] >= HEIGHT_ALERT_THRESHOLD else "")
            if row["tree_count"] > 0
            else "No trees in buffer"
        ),
        axis=1,
    )

    n_alert = int(result["height_alert"].sum())
    ok(f"Height alert segments (max tree > 15m): {n_alert:,}")
else:
    # No height data — populate neutral defaults so app.py is always safe
    result["height_alert"]    = False
    # Do NOT add max_height_label — app.py checks for its existence

# ── Risk level column ────────────────────────────────────────────────────────
result["risk_level"] = result["tree_count"].apply(assign_risk_level)

enc = int((result["tree_count"] > 0).sum())
ok(f"Segments with ≥1 tree   : {enc:,} / {len(result):,}")
ok(f"Max trees (one segment) : {result['tree_count'].max():,}")
ok(f"Mean trees per segment  : {result['tree_count'].mean():.1f}")

# =============================================================================
# STEP 5 — NEAREST FACILITY per segment
# =============================================================================
banner("STEP 5/5 — Nearest Facility per Segment")

fac = facilities.copy()
fac.geometry = fac.geometry.centroid      # reduce polygons to points
fac = fac.reset_index(drop=True)

# Centroid of each powerline segment (more representative than an endpoint)
pl_centroids = gpd.GeoDataFrame(
    {
        "seg_id":   powerlines["seg_id"].values,
        "geometry": powerlines.geometry.centroid.values,
    },
    crs=TARGET_CRS,
)

keep_fac = ["geometry"]
if "name"    in fac.columns: keep_fac.append("name")
if "amenity" in fac.columns: keep_fac.append("amenity")

t0 = time.time()
info(f"sjoin_nearest: {len(pl_centroids):,} centroids → {len(fac):,} facilities ...")

near = gpd.sjoin_nearest(
    pl_centroids,
    fac[keep_fac],
    how="left",
    distance_col="dist_to_critical",
)
# sjoin_nearest can produce duplicates for equidistant facilities — keep first
near = near[~near.index.duplicated(keep="first")].reset_index(drop=True)

ok(f"Done in {time.time() - t0:.1f}s")

# Attach to result (near is aligned to pl_centroids row order = result row order)
result["dist_to_critical"] = near["dist_to_critical"].round(1).values
if "name"    in near.columns: result["nearest_facility"] = near["name"].fillna("N/A").values
if "amenity" in near.columns: result["nearest_fac_type"] = near["amenity"].fillna("N/A").values

d = result["dist_to_critical"].dropna()
ok(f"Facility dist — min: {d.min():.0f}m  mean: {d.mean():.0f}m  max: {d.max():.0f}m")

# =============================================================================
# SAVE
# =============================================================================
out = result.to_crs(WGS84)

# Drop columns GeoJSON cannot serialise (list/dict cell values)
bad_cols = [
    c for c in out.columns
    if c != "geometry"
    and len(out[c].dropna()) > 0
    and isinstance(out[c].dropna().iloc[0], (list, dict))
]
if bad_cols:
    warn(f"Dropping non-serialisable columns: {bad_cols}")
    out = out.drop(columns=bad_cols)

out.to_file(OUT_FILE, driver="GeoJSON")
mb = OUT_FILE.stat().st_size / 1_048_576
ok(f"Saved: {OUT_FILE.name}  ({len(out):,} features  |  {mb:.2f} MB)")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
rl = result["risk_level"].value_counts()

print(f"""
╔══════════════════════════════════════════════════════════╗
║           ✅  PHASE 2 COMPLETE                          ║
╠══════════════════════════════════════════════════════════╣
║  Unique key          : seg_id (row index, always unique) ║
║  Powerline segments  : {len(result):>6,}                         ║
║  Trees in corridors  : {int(result['tree_count'].sum()):>6,}                         ║
║  Segments with trees : {enc:>6,}                         ║
║  Height alert (>15m) : {int(result['height_alert'].sum()):>6,}                         ║
║  Min facility dist   : {d.min():>6.0f} m                        ║
╠══════════════════════════════════════════════════════════╣
║  Risk breakdown:                                         ║
║    Safe     : {int(rl.get('Safe',    0)):>6,}                         ║
║    Low      : {int(rl.get('Low',     0)):>6,}                         ║
║    Moderate : {int(rl.get('Moderate',0)):>6,}                         ║
║    High     : {int(rl.get('High',    0)):>6,}                         ║
║    Critical : {int(rl.get('Critical',0)):>6,}                         ║
╠══════════════════════════════════════════════════════════╣
║  Output → data/enriched_powerlines.geojson               ║
║  Next   → & $PY -m streamlit run app.py                  ║
╚══════════════════════════════════════════════════════════╝
""")