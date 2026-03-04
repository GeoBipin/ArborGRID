from pathlib import Path
import geopandas as gpd
import pandas as pd

DATA_DIR = Path("C:/ArborGRID/data")
TREES_F  = DATA_DIR / "trees_raw.geojson"

def main():
    if not TREES_F.exists():
        print(f"[✘] File not found: {TREES_F}")
        return

    gdf = gpd.read_file(TREES_F)
    print(f"[i] Loaded: {TREES_F}  | features: {len(gdf):,}")
    print(f"[i] CRS: {gdf.crs}\n")

    print("=== Columns & dtypes ===")
    print(gdf.dtypes.to_string())
    print("\n=== Top 10 columns ===")
    print(list(gdf.columns[:10]))

    print("\n=== Head (5 rows) ===")
    print(gdf.head(5).to_string(index=False))

    # Candidate height columns (in order of preference)
    candidates = ["height_range", "height_range_id", "height_m",
                  "heightrangeid", "diameter", "diameter_cm"]
    found = [c for c in candidates if c in gdf.columns]
    print("\n=== Height-related columns found ===")
    print(found if found else "None")

    if found:
        for c in found:
            s = gdf[c]
            nonnull = s.dropna()
            print(f"\n-- {c} --")
            print(f"  non-null: {len(nonnull):,} / {len(gdf):,}")
            try:
                print(f"  dtype: {s.dtype}")
                print(f"  sample unique values (up to 10): {nonnull.unique()[:10].tolist()}")
                if pd.api.types.is_numeric_dtype(s):
                    print(f"  min/max: {nonnull.min()} / {nonnull.max()}")
                else:
                    print(f"  value_counts (top 5):\n{nonnull.value_counts().head(5).to_string()}")
            except Exception as e:
                print(f"  (inspect error: {e})")

    # Species preview
    species_cols = [c for c in ["common_name", "genus_name", "species_name"] if c in gdf.columns]
    if species_cols:
        print("\n=== Species top 10 ===")
        col = species_cols[0]
        print(gdf[col].value_counts().head(10).to_string())

    # Basic geometry sanity
    print("\n=== Geometry types summary ===")
    print(gdf.geometry.geom_type.value_counts().to_string())

    print("\nDone.")

if __name__ == '__main__':
    main()