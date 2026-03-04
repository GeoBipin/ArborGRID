# =============================================================================
# ArborGrid 2.0 | Urban Forest Scientific Dashboard — Shared Module
# File: dashboard.py
#
# PURPOSE
# ───────
# This file is a pure importable module.  It contains:
#   • Visual constants  (colours, palette, plotly template)
#   • Utility functions (no_data_box, section, chart_layout, shannon_h)
#   • load_trees()      (cached data loader + feature engineering)
#
# It contains NO top-level Streamlit UI calls — no st.set_page_config,
# no st.sidebar, no st.write, no st.markdown outside of functions.
# This means importing it never renders anything by itself.
#
# USAGE IN app.py
# ───────────────
#   from dashboard import (
#       load_trees,
#       no_data_box, section,
#       PALETTE, NOT_RECORDED,
#       BG_CARD, BORDER, TEXT_MAIN, TEXT_SUB, PLOTLY_TEMPLATE,
#   )
#
# Data source : C:/ArborGRID/data/trees_raw.geojson
#
# Column names used EXACTLY as they appear in the GeoJSON after fetch_data.py:
#   common_name    — tree common name (e.g. NORWAY MAPLE)
#   genus_name     — genus (e.g. ACER)         ← read directly from data
#   species_name   — species (e.g. PLATANOIDES) ← read directly from data
#   cultivar_name  — cultivar (e.g. DEBORAH)    ← read directly from data
#   height_m / height_range  — actual height in metres
#   diameter_cm / diameter   — trunk diameter in centimetres
#   date_planted   — ISO date string
#   neighbourhood  — neighbourhood name (may not exist in all API versions)
#   address / std_street — street address
#
# RULE: No taxonomy is hardcoded.  Genus, Species and Cultivar unique values
#       are read from the actual data.  Empty strings are normalised to None
#       (shown as "Not Recorded" in charts so viewers are never misled).
# All null / missing values are explicitly surfaced — never silently hidden.
# All aggregation happens on the backend before any chart is rendered.
# =============================================================================

import warnings
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

warnings.filterwarnings("ignore")

# =============================================================================
# VISUAL CONSTANTS  (match ArborGrid 2.0 app.py theme exactly)
# =============================================================================
BG_DARK         = "#0e1117"
BG_CARD         = "#1a2030"
BG_CARD2        = "#1e2530"
BORDER          = "#2e3a4e"
TEXT_MAIN       = "#e8f0fe"
TEXT_SUB        = "#8b9ab0"
PLOTLY_TEMPLATE = "plotly_dark"
NOT_RECORDED    = "Not Recorded"

PALETTE = [
    "#4A9EFF", "#27ae60", "#f39c12", "#e67e22", "#e74c3c",
    "#9b59b6", "#1abc9c", "#3498db", "#e91e63", "#ff9800",
    "#00bcd4", "#8bc34a", "#ff5722", "#607d8b", "#795548",
    "#673ab7", "#009688", "#cddc39", "#ff4081", "#00e5ff",
]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def no_data_box(message: str):
    """
    Render a clearly visible dashed placeholder whenever data is absent.
    Never leaves a section silently blank — viewer always knows why.
    """
    st.markdown(
        f"<div class='no-data-box'>⚠️ {message}</div>",
        unsafe_allow_html=True,
    )


def section(title: str):
    """Render a styled section sub-header."""
    st.markdown(
        f"<p class='section-hdr'>{title}</p>",
        unsafe_allow_html=True,
    )


def chart_layout(fig, height=400, b=20, l=0, r=10):
    """Apply consistent dark-card layout to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_CARD,
        height=height,
        margin=dict(t=14, b=b, l=l, r=r),
    )
    return fig


def shannon_h(series: pd.Series) -> float:
    """Shannon Diversity Index  H' = −Σ pᵢ ln pᵢ"""
    counts = series.dropna().value_counts()
    if len(counts) < 2:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


# =============================================================================
# DATA LOADING  (cached — runs once per Streamlit session)
# =============================================================================

@st.cache_data(show_spinner="Loading and processing tree records …")
def load_trees(trees_path: str) -> pd.DataFrame:
    """
    Load trees_raw.geojson, drop geometry, normalise column names,
    engineer derived columns.  Returns a plain DataFrame.

    Column normalisation
    ────────────────────
    fetch_data.py may rename columns across API versions:
      height_m    → stays height_m   OR → height_range
      diameter_cm → stays diameter_cm OR → diameter

    Both variants are detected and unified to canonical names:
      height_m    (metres, numeric)
      diameter_cm (centimetres, numeric)

    Text columns are uppercased and stripped.  Empty strings and literal
    "NONE"/"NAN" strings are converted to Python None so .dropna() and
    .notna() work correctly everywhere.
    """
    import geopandas as gpd

    path = Path(trees_path)
    if not path.exists():
        return pd.DataFrame()

    gdf = gpd.read_file(path)
    df  = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))

    # ── Column aliases ────────────────────────────────────────────────────────
    if "height_range" in df.columns and "height_m" not in df.columns:
        df = df.rename(columns={"height_range": "height_m"})
    if "diameter" in df.columns and "diameter_cm" not in df.columns:
        df = df.rename(columns={"diameter": "diameter_cm"})
    if "std_street" in df.columns and "address" not in df.columns:
        df = df.rename(columns={"std_street": "address"})
    if "neighbourhood_name" in df.columns and "neighbourhood" not in df.columns:
        df = df.rename(columns={"neighbourhood_name": "neighbourhood"})

    # ── Numeric coercion ──────────────────────────────────────────────────────
    for col in ["height_m", "diameter_cm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Text normalisation ────────────────────────────────────────────────────
    # RULE: do NOT replace values with invented labels.
    # Empty / null → Python None.  Charts will show "Not Recorded".
    text_cols = ["common_name", "genus_name", "species_name",
                 "cultivar_name", "neighbourhood", "address"]
    for col in text_cols:
        if col in df.columns:
            df[col] = (df[col]
                       .astype(str).str.strip().str.upper()
                       .replace({"NAN": None, "NONE": None,
                                 "": None, "NA": None}))

    # ── Date planted ──────────────────────────────────────────────────────────
    if "date_planted" in df.columns:
        df["date_planted"] = pd.to_datetime(df["date_planted"], errors="coerce")
        df["plant_year"]   = df["date_planted"].dt.year
        df["plant_decade"] = (df["plant_year"] // 10 * 10).astype("Int64")

    # ── Display Species: top 20 by count, rest → grouped label ───────────────
    if "common_name" in df.columns:
        top20 = df["common_name"].value_counts().head(20).index
        other_count = int(
            (~df["common_name"].isin(top20) & df["common_name"].notna()).sum()
        )
        df["display_species"] = df["common_name"].where(
            df["common_name"].isin(top20),
            other=(f"Other Species ({other_count:,} trees, "
                   f"{df['common_name'].nunique() - 20} types)"),
        )

    # ── DBH Size Class ────────────────────────────────────────────────────────
    if "diameter_cm" in df.columns:
        df["dbh_class"] = pd.cut(
            df["diameter_cm"],
            bins=[0, 15, 30, 50, 9_999],
            labels=["Young  (0 – 15 cm)",
                    "Established  (15 – 30 cm)",
                    "Maturing  (30 – 50 cm)",
                    "Legacy  (50+ cm)"],
            right=False,
        ).astype(object).where(df["diameter_cm"].notna(), other=None)

    # ── Height Class ──────────────────────────────────────────────────────────
    if "height_m" in df.columns:
        df["height_class"] = pd.cut(
            df["height_m"],
            bins=[0, 5, 15, 9_999],
            labels=["Small  (under 5 metres)",
                    "Medium  (5 to 15 metres)",
                    "Large  (over 15 metres)"],
            right=False,
        ).astype(object).where(df["height_m"].notna(), other=None)

    # ── Basal Area per tree (m²) = π × (DBH_m / 2)² ─────────────────────────
    if "diameter_cm" in df.columns:
        df["basal_area_m2"] = np.pi * ((df["diameter_cm"] / 200.0) ** 2)

    # Terminal log (visible in Streamlit server console)
    print("\n[dashboard.py] Detected columns:", list(df.columns))
    print(f"  Total trees loaded: {len(df):,}\n")

    return df






