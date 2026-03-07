# =============================================================================
# CanopyGRID | Phase 2 Dashboard
# File: app.py
# Run: & $PY -m streamlit run app.py
#
# Hover tooltip (per powerline segment) shows:
#   Segment Number | Risk Level | Trees in 15 Metre Buffer | Line Length |
#   Voltage | Nearest Facility Name | Distance to Nearest Facility |
#   Maximum Tree Height in Buffer
#  critical height
# Tooltip background is LIGHT RED when the tallest tree in the 15m buffer
# is >= 15 metres.  Achieved by splitting powerlines into two GeoJson layers —
# one for normal segments, one for height-alert segments — each with its
# own tooltip style string.  This is the only reliable Folium-native approach.
# =============================================================================

import streamlit as st
import folium
from folium import TileLayer
from folium.plugins import Fullscreen
from streamlit_folium import st_folium
import geopandas as gpd
import pandas as pd
from pathlib import Path


import numpy as np
import plotly.express       as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dashboard import (
    load_trees, no_data_box, section,
    PALETTE, NOT_RECORDED,
    BG_CARD, BORDER, TEXT_MAIN, TEXT_SUB, PLOTLY_TEMPLATE,
)
# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR     = Path("data")
POWERLINES_F = DATA_DIR / "powerlines.geojson"
ENRICHED_F   = DATA_DIR / "enriched_powerlines.geojson"
FACILITIES_F = DATA_DIR / "facilities.geojson"
TREES_F      = DATA_DIR / "trees_raw.geojson"
TREES_URL = "https://drive.google.com/uc?export=download&id=1QDukXwI3Xgo-HVk-MCv22sXo2LnbtsDS"

VANCOUVER = [49.2827, -123.1207]
# Map panning locked to Greater Vancouver / SW BC — prevents global tile loading
BC_BOUNDS = [[48.8, -124.5], [49.8, -121.5]]   # [SW, NE]

TOOLTIP_STYLE_NORMAL = (
    "background: #ffffff; border: 1px solid #cccccc; "
    "border-radius: 6px; padding: 8px 12px; "
    "font-family: sans-serif; font-size: 0.83rem;"
)
TOOLTIP_STYLE_ALERT = (
    "background: #ffe8e8; border: 2px solid #e74c3c; "
    "border-radius: 6px; padding: 8px 12px; "
    "font-family: sans-serif; font-size: 0.83rem; "
    "box-shadow: 0 0 10px rgba(231, 76, 60, 0.45);"
)

# =============================================================================
# BASEMAPS
# =============================================================================
BASEMAPS = {
    "☀️ Light (CartoDB Positron)": {
        "tiles": "CartoDB positron", "attr": None},
    "🗺️ Street (OpenStreetMap)": {
        "tiles": "OpenStreetMap",    "attr": None},
    "🌙 Dark (CartoDB Dark)": {
        "tiles": "CartoDB dark_matter", "attr": None},
    "🛰️ Satellite (ESRI)": {
        "tiles": (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        "attr": (
            "Tiles © Esri — Source: Esri, USDA, USGS, AEX, GeoEye, "
            "Getmapping, Aerogrid, IGN, IGP, UPR-EGP"
        ),
    },
    "🏔️ Topo (OpenTopoMap)": {
        "tiles": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        "attr":  (
            "Map data: © OpenStreetMap contributors, SRTM | "
            "Map style: © OpenTopoMap (CC-BY-SA)"
        ),
    },
}
DEFAULT_BASEMAP = "☀️ Light (CartoDB Positron)"

# =============================================================================
# PAGE CONFIG & CSS
# =============================================================================
st.set_page_config(
    page_title="CanopyGRID",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Soother, more professional slate theme instead of crushed black */
    .stApp { background-color: #1e242c; } 
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #2a313c, #323b48);
        border: 1px solid #4a5568; border-radius: 12px;
        padding: 16px 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }
    [data-testid="metric-container"] label {
        color: #b0bbd0 !important; font-size: 0.95rem !important;
        font-weight: 600 !important; letter-spacing: 0.05em !important;
        text-transform: uppercase;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important; font-size: 2.2rem !important;
        font-weight: 700 !important; text-shadow: 0px 1px 2px rgba(0,0,0,0.2);
    }
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #2a313c;
    }
    hr { border-color: #3a4250 !important; }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #232a35; border-radius: 8px;
        padding: 6px; gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px; color: #a0abc0; font-weight: 600; font-size: 1.05rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3a4250 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPERS
# =============================================================================

@st.cache_data(show_spinner=False)
def load_gdf(path: str):
    try:
        gdf = gpd.read_file(path)
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")
        return gdf
    except Exception:
        return None


def gdf_ready(gdf) -> bool:
    return gdf is not None and len(gdf) > 0


def tree_count_to_color(count: float) -> str:
    """Line colour by tree_count (risk level)."""
    if   count == 0:   return "#4A9EFF"   # Safe     — blue
    elif count <=  5:  return "#27ae60"   # Low      — green
    elif count <= 15:  return "#f39c12"   # Moderate — amber
    elif count <= 30:  return "#e67e22"   # High     — orange
    else:              return "#e74c3c"   # Critical — red


def height_to_color(height) -> str:
    """
    Gradient fill colour for individual tree points based on actual height in
    metres (height_m / renamed to height_range by fetch_data.py).
    Returns grey for null / missing height data.
    """
    try:
        h = float(height)
    except (TypeError, ValueError):
        return "#9e9e9e"   # grey — no height data
    if   h <  5:  return "#c8e6c9"   # very light green — short tree / shrub
    elif h < 10:  return "#66bb6a"   # light green      — small tree
    elif h < 15:  return "#2e7d32"   # dark green       — medium tree
    elif h < 20:  return "#f9a825"   # amber            — tall (approaching alert)
    elif h < 25:  return "#e65100"   # deep orange      — very tall
    else:         return "#b71c1c"   # dark red         — critical height (25m+)


def build_tooltip_fields(gdf: gpd.GeoDataFrame):
    """
    Return (fields, aliases) for GeoJsonTooltip based on which columns
    actually exist in gdf.  Order matches what the user requested.
    """
    candidates = [
        ("seg_id",           "Segment Number"),
        ("risk_level",       "Risk Level"),
        ("tree_count",       "Trees in 15 Metre Buffer"),
        ("length_m",         "Line Length (Metres)"),
        ("voltage",          "Voltage"),
        ("nearest_facility", "Nearest Facility Name"),
        ("dist_to_critical", "Distance to Nearest Facility (Metres)"),
        ("max_height_label", "Maximum Tree Height in Buffer"),
    ]
    fields  = [col   for col, _   in candidates if col in gdf.columns]
    aliases = [alias for col, alias in candidates if col in gdf.columns]
    return fields, aliases


def line_style(feature):
    count = float(feature["properties"].get("tree_count", 0) or 0)
    return {"color": tree_count_to_color(count), "weight": 3, "opacity": 0.9}


# =============================================================================
# LOAD DATA
# =============================================================================
use_enriched = ENRICHED_F.exists()
lines_path   = str(ENRICHED_F) if use_enriched else str(POWERLINES_F)

gdf_lines = load_gdf(lines_path)          if (ENRICHED_F.exists() or POWERLINES_F.exists()) else None
gdf_fac   = load_gdf(str(FACILITIES_F))   if FACILITIES_F.exists() else None
gdf_trees = load_gdf(str(TREES_F))        if TREES_F.exists()      else None

# ── Post-load type fixes ───────────────────────────────────────────────────
# GeoJSON stores booleans as 0/1 integers on re-read.
# ~int64 is bitwise NOT (always True for 0, large negative for 1) — wrong.
# Must cast explicitly to bool so the normal/alert split works correctly.
if gdf_lines is not None and "height_alert" in gdf_lines.columns:
    gdf_lines["height_alert"] = gdf_lines["height_alert"].astype(bool)

# date_planted is a pandas Timestamp — not JSON-serialisable.
# Convert to ISO date string so gdf_trees.to_json() never crashes.
if gdf_trees is not None and "date_planted" in gdf_trees.columns:
    gdf_trees["date_planted"] = gdf_trees["date_planted"].astype(str).replace("NaT", "null")

# Determine whether height alert data is present in this run's output
has_height_alerts = (
    gdf_lines is not None
    and "height_alert" in gdf_lines.columns
    and "max_height_label" in gdf_lines.columns
    and gdf_lines["height_alert"].any()
)

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:8px 0 16px 0;'>
        <div style='font-size:2.4rem;'>🌳</div>
        <div style='font-size:1.6rem; font-weight:700; color:#ffffff;'>
            CanopyGRID
        </div>
        <div style='font-size:0.95rem; color:#b0bbd0; margin-top:4px;'>
            Urban Vegetation & Utility Risk Assessment
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Phase tracker
    st.markdown("#### 📋 Phase Status")
    for num, label, color, active in [
        ("1", "Data Ingestion",  "#2ecc71", False),
        ("2", "ETL / Corridors", "#2ecc71" if use_enriched else "#f39c12", True),
        ("3", "Risk Engine",     "#4a5568", False),
        ("4", "Weather Trigger", "#4a5568", False),
        ("5", "Command Centre",  "#4a5568", False),
    ]:
        icon = "🟢" if color == "#2ecc71" else ("🟡" if color == "#f39c12" else "⚪")
        st.markdown(f"""
        <div style='background:{"#1e2d24" if active else "#1a212a"};
                    border:1px solid {color if active else "#3a4250"};
                    border-radius:8px; padding:10px 14px; margin-bottom:6px;
                    display:flex; align-items:center; gap:10px;'>
            <span style='font-size:0.8rem;font-weight:700;
                         color:{color};min-width:18px;'>P{num}</span>
            <span style='font-size:0.95rem;color:#d0d7e2;font-weight:500;'>{label}</span>
            <span style='margin-left:auto;font-size:0.9rem;'>{icon}</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    
    # Basemap
    st.markdown("#### 🗺️ Basemap")
    selected_basemap = st.radio(
        "basemap",
        options=list(BASEMAPS.keys()),
        index=list(BASEMAPS.keys()).index(DEFAULT_BASEMAP),
        label_visibility="collapsed",
    )

    st.divider()

    # Height alert legend (only shown when height data is present)
    if has_height_alerts:
        st.markdown("#### ⚠️ Height Alert")
        st.markdown("""
        <div style='background:#2a1515;border:1px solid #e74c3c44;
                    border-radius:8px;padding:12px 14px;
                    font-size:0.9rem;color:#ffcccc;line-height:1.6;'>
            Segments where the tallest tree in the 15 metre buffer is
            <b>≥ 15 metres tall</b>.<br><br>
            
        </div>
        """, unsafe_allow_html=True)
        n_alert = int(gdf_lines["height_alert"].sum())
        st.metric("⚠️ Alert segments", n_alert)
        st.divider()

    

# =============================================================================
# HEADER
# =============================================================================
phase_badge = (
    '<span style="background:#1a3a1a;border:1px solid #2ecc71;'
    'border-radius:6px;padding:4px 12px;font-size:0.9rem;'
    'color:#2ecc71;font-weight:600;margin-left:10px;">Phase 2 ✅</span>'
    if use_enriched else
    '<span style="background:#3a2a0a;border:1px solid #f39c12;'
    'border-radius:6px;padding:4px 12px;font-size:0.9rem;'
    'color:#f39c12;font-weight:600;margin-left:10px;">Run compute_risk.py</span>'
)
st.markdown(f"""
<div style='padding:8px 0 20px 0;'>
    <h1 style='margin:0;font-size:2.4rem;font-weight:700;color:#ffffff;'>
        🌳 CanopyGRID {phase_badge}
    </h1>
    <p style='margin:8px 0 0 0;color:#b0bbd0;font-size:1.1rem;'>
        Vancouver, BC &nbsp;·&nbsp; 15 Metre Tree Corridors &nbsp;·&nbsp;
        Basemap: <b style='color:#ffffff;'>{selected_basemap}</b>
    </p>
</div>
""", unsafe_allow_html=True)

if not use_enriched:
    st.warning(
        "⚠️ **Phase 2 ETL not run yet.** "
        "Run `compute_risk.py` first, then refresh.",
        icon="🚨",
    )

# =============================================================================
# KPI METRICS
# =============================================================================
c1, c2, c3, c4, c5 = st.columns(5)

total_km = trees_in_corridor = segs_with_trees = min_fac_dist = alert_segs = "–"
if gdf_ready(gdf_lines):
    if "length_m"        in gdf_lines.columns:
        total_km           = f"{gdf_lines['length_m'].sum() / 1000:.1f} km"
    if "tree_count"      in gdf_lines.columns:
        trees_in_corridor  = f"{int(gdf_lines['tree_count'].sum()):,}"
        segs_with_trees    = f"{int((gdf_lines['tree_count'] > 0).sum()):,}"
    if "dist_to_critical" in gdf_lines.columns:
        min_fac_dist       = f"{gdf_lines['dist_to_critical'].min():.0f} m"
    if "height_alert"    in gdf_lines.columns:
        alert_segs         = f"{int(gdf_lines['height_alert'].sum()):,}"

c1.metric("⚡ Network Length",     total_km)
c2.metric("🌲 Trees in Corridors", trees_in_corridor)
c3.metric("⚠️ Segments with Trees", segs_with_trees)
c4.metric("🏥 Minimum Facility Distance", min_fac_dist)
c5.metric("🌳 Height Alert Segments",    alert_segs)

st.divider()

# =============================================================================
# MAP
# =============================================================================
if not (POWERLINES_F.exists() or ENRICHED_F.exists()):
    st.info("⏳ Run `fetch_data.py` then `compute_risk.py` to load data.", icon="⚠️")
else:
    st.markdown("#### 🗺️ Risk Map")

    lc1, lc2, lc3 = st.columns(3)
    show_lines = lc1.toggle("⚡ Powerlines",   value=True)
    show_fac   = lc2.toggle("🏥 Hospitals & 🎓 Schools",   value=True)
    show_trees = lc3.toggle("🌲 All Trees",    value=False)

    # ── Base map ──────────────────────────────────────────────────────────────
    bm = BASEMAPS[selected_basemap]
    m  = folium.Map(
        location=VANCOUVER,
        zoom_start=12,
        min_zoom=9,
        max_zoom=19,
        tiles=None,
        prefer_canvas=True,
    )
    TileLayer(
        tiles=bm["tiles"],
        attr=bm["attr"] if bm["attr"] else "© OpenStreetMap contributors",
        name=selected_basemap,
        overlay=False,
        control=True,
        max_zoom=19,
        min_zoom=9,
    ).add_to(m)

    # Lock panning to Vancouver/BC — prevents global tile loading
    m.options["maxBounds"]          = BC_BOUNDS
    m.options["maxBoundsViscosity"] = 1.0

    # ── Powerlines ────────────────────────────────────────────────────────────
    # Split into two layers when height alert data is present:
    #   Layer A — normal segments  → white tooltip
    #   Layer B — alert segments   → light-red tooltip with red border glow
    # This is the only reliable Folium approach for per-feature tooltip styles.
    if show_lines and gdf_ready(gdf_lines):

        tt_fields, tt_aliases = build_tooltip_fields(gdf_lines)

        if has_height_alerts:
            normal_gdf = gdf_lines[~gdf_lines["height_alert"]].copy()
            alert_gdf  = gdf_lines[ gdf_lines["height_alert"]].copy()
        else:
            normal_gdf = gdf_lines.copy()
            alert_gdf  = None

        # Layer A — Normal powerlines
        if len(normal_gdf) > 0:
            folium.GeoJson(
                normal_gdf.to_json(),
                style_function=line_style,
                tooltip=folium.GeoJsonTooltip(
                    fields=tt_fields,
                    aliases=tt_aliases,
                    localize=True,
                    sticky=False,
                    style=TOOLTIP_STYLE_NORMAL,
                ),
                name="⚡ Powerlines",
                show=True,
            ).add_to(m)

        # Layer B — Height-alert powerlines (red glow tooltip)
        if alert_gdf is not None and len(alert_gdf) > 0:
            folium.GeoJson(
                alert_gdf.to_json(),
                style_function=line_style,
                tooltip=folium.GeoJsonTooltip(
                    fields=tt_fields,
                    aliases=tt_aliases,
                    localize=True,
                    sticky=False,
                    style=TOOLTIP_STYLE_ALERT,
                ),
                name="⚠️ Powerlines — Height Alert (Tallest Tree Over 15 Metres)",
                show=True,
            ).add_to(m)

    # ── Facilities — hospitals and schools as distinct markers ───────────────
    # Hospitals: red circle with white H (larger)
    # Schools: blue circle with graduation cap (smaller, different shape)
    # Split into two GeoJson layers so icons differ per type.
    if show_fac and gdf_ready(gdf_fac):
        fac_pts = gdf_fac.copy()
        fac_pts.geometry = fac_pts.geometry.centroid
        tt_f = [c for c in ["name", "amenity"] if c in fac_pts.columns]
        tt_a = ["Name", "Facility Type"][:len(tt_f)]

        if "amenity" in fac_pts.columns:
            hospitals = fac_pts[fac_pts["amenity"] == "hospital"].copy()
            schools   = fac_pts[fac_pts["amenity"] == "school"].copy()
        else:
            hospitals = fac_pts.copy()
            schools   = gpd.GeoDataFrame(columns=fac_pts.columns, crs=fac_pts.crs)

        if len(hospitals) > 0:
            folium.GeoJson(
                hospitals.to_json(),
                marker=folium.Marker(
                    icon=folium.Icon(
                        color="red", icon="h-square",
                        prefix="fa", icon_color="white",
                    )
                ),
                tooltip=folium.GeoJsonTooltip(
                    fields=tt_f, aliases=tt_a,
                    sticky=False, style=TOOLTIP_STYLE_NORMAL,
                ),
                name="🏥 Hospitals",
                show=True,
            ).add_to(m)

        if len(schools) > 0:
            folium.GeoJson(
                schools.to_json(),
                marker=folium.Marker(
                    icon=folium.DivIcon(
                        html=(
                            "<div style='"
                            "background:#1565c0;color:#fff;"
                            "border-radius:50%;width:22px;height:22px;"
                            "display:flex;align-items:center;"
                            "justify-content:center;"
                            "font-size:12px;font-weight:bold;"
                            "border:2px solid #fff;"
                            "box-shadow:0 1px 4px rgba(0,0,0,0.4);'>"
                            "🎓</div>"
                        ),
                        icon_size=(22, 22),
                        icon_anchor=(11, 11),
                    )
                ),
                tooltip=folium.GeoJsonTooltip(
                    fields=tt_f, aliases=tt_a,
                    sticky=False, style=TOOLTIP_STYLE_NORMAL,
                ),
                name="🎓 Schools",
                show=True,
            ).add_to(m)

    # ── All Trees — full dataset, gradient colour by height_m ─────────────────
    # Uses a single folium.GeoJson() call (NOT a Python loop) so Leaflet
    # renders all ~150k points natively.  style_function reads the 'height_range'
    # property (= height_m in metres, renamed by fetch_data.py) per feature.
    # Null heights are shown as grey.  All available attributes shown in tooltip.
    if show_trees and gdf_ready(gdf_trees):

        # Build tooltip field list — check both column name variants that
        # fetch_data.py may have used (height_m OR height_range, etc.)
        # Each candidate is (possible_col_name, full_label).
        # If the column exists in gdf_trees it is included; otherwise skipped.
        tree_tt_candidates = [
            ("common_name",   "Common Name"),
            ("genus_name",    "Genus Name"),
            ("species_name",  "Species Name"),
            ("cultivar_name", "Cultivar Name"),
            ("height_m",      "Height (Metres)"),
            ("height_range",  "Height (Metres)"),   # fetch_data.py alias
            ("diameter_cm",   "Diameter (Centimetres)"),
            ("diameter",      "Diameter (Centimetres)"),  # fetch_data.py alias
            ("date_planted",  "Date Planted"),
            ("neighbourhood", "Neighbourhood"),
            ("std_street",    "Street Address"),
            ("address",       "Street Address"),
        ]
        # De-duplicate: once a label has been added, skip further aliases for it
        seen_labels     = set()
        tree_tt_fields  = []
        tree_tt_aliases = []
        for col, label in tree_tt_candidates:
            if col in gdf_trees.columns and label not in seen_labels:
                tree_tt_fields.append(col)
                tree_tt_aliases.append(label)
                seen_labels.add(label)

        # Height column for gradient colouring — prefer height_m, fall back to height_range
        ht_display_col = next(
            (c for c in ["height_m", "height_range"] if c in gdf_trees.columns),
            None
        )

        def tree_style(feature):
            h = feature["properties"].get(ht_display_col) if ht_display_col else None
            fill = height_to_color(h)
            return {
                "fillColor":   fill,
                "color":       fill,
                "fillOpacity": 0.85,
                "weight":      0.3,
            }

        folium.GeoJson(
            gdf_trees.to_json(),
            marker=folium.CircleMarker(radius=3),
            style_function=tree_style,
            tooltip=folium.GeoJsonTooltip(
                fields=tree_tt_fields,
                aliases=tree_tt_aliases,
                localize=True,
                sticky=False,
                style=TOOLTIP_STYLE_NORMAL,
            ),
            name="🌲 All Trees (height gradient)",
            show=True,
        ).add_to(m)

    # ── Legend ────────────────────────────────────────────────────────────────
    m.get_root().html.add_child(folium.Element("""
    <div style='position:fixed;bottom:30px;left:50px;z-index:1000;
                background:rgba(255,255,255,0.96);border:1px solid #ccc;
                border-radius:10px;padding:14px 18px;font-family:sans-serif;
                font-size:0.78rem;color:#333;
                box-shadow:0 4px 12px rgba(0,0,0,0.15);min-width:230px;'>

        <div style='font-weight:700;margin-bottom:8px;font-size:0.84rem;
                    border-bottom:1px solid #eee;padding-bottom:6px;'>
            ⚡ Powerline Risk (Trees per 15 Metre Buffer)
        </div>
        <div style='display:flex;align-items:center;gap:8px;margin-bottom:4px;'>
            <div style='width:26px;height:4px;background:#4A9EFF;border-radius:2px;'></div>
            <span>0 trees — <b style='color:#4A9EFF'>Safe</b></span>
        </div>
        <div style='display:flex;align-items:center;gap:8px;margin-bottom:4px;'>
            <div style='width:26px;height:4px;background:#27ae60;border-radius:2px;'></div>
            <span>1–5 — <b style='color:#27ae60'>Low</b></span>
        </div>
        <div style='display:flex;align-items:center;gap:8px;margin-bottom:4px;'>
            <div style='width:26px;height:4px;background:#f39c12;border-radius:2px;'></div>
            <span>6–15 — <b style='color:#f39c12'>Moderate</b></span>
        </div>
        <div style='display:flex;align-items:center;gap:8px;margin-bottom:4px;'>
            <div style='width:26px;height:4px;background:#e67e22;border-radius:2px;'></div>
            <span>16–30 — <b style='color:#e67e22'>High</b></span>
        </div>
        <div style='display:flex;align-items:center;gap:8px;margin-bottom:10px;'>
            <div style='width:26px;height:4px;background:#e74c3c;border-radius:2px;'></div>
            <span>31+ — <b style='color:#e74c3c'>Critical</b></span>
        </div>

        <div style='font-weight:700;margin-bottom:8px;font-size:0.84rem;
                    border-top:1px solid #eee;padding-top:8px;'>
            
    </div>
    """))

    # LayerControl must be added AFTER all layers
    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    Fullscreen(position="topleft").add_to(m)

    st_folium(m, width=None, height=660, returned_objects=[])
    st.caption(
        f"🗺️ Basemap: **{selected_basemap}** · "
        "Map locked to Greater Vancouver for tile performance. "
        "Red-glowing tooltip = tallest tree in the 15 metre buffer is over 15 metres tall."
    )

    # =========================================================================
    # ANALYSIS TABS
    # =========================================================================
    st.divider()
    st.markdown("### 📊 Phase 2 Analysis")

    t1, t2 = st.tabs(["🚨 Risk Summary", "📋 Segment Data"])

    # ── Tab 1: Risk Summary ───────────────────────────────────────────────────
    with t1:
        if gdf_ready(gdf_lines) and "risk_level" in gdf_lines.columns:
            st.markdown("#### Encroachment Risk Distribution")
            rc1, rc2, rc3, rc4, rc5 = st.columns(5)
            risk_counts = gdf_lines["risk_level"].value_counts()
            for col, lab, clr in zip(
                [rc1, rc2, rc3, rc4, rc5],
                ["Safe", "Low", "Moderate", "High", "Critical"],
                ["#4A9EFF", "#27ae60", "#f39c12", "#e67e22", "#e74c3c"],
            ):
                n = int(risk_counts.get(lab, 0))
                col.markdown(
                    f"<div style='background:#2a313c;"
                    f"border:1px solid {clr}44;border-radius:10px;"
                    f"padding:16px;text-align:center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>"
                    f"<div style='color:{clr};font-size:2.2rem;"
                    f"font-weight:700;text-shadow: 0px 1px 2px rgba(0,0,0,0.3);'>{n:,}</div>"
                    f"<div style='color:#b0bbd0;font-size:0.95rem;"
                    f"margin-top:4px;text-transform:uppercase;"
                    f"font-weight:600;letter-spacing:0.06em;'>{lab}</div></div>",
                    unsafe_allow_html=True,
                )

            
        else:
            st.info("Run `compute_risk.py` to generate risk data.")

    
#
#URBAN FORESTRY
_d_df_all = load_trees(str(TREES_F))

if not _d_df_all.empty:

    # ── Column-presence flags ─────────────────────────────────────────────────
    _d_HAS_GENUS   = "genus_name"    in _d_df_all.columns
    _d_HAS_SPECIES = "species_name"  in _d_df_all.columns
    _d_HAS_CULTI   = "cultivar_name" in _d_df_all.columns
    _d_HAS_COMMON  = "common_name"   in _d_df_all.columns
    _d_HAS_HEIGHT  = "height_m"      in _d_df_all.columns
    _d_HAS_DBH     = "diameter_cm"   in _d_df_all.columns
    _d_HAS_NEIGH   = "neighbourhood" in _d_df_all.columns
    _d_HAS_DATE    = "plant_year"    in _d_df_all.columns

    # Terminal log visible in the Streamlit server console
    print("\n[app.py — Urban Forest Dashboard] Column flags:")
    print(f"  Genus    : {_d_HAS_GENUS}")
    print(f"  Species  : {_d_HAS_SPECIES}")
    print(f"  Cultivar : {_d_HAS_CULTI}")
    print(f"  Height   : {_d_HAS_HEIGHT}")
    print(f"  Diameter : {_d_HAS_DBH}")
    print(f"  Neighbourhood: {_d_HAS_NEIGH}")
    print(f"  Date planted : {_d_HAS_DATE}")
    print(f"  Total trees  : {len(_d_df_all):,}\n")

    # =========================================================================
    # SIDEBAR FILTER WIDGETS
    # Appended to the existing app.py sidebar — Streamlit accumulates all
    # `with st.sidebar:` blocks in the order they appear in the script.
    # =========================================================================
    with st.sidebar:
        st.divider()
        st.markdown("#### 🌳 Urban Forest Filters")
        st.caption(
            "Filters are hierarchical: selecting a Genus narrows the "
            "Species list, which narrows the Cultivar list."
        )

        # 1. Genus
        if _d_HAS_GENUS:
            _d_genus_vals = sorted(
                _d_df_all["genus_name"].dropna().unique().tolist())
            _d_sel_genus  = st.selectbox(
                "Genus Name (scientific)",
                ["All Genera"] + _d_genus_vals,
                key="d_genus",
            )
        else:
            _d_sel_genus = "All Genera"

        # 2. Species (narrowed by genus)
        _d_df_g = _d_df_all if _d_sel_genus == "All Genera" \
            else _d_df_all[_d_df_all["genus_name"] == _d_sel_genus]

        if _d_HAS_SPECIES:
            _d_sp_vals = sorted(
                _d_df_g["species_name"].dropna().unique().tolist())
            _d_sel_sp  = st.selectbox(
                "Species Name (scientific)",
                ["All Species"] + _d_sp_vals,
                key="d_species",
            )
        else:
            _d_sel_sp = "All Species"

        # 3. Cultivar (narrowed by species)
        _d_df_gs = _d_df_g if _d_sel_sp == "All Species" \
            else _d_df_g[_d_df_g["species_name"] == _d_sel_sp]

        if _d_HAS_CULTI:
            _d_cv_vals = sorted(
                _d_df_gs["cultivar_name"].dropna().unique().tolist())
            _d_sel_cv  = st.selectbox(
                "Cultivar Name",
                ["All Cultivars"] + _d_cv_vals,
                key="d_cultivar",
            )
        else:
            _d_sel_cv = "All Cultivars"

        st.divider()

        # 4. Neighbourhood
        if _d_HAS_NEIGH:
            _d_nb_vals = sorted(
                _d_df_all["neighbourhood"].dropna().unique().tolist())
            _d_sel_nb  = st.selectbox(
                "Neighbourhood",
                ["All Neighbourhoods"] + _d_nb_vals,
                key="d_neighbourhood",
            )
        else:
            _d_sel_nb = "All Neighbourhoods"
            st.info(
                "ℹ️ Neighbourhood data is not present in this dataset — "
                "spatial filtering by area is unavailable.", icon="ℹ️"
            )

        st.divider()

        # 5. Height slider
        if _d_HAS_HEIGHT:
            _d_h_min = float(_d_df_all["height_m"].min(skipna=True) or 0)
            _d_h_max = float(_d_df_all["height_m"].max(skipna=True) or 50)
            _d_sel_h = st.slider(
                "Tree Height Range (metres)",
                min_value=_d_h_min, max_value=_d_h_max,
                value=(_d_h_min, _d_h_max), step=0.5,
                key="d_height",
            )
        else:
            _d_sel_h = None
            st.info("ℹ️ Height data is not present in this dataset.",
                    icon="ℹ️")

        # 6. Diameter slider
        if _d_HAS_DBH:
            _d_d_min = float(_d_df_all["diameter_cm"].min(skipna=True) or 0)
            _d_d_max = float(_d_df_all["diameter_cm"].max(skipna=True) or 300)
            _d_sel_d = st.slider(
                "Trunk Diameter Range (centimetres)",
                min_value=_d_d_min, max_value=_d_d_max,
                value=(_d_d_min, _d_d_max), step=1.0,
                key="d_diameter",
            )
        else:
            _d_sel_d = None
            st.info("ℹ️ Diameter data is not present in this dataset.",
                    icon="ℹ️")

    # =========================================================================
    # APPLY FILTERS
    # =========================================================================
    _d_df = _d_df_all.copy()

    if _d_sel_genus != "All Genera"         and _d_HAS_GENUS:
        _d_df = _d_df[_d_df["genus_name"]    == _d_sel_genus]
    if _d_sel_sp    != "All Species"         and _d_HAS_SPECIES:
        _d_df = _d_df[_d_df["species_name"]  == _d_sel_sp]
    if _d_sel_cv    != "All Cultivars"       and _d_HAS_CULTI:
        _d_df = _d_df[_d_df["cultivar_name"] == _d_sel_cv]
    if _d_sel_nb    != "All Neighbourhoods"  and _d_HAS_NEIGH:
        _d_df = _d_df[_d_df["neighbourhood"] == _d_sel_nb]
    if _d_sel_h and _d_HAS_HEIGHT:
        _d_df = _d_df[
            _d_df["height_m"].between(
                _d_sel_h[0], _d_sel_h[1], inclusive="both")
            | _d_df["height_m"].isna()
        ]
    if _d_sel_d and _d_HAS_DBH:
        _d_df = _d_df[
            _d_df["diameter_cm"].between(
                _d_sel_d[0], _d_sel_d[1], inclusive="both")
            | _d_df["diameter_cm"].isna()
        ]

    _d_is_filtered = len(_d_df) < len(_d_df_all)
    _d_filter_label = (
        f'<span style="color:#f39c12;font-size:0.9rem;">'
        f'Showing {len(_d_df):,} of {len(_d_df_all):,} trees</span>'
        if _d_is_filtered else ""
    )

    # =========================================================================
    # DASHBOARD HEADER  (section break from the Risk Map above)
    # =========================================================================
    st.divider()
    st.markdown(f"""
    <div style='padding:8px 0 18px 0;'>
        <h2 style='margin:0;font-size:2.2rem;font-weight:700;color:#ffffff;'>
            🌳 Vancouver Urban Forest Scientific Dashboard
            &nbsp;&nbsp;{_d_filter_label}
        </h2>
        <p style='margin:8px 0 0 0;color:#b0bbd0;font-size:1.1rem;'>
            Scientific analysis of {len(_d_df_all):,} publicly maintained
            street trees · ArborGrid 2.0 · Phase 2
        </p>
    </div>
    """, unsafe_allow_html=True)

    if _d_df.empty:
        st.warning(
            "⚠️ No trees match the current filter combination.  "
            "Adjust the Urban Forest Filters in the sidebar to see data.",
            icon="🚨"
        )
    else:
        # =====================================================================
        # KPI — FOREST VITAL SIGNS
        # =====================================================================
        section("📊 Forest Vital Signs — Current Selection")

        _d_k1, _d_k2, _d_k3, _d_k4, _d_k5, _d_k6, _d_k7 = st.columns(7)

        _d_total     = len(_d_df)
        _d_n_genus   = _d_df["genus_name"].nunique()   if _d_HAS_GENUS   else "No Data"
        _d_n_species = _d_df["species_name"].nunique() if _d_HAS_SPECIES else "No Data"

        if _d_HAS_COMMON and _d_df["common_name"].notna().any():
            _d_top_name  = _d_df["common_name"].value_counts().idxmax()
            _d_top_pct   = (_d_df["common_name"].value_counts().iloc[0]
                            / _d_total * 100)
            _d_dominance = f"{_d_top_pct:.1f}%"
        else:
            _d_top_name, _d_top_pct, _d_dominance = "", 0, "No Data"

        _d_med_h  = (f"{_d_df['height_m'].median():.1f} m"
                     if _d_HAS_HEIGHT and _d_df["height_m"].notna().any()
                     else "No Data")
        _d_med_d  = (f"{_d_df['diameter_cm'].median():.1f} cm"
                     if _d_HAS_DBH and _d_df["diameter_cm"].notna().any()
                     else "No Data")
        _d_tot_ba = (f"{_d_df['basal_area_m2'].sum():,.0f} m²"
                     if "basal_area_m2" in _d_df.columns else "No Data")

        _d_k1.metric("🌳 Total Trees",            f"{_d_total:,}")
        _d_k2.metric("🔬 Unique Genera",           str(_d_n_genus))
        _d_k3.metric("🌿 Unique Species",          str(_d_n_species))
        _d_k4.metric("⚠️ Dominance Alert",         _d_dominance,
                     help=(f"Most common: {_d_top_name}"
                           if _d_top_name else None))
        _d_k5.metric("📏 Median Height",           _d_med_h)
        _d_k6.metric("📐 Median Trunk Diameter",   _d_med_d)
        _d_k7.metric("🌱 Total Basal Area",        _d_tot_ba,
                     help="Basal Area (π × (DBH/2)²) — proxy for biomass "
                          "and canopy")

        if _d_top_pct > 10:
            st.warning(
                f"⚠️ **Monoculture Risk Detected:** '{_d_top_name}' makes up "
                f"**{_d_top_pct:.1f}%** of the current selection. "
                "Urban forests where a single species exceeds 10% of the "
                "inventory are highly vulnerable to species-specific pests, "
                "diseases, and climate stress events.",
                icon="🚨"
            )

        st.divider()

        # =====================================================================
        # ANALYTICS TABS
        # Three tabs kept exactly as in the user's dashboard.py:
        #   1. Composition & Taxonomy
        #   2. Structure & Size
        #   3. Planting History
        # =====================================================================
        _d_tab_comp, _d_tab_struct, _d_tab_time = st.tabs([
            "🌿  Composition & Taxonomy",
            "📐  Structure & Size",
            "📅  Planting History",
        ])

        # ─────────────────────────────────────────────────────────────────────
        # TAB 1 — COMPOSITION & TAXONOMY
        # ─────────────────────────────────────────────────────────────────────
        with _d_tab_comp:

            st.divider()

            _d_left_c, _d_right_c = st.columns(2, gap="large")

            # Pareto Chart — Top 20 species by common name
            with _d_left_c:
                section("📊 Pareto Chart — Top 20 Species by Tree Count")
                st.caption(
                    "Bars = individual species counts (from 'common_name' "
                    "in the data).  Orange line = cumulative percentage.  "
                    "Where the line crosses 80% shows how concentrated the "
                    "forest is."
                )
                if _d_HAS_COMMON and _d_df["common_name"].notna().any():
                    _d_pareto = (_d_df["common_name"].value_counts()
                                 .head(20).reset_index())
                    _d_pareto.columns = ["Species", "Count"]
                    _d_pareto["Cumulative Percentage"] = (
                        _d_pareto["Count"].cumsum()
                        / _d_pareto["Count"].sum() * 100
                    )
                    _d_fig_p = make_subplots(specs=[[{"secondary_y": True}]])
                    _d_fig_p.add_trace(
                        go.Bar(
                            x=_d_pareto["Species"],
                            y=_d_pareto["Count"],
                            name="Tree Count",
                            marker_color=PALETTE[1],
                            hovertemplate=(
                                "%{x}<br>Count: %{y:,}<extra></extra>"
                            ),
                        ),
                        secondary_y=False,
                    )
                    _d_fig_p.add_trace(
                        go.Scatter(
                            x=_d_pareto["Species"],
                            y=_d_pareto["Cumulative Percentage"],
                            name="Cumulative Percentage",
                            mode="lines+markers",
                            line=dict(color=PALETTE[2], width=2),
                            marker=dict(size=5),
                            hovertemplate=(
                                "%{x}<br>Cumulative: %{y:.1f}%<extra></extra>"
                            ),
                        ),
                        secondary_y=True,
                    )
                    _d_fig_p.add_hline(
                        y=80, line_dash="dash", line_color="#e74c3c",
                        annotation_text="80% Threshold",
                        annotation_position="top right",
                        secondary_y=True,
                    )
                    _d_fig_p.update_yaxes(
                        title_text="Number of Trees",
                        gridcolor=BORDER, secondary_y=False)
                    _d_fig_p.update_yaxes(
                        title_text="Cumulative Percentage",
                        range=[0, 105], secondary_y=True)
                    _d_fig_p.update_layout(
                        template=PLOTLY_TEMPLATE,
                        paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
                        height=440,
                        margin=dict(t=14, b=110, l=0, r=60),
                        xaxis=dict(tickangle=-42, tickfont_size=9,
                                   gridcolor=BORDER),
                        legend=dict(orientation="h", y=1.1),
                    )
                    st.plotly_chart(_d_fig_p, use_container_width=True)
                else:
                    no_data_box(
                        "Common name data ('common_name') is not available "
                        "in the loaded dataset.  The Pareto chart cannot be "
                        "generated."
                    )

            # Top 20 Genera bar
            with _d_right_c:
                section("🔬 Top 20 Genera by Tree Count")
                st.caption(
                    "Genus names are read directly from the 'genus_name' "
                    "column in the data — nothing is assumed or invented.  "
                    "Bars are sorted from largest to smallest."
                )
                if _d_HAS_GENUS and _d_df["genus_name"].notna().any():
                    _d_gdf2 = (_d_df["genus_name"].value_counts()
                               .head(20).reset_index())
                    _d_gdf2.columns = ["Genus Name", "Tree Count"]
                    _d_null_g = int(_d_df["genus_name"].isna().sum())

                    _d_fig_g = px.bar(
                        _d_gdf2.sort_values("Tree Count"),
                        x="Tree Count", y="Genus Name",
                        orientation="h",
                        color="Tree Count",
                        color_continuous_scale="Teal",
                        template=PLOTLY_TEMPLATE,
                        text_auto=",",
                    )
                    _d_fig_g.update_traces(
                        textposition="outside",
                        hovertemplate="%{y}<br>Trees: %{x:,}<extra></extra>",
                    )
                    _d_fig_g.update_layout(
                        paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
                        height=440,
                        margin=dict(t=14, b=20, l=0, r=50),
                        coloraxis_showscale=False,
                        xaxis=dict(gridcolor=BORDER,
                                   title="Number of Trees"),
                        yaxis=dict(tickfont_size=11),
                    )
                    st.plotly_chart(_d_fig_g, use_container_width=True)

                    if _d_null_g > 0:
                        st.info(
                            f"ℹ️ {_d_null_g:,} trees have no genus name "
                            f"recorded and are excluded from this chart.",
                            icon="ℹ️"
                        )
                else:
                    no_data_box(
                        "Genus name data ('genus_name') is not available "
                        "in the loaded dataset."
                    )

            st.divider()

            # Cultivar breakdown
            section("🌿 Cultivar Breakdown for Selected Genus / Species")
            st.caption(
                "Cultivar names are read directly from the 'cultivar_name' "
                "column.  Showing the top 25 cultivars within the current "
                "filter selection.  If no genus or species filter is active, "
                "the top 25 across all trees are shown."
            )
            if _d_HAS_CULTI and _d_df["cultivar_name"].notna().any():
                _d_cv_df = (_d_df["cultivar_name"].value_counts()
                            .head(25).reset_index())
                _d_cv_df.columns = ["Cultivar Name", "Tree Count"]
                _d_null_cv = int(_d_df["cultivar_name"].isna().sum())

                _d_fig_cv = px.bar(
                    _d_cv_df.sort_values("Tree Count"),
                    x="Tree Count", y="Cultivar Name",
                    orientation="h",
                    color="Tree Count",
                    color_continuous_scale="Viridis",
                    template=PLOTLY_TEMPLATE,
                    text_auto=",",
                )
                _d_fig_cv.update_traces(
                    textposition="outside",
                    hovertemplate="%{y}<br>Trees: %{x:,}<extra></extra>",
                )
                _d_fig_cv.update_layout(
                    paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
                    height=480,
                    margin=dict(t=14, b=10, l=0, r=50),
                    coloraxis_showscale=False,
                    xaxis=dict(gridcolor=BORDER, title="Number of Trees"),
                    yaxis=dict(tickfont_size=11),
                )
                st.plotly_chart(_d_fig_cv, use_container_width=True)

                if _d_null_cv > 0:
                    st.info(
                        f"ℹ️ {_d_null_cv:,} trees have no cultivar name "
                        f"recorded in the source data.  These are excluded "
                        f"from the chart above.",
                        icon="ℹ️"
                    )
            else:
                no_data_box(
                    "Cultivar name data ('cultivar_name') is not available "
                    "or all values are empty in the current selection."
                )

        # ─────────────────────────────────────────────────────────────────────
        # TAB 2 — STRUCTURE & SIZE
        # ─────────────────────────────────────────────────────────────────────
        with _d_tab_struct:

            # [2, 1] ratio: left column (big chart) is twice the width
            _d_s1, _d_s2 = st.columns([2, 1], gap="large")

            # LEFT — Height box plots (the big chart)
            with _d_s1:
                section("📏 Tree Height Distribution by Genus — Top 12")
                st.caption(
                    "Box shows median, interquartile range, and whiskers.  "
                    "Sorted by median height, tallest first."
                )
                if (_d_HAS_HEIGHT and _d_HAS_GENUS
                        and _d_df["height_m"].notna().any()):
                    _d_null_ht = int(_d_df["height_m"].isna().sum())
                    _d_top12g  = (_d_df["genus_name"].value_counts()
                                  .head(12).index)
                    _d_box_df  = _d_df[
                        _d_df["genus_name"].isin(_d_top12g)
                        & _d_df["height_m"].notna()
                    ].copy()

                    if not _d_box_df.empty:
                        _d_order = (
                            _d_box_df.groupby("genus_name")["height_m"]
                            .median()
                            .sort_values(ascending=False)
                            .index.tolist()
                        )
                        _d_fig_box = px.box(
                            _d_box_df,
                            x="genus_name", y="height_m",
                            color="genus_name",
                            category_orders={"genus_name": _d_order},
                            color_discrete_sequence=PALETTE,
                            template=PLOTLY_TEMPLATE,
                            points="outliers",
                        )
                        _d_fig_box.update_layout(
                            paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
                            height=750,
                            margin=dict(t=20, b=120, l=10, r=10),
                            showlegend=False,
                            xaxis=dict(tickangle=-45, gridcolor=BORDER,
                                       title="Genus Name"),
                            yaxis=dict(gridcolor=BORDER,
                                       title="Tree Height (Metres)"),
                        )
                        st.plotly_chart(_d_fig_box, use_container_width=True)

                        if _d_null_ht > 0:
                            st.info(
                                f"ℹ️ {_d_null_ht:,} trees excluded due to "
                                f"missing height data.",
                                icon="ℹ️"
                            )
                    else:
                        no_data_box(
                            "No height data available for the top genera "
                            "in the current filter selection."
                        )
                else:
                    no_data_box("Height or Genus data is unavailable.")

            # RIGHT — Size class donut + Height class donut
            with _d_s2:

                section("📊 Size Class Distribution")
                if "dbh_class" in _d_df.columns:
                    _d_dbc = _d_df["dbh_class"].value_counts().reset_index()
                    _d_dbc.columns = ["Size Class", "Count"]
                    _d_null_dbc = int(_d_df["dbh_class"].isna().sum())
                    if _d_null_dbc > 0:
                        _d_dbc = pd.concat([
                            _d_dbc,
                            pd.DataFrame([{
                                "Size Class": "No Diameter Data Recorded",
                                "Count": _d_null_dbc,
                            }])
                        ], ignore_index=True)

                    _d_fig_dbc = px.pie(
                        _d_dbc, names="Size Class", values="Count",
                        color_discrete_sequence=[
                            "#2D6A4F", "#52B788", "#95D5B2",
                            "#D8E2DC", "#7F8C8D"],
                        hole=0.6, template=PLOTLY_TEMPLATE,
                    )
                    _d_fig_dbc.update_traces(
                        textinfo="percent+label",
                        hovertemplate=(
                            "%{label}<br>"
                            "Trees: %{value:,}<extra></extra>"
                        ),
                    )
                    _d_fig_dbc.update_layout(
                        paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
                        height=350,
                        margin=dict(t=40, b=20, l=10, r=10),
                        showlegend=True,
                        legend=dict(
                            orientation="h", yanchor="bottom",
                            y=-0.2, xanchor="center", x=0.5),
                    )
                    st.plotly_chart(_d_fig_dbc, use_container_width=True)
                else:
                    no_data_box(
                        "Diameter data is not available — size class "
                        "chart cannot be generated."
                    )

                st.write("---")

                section("🌲 Height Class Breakdown")
                if _d_HAS_HEIGHT and "height_class" in _d_df.columns:
                    _d_hc = (_d_df["height_class"].value_counts()
                             .reset_index())
                    _d_hc.columns = ["Height Class", "Count"]
                    _d_null_hc = int(_d_df["height_class"].isna().sum())
                    if _d_null_hc > 0:
                        _d_hc = pd.concat([
                            _d_hc,
                            pd.DataFrame([{
                                "Height Class": "No Height Data Recorded",
                                "Count": _d_null_hc,
                            }])
                        ], ignore_index=True)

                    _d_fig_hc = px.pie(
                        _d_hc, names="Height Class", values="Count",
                        color_discrete_sequence=[
                            "#1B4332", "#40916C", "#74C69D", "#B7E4C7"],
                        hole=0.6, template=PLOTLY_TEMPLATE,
                    )
                    _d_fig_hc.update_traces(
                        textinfo="percent+label",
                        hovertemplate=(
                            "%{label}<br>"
                            "Trees: %{value:,}<extra></extra>"
                        ),
                    )
                    _d_fig_hc.update_layout(
                        paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
                        height=350,
                        margin=dict(t=40, b=20, l=10, r=10),
                        showlegend=True,
                        legend=dict(
                            orientation="h", yanchor="bottom",
                            y=-0.2, xanchor="center", x=0.5),
                    )
                    st.plotly_chart(_d_fig_hc, use_container_width=True)
                else:
                    no_data_box(
                        "Height data is not available — height class "
                        "chart cannot be generated."
                    )

        # ─────────────────────────────────────────────────────────────────────
        # TAB 3 — PLANTING HISTORY
        # ─────────────────────────────────────────────────────────────────────
        with _d_tab_time:

            if not _d_HAS_DATE:
                no_data_box(
                    "Planting date data ('date_planted') is not available "
                    "in the loaded dataset.  The column was either absent "
                    "in the original source data or was not retained during "
                    "the fetch_data.py download step.  No planting history "
                    "charts can be generated."
                )
            else:
                _d_dated   = _d_df[_d_df["plant_year"].notna()].copy()
                _d_undated = int(_d_df["plant_year"].isna().sum())

                if _d_dated.empty:
                    no_data_box(
                        "The 'date_planted' column exists but all values "
                        "are empty for the current filter selection.  "
                        "No planting history charts can be generated."
                    )
                else:
                    _d_t1, _d_t2 = st.columns(2, gap="large")

                    with _d_t1:
                        section("📈 Trees Planted Per Year")
                        st.caption(
                            "Annual count of trees with a recorded planting "
                            "date.  Gaps may reflect missing records.  "
                            "Spikes often correspond to large planting "
                            "programmes or post-storm replacement cycles."
                        )
                        _d_yr = (_d_dated["plant_year"].value_counts()
                                 .sort_index().reset_index())
                        _d_yr.columns = ["Year", "Trees Planted"]

                        _d_fig_yr = px.area(
                            _d_yr, x="Year", y="Trees Planted",
                            template=PLOTLY_TEMPLATE,
                            color_discrete_sequence=[PALETTE[1]],
                            line_shape="spline",
                        )
                        _d_fig_yr.update_traces(
                            hovertemplate=(
                                "Year: %{x}<br>"
                                "Trees Planted: %{y:,}<extra></extra>"
                            ),
                            fillcolor="rgba(39,174,96,0.25)",
                        )
                        _d_fig_yr.update_layout(
                            paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
                            height=340,
                            margin=dict(t=14, b=14, l=0, r=0),
                            xaxis=dict(gridcolor=BORDER, title="Year"),
                            yaxis=dict(gridcolor=BORDER,
                                       title="Number of Trees Planted"),
                        )
                        st.plotly_chart(_d_fig_yr, use_container_width=True)

                    with _d_t2:
                        section("📊 Trees Planted Per Decade")
                        st.caption(
                            "Decade-level view smooths annual variation and "
                            "shows long-term planting policy trends."
                        )
                        _d_dec = (_d_dated["plant_decade"].dropna()
                                  .astype(int).value_counts()
                                  .sort_index().reset_index())
                        _d_dec.columns = ["Decade", "Trees Planted"]
                        _d_dec["Decade Label"] = (
                            _d_dec["Decade"].astype(str) + "s"
                        )
                        _d_fig_dec = px.bar(
                            _d_dec, x="Decade Label", y="Trees Planted",
                            color="Trees Planted",
                            color_continuous_scale="Greens",
                            template=PLOTLY_TEMPLATE,
                            text_auto=",",
                        )
                        _d_fig_dec.update_traces(
                            textposition="outside",
                            hovertemplate=(
                                "Decade: %{x}<br>"
                                "Trees Planted: %{y:,}<extra></extra>"
                            ),
                        )
                        _d_fig_dec.update_layout(
                            paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
                            height=340,
                            margin=dict(t=14, b=20, l=0, r=0),
                            coloraxis_showscale=False,
                            xaxis=dict(gridcolor=BORDER, title="Decade"),
                            yaxis=dict(gridcolor=BORDER,
                                       title="Number of Trees Planted"),
                        )
                        st.plotly_chart(_d_fig_dec, use_container_width=True)

                    if _d_undated > 0:
                        st.info(
                            f"ℹ️ {_d_undated:,} trees "
                            f"({_d_undated / len(_d_df) * 100:.1f}% of the "
                            f"current selection) have no planting date "
                            f"recorded and are not shown in the timeline "
                            f"charts above.  Missing dates are common for "
                            f"trees established before digital record-keeping "
                            f"began.",
                            icon="ℹ️"
                        )

                    st.divider()

                    section("🌿 Most Planted Species Per Decade")
                    st.caption(
                        "The single most frequently planted species in each "
                        "decade.  Species names come directly from the "
                        "'common_name' column."
                    )
                    if _d_HAS_COMMON:
                        _d_dec_sp = (
                            _d_dated[
                                _d_dated["plant_decade"].notna()
                                & _d_dated["common_name"].notna()
                            ]
                            .groupby(["plant_decade", "common_name"])
                            .size().reset_index(name="Count")
                        )
                        if _d_dec_sp.empty:
                            no_data_box(
                                "No rows have both a planting decade and a "
                                "species name in the current selection."
                            )
                        else:
                            _d_top_dec = (
                                _d_dec_sp
                                .sort_values("Count", ascending=False)
                                .groupby("plant_decade").head(1)
                                .sort_values("plant_decade")
                                .rename(columns={
                                    "plant_decade": "Decade",
                                    "common_name":  "Most Planted Species",
                                    "Count":        "Trees Planted",
                                })
                            )
                            _d_top_dec["Decade"] = (
                                _d_top_dec["Decade"]
                                .astype(int).astype(str) + "s"
                            )
                            st.dataframe(
                                _d_top_dec,
                                use_container_width=True,
                                hide_index=True,
                            )
                    else:
                        no_data_box(
                            "Common name data ('common_name') is not "
                            "available — dominant species per decade cannot "
                            "be determined."
                        )

else:
    # trees_raw.geojson missing — clear message, never a silent blank
    st.divider()
    st.warning(
        "⚠️ **Urban Forest Dashboard unavailable.** "
        f"`trees_raw.geojson` was not found at `{TREES_F}`.  "
        "Run `fetch_data.py` first, then refresh.",
        icon="🌳"
    )


# =============================================================================
# FOOTER
# =============================================================================
st.divider()
cf1, cf2, cf3 = st.columns(3)
cf1.caption("🌳 ArborGRID | Phase 2")
cf2.caption("📍 Vancouver, BC 🇨🇦")

cf3.caption("⚡ Streamlit · Folium · GeoPandas · Shapely")


