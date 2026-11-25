"""
AgroSat Lilongwe — Agricultural Satellite Monitoring & Drought Early-Warning Platform
Single-file Streamlit application for smallholder & commercial farm monitoring
in Lilongwe, Malawi using Google Earth Engine (Sentinel-2 SR, Sentinel-1 GRD, CHIRPS, WorldCover).
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any

import branca.colormap as cm
import ee
import folium
from folium.plugins import Draw
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Page Configuration & Constants
# ---------------------------------------------------------------------------
LILONGWE_LAT = -13.9626
LILONGWE_LON = 33.7741
DEFAULT_ZOOM = 12
S2_COLLECTION_ID = "COPERNICUS/S2_SR_HARMONIZED"
S1_COLLECTION_ID = "COPERNICUS/S1_GRD"
CHIRPS_COLLECTION_ID = "UCSB-CHG/CHIRPS/DAILY"
WORLDCOVER_COLLECTION_ID = "ESA/WorldCover/v200"

LOOKBACK_DAYS = 30
MAX_CLOUD_PERCENT = 50
MIN_BUFFER_PLOT_AREA_M2 = 4000.0
MIN_BUFFERED_REMAINING_M2 = 900.0
INWARD_BUFFER_METERS = -7.5
MIN_CROPLAND_FRACTION = 0.15
MIN_DIMBA_FRACTION = 0.25
MAX_WATER_FRACTION = 0.30

NDVI_PALETTE = [
    "#a50026",
    "#d73027",
    "#f46d43",
    "#fdae61",
    "#fee08b",
    "#d9ef8b",
    "#a6d96a",
    "#66bd63",
    "#1a9850",
    "#006837",
]

st.set_page_config(
    page_title="AgroSat Lilongwe | Drought & Crop Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Helper Formatting, Season Calendar & Coordinate Geometry Builder
# ---------------------------------------------------------------------------
def format_mwk_compact(amount: float) -> str:
    """Format large MWK figures cleanly so Streamlit metric cards don't truncate."""
    abs_val = abs(amount)
    if abs_val >= 1_000_000_000:
        return f"MWK {amount / 1_000_000_000:.2f}B"
    if abs_val >= 1_000_000:
        return f"MWK {amount / 1_000_000:.2f}M"
    if abs_val >= 1_000:
        return f"MWK {amount / 1_000:.1f}K"
    return f"MWK {amount:,.0f}"


def format_area_compact(area_ha: float) -> str:
    """Format farm area cleanly without overflowing narrow metric cards."""
    if area_ha >= 10_000:
        return f"{area_ha / 1_000:.1f}k ha"
    if area_ha >= 100:
        return f"{area_ha:.1f} ha"
    return f"{area_ha:.2f} ha"


def get_malawi_season_stage(month: int) -> dict[str, Any]:
    """
    Return Malawi unimodal agricultural season stage and NDVI thresholds for a given month.
    - Nov-Dec: Planting / Early Emergence
    - Jan-Mar: Peak Vegetative / Tasseling
    - Apr-May: Grain Filling / Maturation
    - Jun-Oct: Dry / Off-Season
    """
    if month in (11, 12):
        return {
            "stage": "Planting / Early Emergence",
            "healthy_ndvi": 0.42,
            "stress_ndvi": 0.22,
            "early_emergence_band": True,
            "default_retention": 0.92,
        }
    if month in (1, 2, 3):
        return {
            "stage": "Peak Vegetative / Tasseling",
            "healthy_ndvi": 0.55,
            "stress_ndvi": 0.35,
            "early_emergence_band": False,
            "default_retention": 0.95,
        }
    if month in (4, 5):
        return {
            "stage": "Grain Filling / Maturation",
            "healthy_ndvi": 0.45,
            "stress_ndvi": 0.25,
            "early_emergence_band": False,
            "default_retention": 0.95,
        }
    return {
        "stage": "Dry / Off-Season",
        "healthy_ndvi": 0.45,
        "stress_ndvi": 0.20,
        "early_emergence_band": False,
        "default_retention": 0.90,
    }


def coords_to_polygon_geojson(lat: float, lon: float, radius_m: float) -> dict[str, Any]:
    """
    Create an approximate square polygon (GeoJSON) around (lat, lon) with half-width
    `radius_m` meters so point/GPS coordinates work seamlessly with polygon analysis.
    """
    half_side = max(25.0, float(radius_m))
    lat_rad = math.radians(lat)
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = max(10_000.0, 111_320.0 * math.cos(lat_rad))

    d_lat = half_side / meters_per_deg_lat
    d_lon = half_side / meters_per_deg_lon

    ring = [
        [lon - d_lon, lat - d_lat],
        [lon + d_lon, lat - d_lat],
        [lon + d_lon, lat + d_lat],
        [lon - d_lon, lat + d_lat],
        [lon - d_lon, lat - d_lat],
    ]
    return {"type": "Polygon", "coordinates": [ring]}


# ---------------------------------------------------------------------------
# 1. Google Earth Engine Service Account Authentication
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Authenticating with Google Earth Engine...")
def initialize_gee() -> tuple[bool, str]:
    """Authenticate EE using st.secrets['GEE_SERVICE_ACCOUNT']."""
    if "GEE_SERVICE_ACCOUNT" not in st.secrets:
        return (
            False,
            "Missing `GEE_SERVICE_ACCOUNT` in Streamlit secrets. "
            "Add your Google Cloud service account JSON to `.streamlit/secrets.toml` "
            "or Streamlit Community Cloud Secrets.",
        )

    try:
        raw_secret = st.secrets["GEE_SERVICE_ACCOUNT"]
        if isinstance(raw_secret, str):
            sa_info: dict[str, Any] = json.loads(raw_secret)
        else:
            sa_info = dict(raw_secret)

        if "private_key" in sa_info and isinstance(sa_info["private_key"], str):
            sa_info["private_key"] = sa_info["private_key"].replace("\\n", "\n")

        client_email = sa_info.get("client_email")
        project_id = sa_info.get("project_id")
        if not client_email or not sa_info.get("private_key"):
            return (
                False,
                "`GEE_SERVICE_ACCOUNT` is missing required fields (`client_email`, `private_key`).",
            )

        credentials = ee.ServiceAccountCredentials(
            email=client_email,
            key_data=json.dumps(sa_info),
        )
        if project_id:
            ee.Initialize(credentials=credentials, project=project_id)
        else:
            ee.Initialize(credentials=credentials)

        return True, f"Connected as {client_email}"
    except Exception as exc:
        return False, f"Earth Engine authentication failed: {exc}"


# ---------------------------------------------------------------------------
# 2. Malawi Geoscience Cloud Masking, Geometry & Spectral Index Calculation
# ---------------------------------------------------------------------------
def mask_malawi_s2_clouds(image: ee.Image) -> ee.Image:
    """
    Mask clouds, cirrus, cloud shadows, and saturated/defective pixels
    for Sentinel-2 SR over Malawi using QA60 and SCL.
    """
    qa = image.select("QA60")
    cloud_free_qa = (
        qa.bitwiseAnd(1 << 10)
        .eq(0)
        .And(qa.bitwiseAnd(1 << 11).eq(0))
    )

    scl = image.select("SCL")
    cloud_free_scl = (
        scl.neq(1)
        .And(scl.neq(3))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
    )

    valid_mask = cloud_free_qa.And(cloud_free_scl)
    return image.updateMask(valid_mask).copyProperties(image, ["system:time_start"])


def add_enhanced_indices(image: ee.Image) -> ee.Image:
    """
    Compute vegetation, red-edge chlorophyll, moisture, and bare-soil indices
    tailored for Lilongwe / Central Region agricultural landscapes in Malawi.
    """
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndre = image.normalizedDifference(["B8A", "B5"]).rename("NDRE")
    savi = (
        image.expression(
            "((NIR - RED) / (NIR + RED + 0.5)) * 1.5",
            {
                "NIR": image.select("B8"),
                "RED": image.select("B4"),
            },
        )
        .rename("SAVI")
    )
    ndwi_canopy = image.normalizedDifference(["B8", "B11"]).rename("NDWI")
    ndwi_surface = image.normalizedDifference(["B3", "B8"]).rename("NDWI_SURFACE")
    bsi = (
        image.expression(
            "((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE) + 1e-6)",
            {
                "SWIR1": image.select("B11"),
                "RED": image.select("B4"),
                "NIR": image.select("B8"),
                "BLUE": image.select("B2"),
            },
        )
        .rename("BSI")
    )

    return image.addBands([ndvi, ndre, savi, ndwi_canopy, ndwi_surface, bsi])


def _build_month_day_filter(start_dt: datetime, end_dt: datetime) -> ee.Filter:
    """Build a calendar-window filter across multiple years without looping per year."""
    start_m = start_dt.month
    start_d = start_dt.day
    end_m = end_dt.month
    end_d = end_dt.day

    if start_m == end_m:
        return ee.Filter.And(
            ee.Filter.calendarRange(start_m, end_m, "month"),
            ee.Filter.calendarRange(start_d, end_d, "day_of_month"),
        )
    if start_m < end_m:
        start_part = ee.Filter.And(
            ee.Filter.calendarRange(start_m, start_m, "month"),
            ee.Filter.calendarRange(start_d, 31, "day_of_month"),
        )
        middle_part = (
            ee.Filter.calendarRange(start_m + 1, end_m - 1, "month")
            if (end_m - start_m) > 1
            else None
        )
        end_part = ee.Filter.And(
            ee.Filter.calendarRange(end_m, end_m, "month"),
            ee.Filter.calendarRange(1, end_d, "day_of_month"),
        )
        filters = [f for f in [start_part, middle_part, end_part] if f is not None]
        return ee.Filter.Or(*filters)

    dec_part = ee.Filter.And(
        ee.Filter.calendarRange(start_m, 12, "month"),
        ee.Filter.calendarRange(start_d, 31, "day_of_month"),
    )
    jan_part = ee.Filter.And(
        ee.Filter.calendarRange(1, end_m, "month"),
        ee.Filter.calendarRange(1, end_d, "day_of_month"),
    )
    return ee.Filter.Or(dec_part, jan_part)


def compute_rainfall_and_radar_signals(
    aoi: ee.Geometry,
    lookback_days: int,
    end_dt: datetime,
    area_ha: float = 1.0,
) -> dict[str, Any]:
    """
    Compute CHIRPS rainfall (lag-adjusted + 10-year baseline) and Sentinel-1
    DESCENDING IW VV radar backscatter in separate try/except blocks with safe
    empty-collection band handling so Dictionary.get never fails.
    """
    cur_rain = 0.0
    hist_rain = 0.0
    rain_anomaly_pct = 0.0
    dry_days = 0
    s1_vv: float | None = None
    status_parts: list[str] = []

    chirps_scale = 11132 if area_ha > 50 else 5566

    # 1. Independent CHIRPS block
    try:
        chirps_end_dt = end_dt - timedelta(days=5)
        chirps_start_dt = chirps_end_dt - timedelta(days=lookback_days)
        c_start_str = chirps_start_dt.strftime("%Y-%m-%d")
        c_end_str = (chirps_end_dt + timedelta(days=1)).strftime("%Y-%m-%d")

        chirps_current = (
            ee.ImageCollection(CHIRPS_COLLECTION_ID)
            .filterBounds(aoi)
            .filterDate(c_start_str, c_end_str)
            .select(["precipitation"])
        )
        chirps_count_ee = chirps_current.size()

        current_rain_img = ee.Image(
            ee.Algorithms.If(
                chirps_count_ee.gt(0),
                chirps_current.sum().unmask(0.0).rename("precipitation"),
                ee.Image.constant(0.0).rename("precipitation"),
            )
        )
        current_rain_mm = current_rain_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=chirps_scale,
            maxPixels=1_000_000_000,
            bestEffort=True,
        ).get("precipitation")

        dry_days_img = ee.Image(
            ee.Algorithms.If(
                chirps_count_ee.gt(0),
                chirps_current.map(lambda img: img.lt(1.0).rename("dry")).sum().unmask(0.0).rename("dry"),
                ee.Image.constant(0.0).rename("dry"),
            )
        )
        dry_days_val = dry_days_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=chirps_scale,
            maxPixels=1_000_000_000,
            bestEffort=True,
        ).get("dry")

        hist_start_str = f"{end_dt.year - 10}-01-01"
        hist_end_str = f"{end_dt.year}-01-01"
        cal_filter = _build_month_day_filter(chirps_start_dt, chirps_end_dt)

        hist_col = (
            ee.ImageCollection(CHIRPS_COLLECTION_ID)
            .filterBounds(aoi)
            .filterDate(hist_start_str, hist_end_str)
            .filter(cal_filter)
            .select(["precipitation"])
        )
        hist_10yr_img = ee.Image(
            ee.Algorithms.If(
                hist_col.size().gt(0),
                hist_col.sum().divide(10.0).unmask(0.0).rename("precipitation"),
                ee.Image.constant(0.0).rename("precipitation"),
            )
        )
        hist_10yr_avg = hist_10yr_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=chirps_scale,
            maxPixels=1_000_000_000,
            bestEffort=True,
        ).get("precipitation")

        chirps_bundle = ee.Dictionary(
            {
                "chirps_count": chirps_count_ee,
                "current_rain_mm": current_rain_mm,
                "hist_mean_rain_mm": hist_10yr_avg,
                "dry_days": dry_days_val,
            }
        ).getInfo()

        chirps_count = int(chirps_bundle.get("chirps_count") or 0)
        hist_rain = (
            float(chirps_bundle["hist_mean_rain_mm"])
            if chirps_bundle.get("hist_mean_rain_mm") is not None
            else 0.0
        )

        if chirps_count == 0:
            cur_rain = 0.0
            dry_days = 0
            rain_anomaly_pct = 0.0
            status_parts.append(
                "CHIRPS scenes not yet published for selected end date (check system date or latency)"
            )
        else:
            cur_rain = (
                float(chirps_bundle["current_rain_mm"])
                if chirps_bundle.get("current_rain_mm") is not None
                else 0.0
            )
            dry_days = (
                int(round(float(chirps_bundle["dry_days"])))
                if chirps_bundle.get("dry_days") is not None
                else 0
            )
            if hist_rain >= 5.0:
                rain_anomaly_pct = ((cur_rain - hist_rain) / hist_rain) * 100.0
                status_parts.append("CHIRPS (lag-adjusted 10-yr) active")
            else:
                rain_anomaly_pct = 0.0
                status_parts.append("Off-season / dry baseline (<5mm)")
    except Exception as exc:
        status_parts.append(f"CHIRPS unavailable ({str(exc)[:80]})")

    # 2. Independent Sentinel-1 block with unmask(-999) guard for empty collections
    try:
        s1_start_str = (end_dt - timedelta(days=max(lookback_days, 30))).strftime("%Y-%m-%d")
        s1_end_str = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")

        s1_col = (
            ee.ImageCollection(S1_COLLECTION_ID)
            .filterBounds(aoi)
            .filterDate(s1_start_str, s1_end_str)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .sort("system:time_start", False)
            .limit(6)
            .select(["VV"])
        )

        s1_vv_raw = (
            s1_col.mean()
            .unmask(-999.0)
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=aoi,
                scale=20,
                maxPixels=1_000_000_000,
                bestEffort=True,
            )
            .get("VV")
            .getInfo()
        )

        if s1_vv_raw is not None and float(s1_vv_raw) > -900.0:
            s1_vv = round(float(s1_vv_raw), 2)
            status_parts.append("S1 DESC active")
        else:
            s1_vv = None
            status_parts.append("S1 DESC unavailable (no scenes)")
    except Exception as exc:
        s1_vv = None
        status_parts.append(f"S1 DESC unavailable ({str(exc)[:80]})")

    return {
        "rain_mm": round(cur_rain, 1),
        "hist_rain_mm": round(hist_rain, 1),
        "rain_anomaly_pct": round(rain_anomaly_pct, 1),
        "dry_days": dry_days,
        "s1_vv_db": s1_vv,
        "climate_status": " · ".join(status_parts),
    }


def extract_polygon_geometry(map_output: dict[str, Any] | None) -> dict[str, Any] | None:
    """Extract the latest Polygon or MultiPolygon GeoJSON geometry from st_folium output."""
    if not map_output:
        return None

    last_drawing = map_output.get("last_active_drawing")
    if isinstance(last_drawing, dict) and last_drawing.get("geometry"):
        geom = last_drawing["geometry"]
        if geom.get("type") in ("Polygon", "MultiPolygon"):
            return geom

    all_drawings = map_output.get("all_drawings") or []
    for feature in reversed(all_drawings):
        if isinstance(feature, dict):
            geom = feature.get("geometry")
            if geom and geom.get("type") in ("Polygon", "MultiPolygon"):
                return geom

    return None


def run_farm_analysis(
    geojson_geom: dict[str, Any],
    lookback_days: int = LOOKBACK_DAYS,
    max_cloud_pct: int = MAX_CLOUD_PERCENT,
) -> dict[str, Any]:
    """
    Run GEE Sentinel-2, Sentinel-1, CHIRPS, and ESA WorldCover cropland/dimba
    analysis for the target geometry using a single bundled core .getInfo() call
    and server-side inward edge buffer alignment.
    """
    raw_aoi = ee.Geometry(geojson_geom)
    raw_area_sqm = raw_aoi.area(maxError=1)
    buffered_geom = raw_aoi.buffer(distance=INWARD_BUFFER_METERS, maxError=1)
    buffered_area_sqm = buffered_geom.area(maxError=1)

    analysis_aoi_ee = ee.Geometry(
        ee.Algorithms.If(
            raw_area_sqm.gte(MIN_BUFFER_PLOT_AREA_M2).And(
                buffered_area_sqm.gte(MIN_BUFFERED_REMAINING_M2)
            ),
            buffered_geom,
            raw_aoi,
        )
    )

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=lookback_days)

    worldcover = ee.ImageCollection(WORLDCOVER_COLLECTION_ID).first().select("Map")
    primary_crop_mask = worldcover.eq(40)
    dimba_wetland_mask = worldcover.eq(90)
    water_mask = worldcover.eq(80)

    collection = (
        ee.ImageCollection(S2_COLLECTION_ID)
        .filterBounds(analysis_aoi_ee)
        .filterDate(
            start_dt.strftime("%Y-%m-%d"),
            (end_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .map(mask_malawi_s2_clouds)
        .map(add_enhanced_indices)
    )

    scene_count_ee = collection.size()
    band_list = ["NDVI", "NDRE", "SAVI", "NDWI", "NDWI_SURFACE", "BSI"]

    raw_composite = ee.Algorithms.If(
        scene_count_ee.gt(0),
        collection.select(band_list).median().clip(analysis_aoi_ee),
        None,
    )

    unmasked_stats_ee = ee.Algorithms.If(
        scene_count_ee.gt(0),
        ee.Image(raw_composite).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=analysis_aoi_ee,
            scale=10,
            maxPixels=1_000_000_000,
            bestEffort=True,
        ),
        ee.Dictionary({}),
    )

    landcover_fractions_img = ee.Image.cat(
        [
            primary_crop_mask.rename("crop"),
            dimba_wetland_mask.rename("dimba"),
            water_mask.rename("water"),
        ]
    ).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=analysis_aoi_ee,
        scale=10,
        maxPixels=1_000_000_000,
        bestEffort=True,
    )

    centroid_ee = raw_aoi.centroid(maxError=1).coordinates()

    core_bundle = ee.Dictionary(
        {
            "raw_area_sqm": raw_area_sqm,
            "buffered_area_sqm": buffered_area_sqm,
            "scene_count": scene_count_ee,
            "crop_fraction": landcover_fractions_img.get("crop"),
            "dimba_fraction": landcover_fractions_img.get("dimba"),
            "water_fraction": landcover_fractions_img.get("water"),
            "unmasked_stats": unmasked_stats_ee,
            "centroid": centroid_ee,
        }
    ).getInfo()

    raw_sqm = float(core_bundle.get("raw_area_sqm") or 0.0)
    buf_sqm = float(core_bundle.get("buffered_area_sqm") or 0.0)
    raw_area_ha = raw_sqm / 10_000.0

    if raw_sqm >= MIN_BUFFER_PLOT_AREA_M2 and buf_sqm >= MIN_BUFFERED_REMAINING_M2:
        analysis_aoi = buffered_geom
        analysis_area_ha = buf_sqm / 10_000.0
        base_buffer_note = f"Inward {INWARD_BUFFER_METERS:.1f}m edge buffer applied"
    else:
        analysis_aoi = raw_aoi
        analysis_area_ha = raw_area_ha
        base_buffer_note = "Full drawn/coord boundary used"

    scene_count = int(core_bundle.get("scene_count") or 0)
    crop_fraction = (
        float(core_bundle["crop_fraction"])
        if core_bundle.get("crop_fraction") is not None
        else 0.0
    )
    dimba_fraction = (
        float(core_bundle["dimba_fraction"])
        if core_bundle.get("dimba_fraction") is not None
        else 0.0
    )
    water_fraction = (
        float(core_bundle["water_fraction"])
        if core_bundle.get("water_fraction") is not None
        else 0.0
    )
    centroid_coords = core_bundle.get("centroid") or [LILONGWE_LON, LILONGWE_LAT]
    centroid = [float(centroid_coords[1]), float(centroid_coords[0])]

    climate_radar = compute_rainfall_and_radar_signals(
        aoi=analysis_aoi,
        lookback_days=lookback_days,
        end_dt=end_dt,
        area_ha=analysis_area_ha,
    )

    if crop_fraction >= MIN_CROPLAND_FRACTION:
        active_mask = primary_crop_mask
        active_fraction = crop_fraction
        land_cover_label = "Cropland (Class 40)"
        buffer_note = base_buffer_note
    elif dimba_fraction >= MIN_DIMBA_FRACTION and water_fraction < MAX_WATER_FRACTION:
        active_mask = dimba_wetland_mask
        active_fraction = dimba_fraction
        land_cover_label = "Dambo / Dimba Margin"
        buffer_note = f"{base_buffer_note} (Dambo / Dimba Margin)"
    else:
        return {
            "has_data": False,
            "scene_count": scene_count,
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_date": end_dt.strftime("%Y-%m-%d"),
            "area_ha": round(raw_area_ha, 2),
            "analyzed_area_ha": round(analysis_area_ha, 2),
            "buffer_note": base_buffer_note,
            "crop_fraction": round(crop_fraction, 3),
            "dimba_fraction": round(dimba_fraction, 3),
            "water_fraction": round(water_fraction, 3),
            "centroid": centroid,
            "reason": (
                f"Only {crop_fraction * 100:.1f}% of the selected area is ESA WorldCover "
                f"cropland (minimum {MIN_CROPLAND_FRACTION * 100:.0f}% required). Draw a polygon "
                "or select coordinates directly over farmland rather than buildings, forest, or water."
            ),
            **climate_radar,
        }

    if scene_count == 0:
        return {
            "has_data": False,
            "scene_count": 0,
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_date": end_dt.strftime("%Y-%m-%d"),
            "area_ha": round(raw_area_ha, 2),
            "analyzed_area_ha": round(analysis_area_ha, 2),
            "buffer_note": buffer_note,
            "land_cover_label": land_cover_label,
            "crop_fraction": round(active_fraction, 3),
            "dimba_fraction": round(dimba_fraction, 3),
            "water_fraction": round(water_fraction, 3),
            "centroid": centroid,
            **climate_radar,
        }

    masked_composite = collection.select(band_list).median().clip(analysis_aoi).updateMask(active_mask)
    stats = (
        masked_composite.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=analysis_aoi,
            scale=10,
            maxPixels=1_000_000_000,
            bestEffort=True,
        ).getInfo()
        or {}
    )
    mean_ndvi = stats.get("NDVI")
    mean_ndre = stats.get("NDRE")
    mean_savi = stats.get("SAVI")
    mean_ndwi = stats.get("NDWI")
    mean_ndwi_surface = stats.get("NDWI_SURFACE")
    mean_bsi = stats.get("BSI")

    if mean_ndvi is None:
        return {
            "has_data": False,
            "scene_count": scene_count,
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_date": end_dt.strftime("%Y-%m-%d"),
            "area_ha": round(raw_area_ha, 2),
            "analyzed_area_ha": round(analysis_area_ha, 2),
            "buffer_note": buffer_note,
            "land_cover_label": land_cover_label,
            "crop_fraction": round(active_fraction, 3),
            "dimba_fraction": round(dimba_fraction, 3),
            "water_fraction": round(water_fraction, 3),
            "centroid": centroid,
            "reason": f"All {land_cover_label.lower()} pixels inside the boundary were masked by clouds, shadows, or invalid flags.",
            **climate_radar,
        }

    # Unmasked median composite on analysis_aoi for visual map overlay
    map_composite = collection.select(["NDVI"]).median().clip(analysis_aoi)
    vis_params = {
        "min": 0.0,
        "max": 0.85,
        "palette": NDVI_PALETTE,
    }
    map_id_dict = map_composite.getMapId(vis_params)
    ndvi_tile_url = map_id_dict["tile_fetcher"].url_format

    # Per-scene time series masked to active_mask inside analysis_aoi
    def summarize_scene(img: ee.Image) -> ee.Feature:
        vals = (
            img.select(["NDVI", "SAVI", "NDWI"])
            .updateMask(active_mask)
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=analysis_aoi,
                scale=20,
                maxPixels=1_000_000_000,
                bestEffort=True,
            )
        )
        return ee.Feature(
            None,
            {
                "date": img.date().format("YYYY-MM-dd"),
                "NDVI": vals.get("NDVI"),
                "SAVI": vals.get("SAVI"),
                "NDWI": vals.get("NDWI"),
            },
        )

    ts_features = collection.map(summarize_scene).getInfo().get("features", [])
    ts_rows = [
        {
            "Date": f["properties"]["date"],
            "NDVI": round(float(f["properties"]["NDVI"]), 3),
            "SAVI": round(float(f["properties"]["SAVI"]), 3)
            if f["properties"].get("SAVI") is not None
            else None,
            "NDWI": round(float(f["properties"]["NDWI"]), 3)
            if f["properties"].get("NDWI") is not None
            else None,
        }
        for f in ts_features
        if f.get("properties", {}).get("NDVI") is not None
    ]
    ts_df = (
        pd.DataFrame(ts_rows).groupby("Date", as_index=False).mean().sort_values("Date")
        if ts_rows
        else pd.DataFrame(columns=["Date", "NDVI", "SAVI", "NDWI"])
    )

    return {
        "has_data": True,
        "scene_count": scene_count,
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"),
        "observation_month": end_dt.month,
        "area_ha": round(raw_area_ha, 2),
        "analyzed_area_ha": round(analysis_area_ha, 2),
        "buffer_note": buffer_note,
        "land_cover_label": land_cover_label,
        "crop_fraction": round(active_fraction, 3),
        "dimba_fraction": round(dimba_fraction, 3),
        "water_fraction": round(water_fraction, 3),
        "mean_ndvi": float(mean_ndvi),
        "mean_ndre": float(mean_ndre) if mean_ndre is not None else 0.0,
        "mean_savi": float(mean_savi) if mean_savi is not None else 0.0,
        "mean_ndwi": float(mean_ndwi) if mean_ndwi is not None else 0.0,
        "mean_ndwi_surface": float(mean_ndwi_surface) if mean_ndwi_surface is not None else 0.0,
        "mean_bsi": float(mean_bsi) if mean_bsi is not None else 0.0,
        "ndvi_tile_url": ndvi_tile_url,
        "centroid": centroid,
        "timeseries": ts_df,
        **climate_radar,
    }


# ---------------------------------------------------------------------------
# 3. Drought Early Warning, Agronomic Status & Chichewa SMS Generator
# ---------------------------------------------------------------------------
def compute_drought_risk(
    rain_anomaly_pct: float,
    dry_days: int,
    lookback_days: int,
    mean_ndwi: float,
    mean_ndvi: float,
    stress_threshold: float,
    climate_status: str = "",
) -> dict[str, Any]:
    """Compute a 0-100 drought early-warning score for food security planning."""
    score = 0.0

    if rain_anomaly_pct <= -40:
        score += 40.0
    elif rain_anomaly_pct < 0:
        score += min(40.0, abs(rain_anomaly_pct) * 1.0)

    dry_ratio = dry_days / max(1, lookback_days)
    score += min(30.0, dry_ratio * 40.0)

    if mean_ndwi < -0.10:
        score += 15.0
    elif mean_ndwi < 0.0:
        score += 8.0

    if mean_ndvi < stress_threshold:
        score += 15.0
    elif mean_ndvi < stress_threshold + 0.12:
        score += 8.0

    risk_score = int(round(min(100.0, max(0.0, score))))

    status_lower = climate_status.lower()
    if "not yet published" in status_lower or "unavailable" in status_lower:
        level = "Rain Data Pending"
        food_security_note = (
            "Recent satellite rainfall scenes are still ingesting — verify local rain gauge and "
            "ridge moisture before applying fertilizer."
        )
    elif risk_score >= 65:
        level = "High Drought Risk"
        food_security_note = (
            "Elevated yield-loss risk for local food stocks. Prioritize extension visits, "
            "moisture conservation (ridging/mulching), and early food/input planning."
        )
    elif risk_score >= 35:
        level = "Moderate Drought Risk"
        food_security_note = (
            "Emerging rainfall or moisture deficit. Monitor soil/canopy moisture, control "
            "weeds early, and time fertilizer application carefully."
        )
    else:
        level = "Low Drought Risk"
        food_security_note = (
            "Rainfall and moisture indicators are near or above normal baseline. Maintain "
            "standard crop management and routine scouting."
        )

    return {
        "risk_score": risk_score,
        "level": level,
        "food_security_note": food_security_note,
    }


def interpret_crop_health(
    mean_ndvi: float,
    mean_savi: float,
    mean_ndre: float,
    mean_ndwi: float,
    mean_bsi: float,
    rain_anomaly_pct: float,
    dry_days: int,
    lookback_days: int,
    area_ha: float,
    crop_type: str,
    farmer_name: str,
    expected_value_per_ha_mwk: int,
    observation_month: int,
    climate_status: str = "",
    land_cover_label: str = "Cropland (Class 40)",
) -> dict[str, Any]:
    """
    Translate spectral indices, Malawi crop calendar stage, and rainfall/moisture
    signals into consistent crop health, drought risk, MWK exposure, and Chichewa SMS.
    """
    name_prefix = f"Bambo/Mai {farmer_name.strip()}, " if farmer_name.strip() else ""
    total_farm_value_mwk = area_ha * expected_value_per_ha_mwk

    season = get_malawi_season_stage(observation_month)
    is_dimba = "dimba" in land_cover_label.lower()
    stage_label = (
        f"{season['stage']} (Dambo / Dimba Margin)" if is_dimba else season["stage"]
    )
    healthy_thresh = float(season["healthy_ndvi"])
    stress_thresh = float(season["stress_ndvi"])
    early_emergence_band = bool(season["early_emergence_band"])
    default_retention = float(season["default_retention"])

    moisture_deficit = (rain_anomaly_pct <= -20.0) or (mean_ndwi < -0.10)
    likely_bare_soil = mean_ndvi < stress_thresh and mean_savi < 0.22 and mean_bsi > 0.02

    drought = compute_drought_risk(
        rain_anomaly_pct=rain_anomaly_pct,
        dry_days=dry_days,
        lookback_days=lookback_days,
        mean_ndwi=mean_ndwi,
        mean_ndvi=mean_ndvi,
        stress_threshold=stress_thresh,
        climate_status=climate_status,
    )

    is_nov_dec_pre_emergence = (
        observation_month in (11, 12)
        and (mean_ndvi < 0.30)
        and (mean_bsi >= 0.15)
        and (rain_anomaly_pct > -25.0)
    )

    is_early_emergence = (
        early_emergence_band
        and (stress_thresh <= mean_ndvi < healthy_thresh)
        and (rain_anomaly_pct > -20.0)
        and (mean_ndwi >= -0.10)
    )

    if is_nov_dec_pre_emergence:
        status = "Pre-Emergence / Bare Soil (Planting Stage)"
        severity_color = "normal"
        summary = (
            f"Bare-soil / early-emergence signal (NDVI {mean_ndvi:.2f}, BSI {mean_bsi:.2f}) "
            f"across {area_ha:.2f} ha of {crop_type} during {stage_label} indicates newly prepared "
            "ridges or recently planted seed awaiting full emergence."
        )
        action = (
            "Plant promptly with effective rains where ridges are ready, verify seed depth/emergence "
            "on the ground, and keep early weeds down as seedlings emerge."
        )
        estimated_yield_retention = 0.95
        sms_chichewa = (
            f"AGROSAT LILONGWE: {name_prefix}munda wanu wa {crop_type} uli pa nthawi yokonzekera "
            f"kapena kubzalira (NDVI {mean_ndvi:.2f}). Bzalani mbeu pa mvula yokwanira ndipo "
            "yang'anani kumera kwa mbewu m'munda mwanu."
        )
    elif is_early_emergence:
        status = "Healthy (Early Emergence)"
        severity_color = "normal"
        summary = (
            f"Normal early emergence across {area_ha:.2f} ha of {crop_type} during {stage_label} "
            f"(NDVI {mean_ndvi:.2f}, SAVI {mean_savi:.2f}, NDWI {mean_ndwi:.2f})."
        )
        action = (
            "Keep fields free of early weeds and continue normal planting/emergence scouting."
        )
        estimated_yield_retention = 0.92
        sms_chichewa = (
            f"AGROSAT LILONGWE: {name_prefix}munda wanu wa {crop_type} ukuyamba bwino "
            f"(NDVI {mean_ndvi:.2f}). Pitirizani kupalira udzu msanga ndipo musadere nkhawa."
        )
    elif mean_ndvi >= healthy_thresh:
        status = "Healthy"
        severity_color = "normal"
        ndre_note = (
            "Red-edge chlorophyll (NDRE) is strong."
            if mean_ndre >= 0.22
            else "Monitor red-edge/nitrogen status as canopy progresses."
        )
        summary = (
            f"Healthy canopy across {area_ha:.2f} ha of {crop_type} ({stage_label}; "
            f"NDVI {mean_ndvi:.2f}, SAVI {mean_savi:.2f}). {ndre_note}"
        )
        action = (
            "Maintain routine weekly field scouting. Avoid unnecessary top-dressing "
            "where canopy and red-edge readings are already strong."
        )
        estimated_yield_retention = default_retention
        sms_chichewa = (
            f"AGROSAT LILONGWE: {name_prefix}munda wanu wa {crop_type} uli bwino kwambiri "
            f"(NDVI {mean_ndvi:.2f}). Mbeu zikukula mokwanira. Pitirizani kuyang'anira munda "
            "mwachizolowezi; musawononge ndalama pogula feteleza wowonjezera."
        )
    elif mean_ndvi >= stress_thresh:
        status = "Moderate Stress"
        severity_color = "off"
        if moisture_deficit:
            moisture_note = (
                "Rainfall or canopy moisture deficit is driving this stress."
            )
            action = (
                "Conserve soil moisture through tie-ridging or mulching, keep weeds down, "
                "and avoid applying fertilizer while soil is dry."
            )
            sms_chichewa = (
                f"AGROSAT LILONGWE: Chenjezo! {name_prefix}munda wanu wa {crop_type} ukuonetsa "
                f"kuchepa kwa chinyezi kapena mvula (NDVI {mean_ndvi:.2f}). Sungani chinyezi "
                "m'nthaka ndi kupalira udzu; musathire feteleza nthaka ikakhala youma."
            )
        else:
            moisture_note = (
                "Rainfall and moisture are near normal; stress likely reflects weeding needs, "
                "Fall Armyworm/pest pressure, or localized nutrient variation."
            )
            action = (
                "Scout for Fall Armyworm and weeds within 5 days, and apply targeted "
                "fertilizer only where moisture is adequate."
            )
            sms_chichewa = (
                f"AGROSAT LILONGWE: Chenjezo! {name_prefix}munda wanu wa {crop_type} ukuonetsa "
                f"kuchepa kwa mphamvu (NDVI {mean_ndvi:.2f}). Palirani udzu, fufuzani mbozi "
                "kapena tizirombo, ndipo thirani feteleza moyenera pamalo achinyezi."
            )
        summary = (
            f"Moderate canopy vigor across {area_ha:.2f} ha of {crop_type} ({stage_label}; "
            f"NDVI {mean_ndvi:.2f}, SAVI {mean_savi:.2f}, NDRE {mean_ndre:.2f}). {moisture_note}"
        )
        estimated_yield_retention = 0.72
    else:
        status = "Severe Stress"
        severity_color = "inverse"
        if moisture_deficit:
            context_note = (
                "Rainfall/moisture deficit is severe—check soil dryness and crop survival."
            )
            action = (
                "Prioritize moisture conservation (mulching/tie-ridging), consult the nearest "
                "Lilongwe EPA extension officer, and do not apply fertilizer on dry soil."
            )
            sms_chichewa = (
                f"AGROSAT LILONGWE: CHENJEZO LALIKULU! {name_prefix}munda wanu wa {crop_type} uli pa "
                f"vuto la chilala kapena kuuma (NDVI {mean_ndvi:.2f}). Sungani chinyezi, lankhulani "
                "ndi mlangizi wa zaulimi (EPA), ndipo musathire feteleza nthaka ikakhala youma."
            )
        else:
            soil_hint = (
                "Elevated bare-soil index (BSI) suggests exposed soil/tillage; "
                if likely_bare_soil
                else ""
            )
            context_note = (
                f"{soil_hint}Rainfall is near normal, so verify Fall Armyworm/pest damage, "
                "stand establishment, or planting timing on the ground."
            )
            action = (
                "Verify on the ground with a Lilongwe EPA extension officer or agronomist, "
                "scout for Fall Armyworm/pests, and document damage."
            )
            sms_chichewa = (
                f"AGROSAT LILONGWE: CHENJEZO LALIKULU! {name_prefix}munda wanu wa {crop_type} uli pa "
                f"vuto lalikulu (NDVI {mean_ndvi:.2f}). Fufuzani mbozi kapena tizirombo lero lomwe "
                "ndipo lankhulani ndi mlangizi wa zaulimi (EPA)."
            )
        summary = (
            f"Low vegetation index across {area_ha:.2f} ha of {crop_type} ({stage_label}; "
            f"NDVI {mean_ndvi:.2f}, SAVI {mean_savi:.2f}). {context_note}"
        )
        estimated_yield_retention = 0.35

    at_risk_mwk = max(0.0, total_farm_value_mwk * (1.0 - estimated_yield_retention))

    return {
        "status": status,
        "season_stage": stage_label,
        "moisture_deficit": moisture_deficit,
        "severity_color": severity_color,
        "summary": summary,
        "action": action,
        "sms_chichewa": sms_chichewa,
        "total_value_mwk": total_farm_value_mwk,
        "at_risk_mwk": at_risk_mwk,
        "yield_retention_pct": int(estimated_yield_retention * 100),
        "drought_score": drought["risk_score"],
        "drought_level": drought["level"],
        "food_security_note": drought["food_security_note"],
    }


# ---------------------------------------------------------------------------
# 4. Interactive Map Builder (Folium + Drawing Tools + NDVI Tile Layer)
# ---------------------------------------------------------------------------
def build_map(
    center: list[float],
    zoom: int,
    ndvi_tile_url: str | None = None,
    farm_geojson: dict[str, Any] | None = None,
) -> folium.Map:
    """Create a Folium map centered on Lilongwe with realistic imagery & NDVI overlay."""
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        control_scale=True,
        tiles=None,
    )

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite Imagery (Esri)",
        overlay=False,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google Hybrid",
        name="Google Satellite + Labels",
        overlay=False,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=True,
    ).add_to(m)

    if farm_geojson:
        folium.GeoJson(
            data={"type": "Feature", "geometry": farm_geojson, "properties": {}},
            name="Target Farm Boundary",
            style_function=lambda _: {
                "color": "#ffffff",
                "weight": 2.5,
                "fillOpacity": 0.0,
                "dashArray": "5, 5",
            },
        ).add_to(m)

    if ndvi_tile_url:
        folium.TileLayer(
            tiles=ndvi_tile_url,
            attr="Google Earth Engine / Copernicus Sentinel-2",
            name="Sentinel-2 NDVI (Median Composite)",
            overlay=True,
            control=True,
            opacity=0.85,
        ).add_to(m)

        colormap = cm.LinearColormap(
            colors=NDVI_PALETTE,
            vmin=0.0,
            vmax=0.85,
            caption="Median NDVI (0.0 = Bare/Severe Stress → 0.85 = Dense Healthy Canopy)",
        )
        colormap.add_to(m)

    Draw(
        export=False,
        position="topleft",
        draw_options={
            "polyline": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "polygon": {
                "allowIntersection": False,
                "shapeOptions": {"color": "#16a34a", "weight": 2},
            },
            "rectangle": {
                "shapeOptions": {"color": "#16a34a", "weight": 2},
            },
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)
    return m


# ---------------------------------------------------------------------------
# 5. Streamlit UI Layout
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("🛰️ AgroSat Lilongwe")
    st.caption(
        "Drought Early Warning, Crop Monitoring & Food Security Advisory · "
        "Lilongwe District, Malawi (-13.9626, 33.7741)"
    )

    if "analysis_result" not in st.session_state:
        st.session_state["analysis_result"] = None
    if "drawn_geojson_key" not in st.session_state:
        st.session_state["drawn_geojson_key"] = None
    if "drawn_geojson" not in st.session_state:
        st.session_state["drawn_geojson"] = None

    with st.sidebar:
        st.header("Satellite & Advisory Settings")

        gee_ok, gee_msg = initialize_gee()
        if gee_ok:
            st.success(f"Earth Engine Connected\n\n{gee_msg}")
        else:
            st.error(gee_msg)
            st.info(
                "Create `.streamlit/secrets.toml` with:\n\n"
                "```toml\n[GEE_SERVICE_ACCOUNT]\ntype = \"service_account\"\n"
                "project_id = \"your-gcp-project\"\nclient_email = \"...\"\n"
                "private_key = \"-----BEGIN PRIVATE KEY-----\\n...\"\n```"
            )
            st.stop()

        st.divider()
        lookback_days = st.slider(
            "Analysis Window (Days)",
            min_value=14,
            max_value=60,
            value=LOOKBACK_DAYS,
            step=5,
        )
        max_cloud_pct = st.slider(
            "Max Sentinel-2 Cloud Cover (%)",
            min_value=10,
            max_value=80,
            value=MAX_CLOUD_PERCENT,
            step=10,
        )

        st.divider()
        st.subheader("Farm & Economic Context")
        crop_type = st.selectbox(
            "Primary Crop",
            options=["Chimanga (Maize)", "Mtedza (Groundnuts)", "Soya (Soybeans)", "Fodya (Tobacco)"],
        )
        farmer_name = st.text_input("Farmer / Cooperative Name (for SMS)", value="Chisomo Banda")
        farmer_phone = st.text_input("Recipient Phone (Malawi)", value="+265 999 123 456")
        value_per_ha_mwk = st.number_input(
            "Expected Harvest Value (MWK / ha)",
            min_value=100_000,
            max_value=10_000_000,
            value=1_450_000,
            step=50_000,
            help="Used to estimate revenue at risk when crop stress is detected.",
        )

        if st.button("Reset Drawn Farm", use_container_width=True):
            st.session_state["analysis_result"] = None
            st.session_state["drawn_geojson"] = None
            st.session_state["drawn_geojson_key"] = None
            st.rerun()

    col_map, col_insights = st.columns([1.3, 1.0], gap="large")

    with col_map:
        st.subheader("1. Select by Map or Coordinates (Lilongwe / Bunda)")
        st.markdown(
            "Draw a **Polygon/Rectangle** on the map **or** enter GPS **Latitude/Longitude** "
            "below to extract live Sentinel-2, Sentinel-1, and CHIRPS data."
        )

        with st.expander("Enter GPS Coordinates (Lat / Lon + Buffer)", expanded=True):
            c1, c2, c3 = st.columns([1.1, 1.1, 0.9])
            lat_input = c1.number_input(
                "Latitude",
                min_value=-17.5,
                max_value=-9.0,
                value=-14.1811,
                step=0.001,
                format="%.4f",
            )
            lon_input = c2.number_input(
                "Longitude",
                min_value=32.5,
                max_value=36.0,
                value=33.7686,
                step=0.001,
                format="%.4f",
            )
            half_width_m = c3.number_input(
                "Half-Box (m)",
                min_value=50,
                max_value=2000,
                value=100,
                step=50,
                help="Creates a square sample box around the coordinate (100m -> ~4 ha).",
            )
            if st.button("Extract Data for Coordinates", use_container_width=True):
                coord_geom = coords_to_polygon_geojson(
                    lat=float(lat_input),
                    lon=float(lon_input),
                    radius_m=float(half_width_m),
                )
                coord_key = f"coord:{lat_input:.5f},{lon_input:.5f},{half_width_m}"
                st.session_state["drawn_geojson"] = coord_geom
                st.session_state["drawn_geojson_key"] = coord_key
                with st.spinner("Extracting live GEE data for GPS coordinates..."):
                    st.session_state["analysis_result"] = run_farm_analysis(
                        geojson_geom=coord_geom,
                        lookback_days=lookback_days,
                        max_cloud_pct=max_cloud_pct,
                    )
                st.rerun()

        result = st.session_state.get("analysis_result")
        map_center = (
            result["centroid"]
            if result and result.get("centroid")
            else [LILONGWE_LAT, LILONGWE_LON]
        )
        map_zoom = 15 if (result and result.get("has_data")) else DEFAULT_ZOOM
        ndvi_tile_url = result.get("ndvi_tile_url") if result and result.get("has_data") else None

        fmap = build_map(
            center=map_center,
            zoom=map_zoom,
            ndvi_tile_url=ndvi_tile_url,
            farm_geojson=st.session_state.get("drawn_geojson"),
        )
        map_output = st_folium(
            fmap,
            height=500,
            use_container_width=True,
            key="agrosat_lilongwe_map",
            returned_objects=["last_active_drawing", "all_drawings"],
        )

        new_geom = extract_polygon_geometry(map_output)
        new_geom_key = json.dumps(new_geom, sort_keys=True) if new_geom else None
        if new_geom and new_geom_key != st.session_state.get("drawn_geojson_key"):
            st.session_state["drawn_geojson"] = new_geom
            st.session_state["drawn_geojson_key"] = new_geom_key
            with st.spinner("Querying Sentinel-2, Sentinel-1 & CHIRPS rainfall data..."):
                st.session_state["analysis_result"] = run_farm_analysis(
                    geojson_geom=new_geom,
                    lookback_days=lookback_days,
                    max_cloud_pct=max_cloud_pct,
                )
            st.rerun()

    with col_insights:
        st.subheader("2. Drought Risk, Crop Vigor & Advisory")
        result = st.session_state.get("analysis_result")

        if not result:
            st.info(
                "Draw a polygon on the map or enter GPS coordinates to compute Sentinel-2 "
                "vegetation/moisture indices, CHIRPS rainfall anomaly & dry days, Sentinel-1 "
                "radar, drought risk, and a Chichewa SMS alert."
            )
            st.markdown(
                "**System Signals for Food Security:**\n"
                "- **Rainfall & Dry Days (CHIRPS):** lag-adjusted recent rain vs. 10-yr baseline\n"
                "- **Cropland & Dimba Mask (ESA WorldCover v200):** filters stats to cropland (Class 40) or dimba margins (Class 90)\n"
                "- **Season-Aware Vigor & Moisture:** adjusts thresholds by Malawi crop calendar stage"
            )
        elif not result.get("has_data"):
            reason_line = f"{result['reason']} " if result.get("reason") else ""
            st.warning(
                f"{reason_line}"
                f"Window checked: **{result['start_date']}** to **{result['end_date']}** "
                f"(Sentinel-2 scenes: {result['scene_count']}, cropland share: "
                f"{result.get('crop_fraction', 0.0) * 100:.1f}%). "
                f"CHIRPS rain: **{result.get('rain_mm', 0.0):.1f} mm** "
                f"({result.get('rain_anomaly_pct', 0.0):+.1f}% vs 10-yr norm), "
                f"dry days: **{result.get('dry_days', 0)}/{lookback_days}**, "
                f"Sentinel-1 VV: **{result.get('s1_vv_db') if result.get('s1_vv_db') is not None else 'N/A'} dB** "
                f"({result.get('climate_status', '')})."
            )
        else:
            land_cover_label = result.get("land_cover_label", "Cropland (Class 40)")
            insights = interpret_crop_health(
                mean_ndvi=result["mean_ndvi"],
                mean_savi=result["mean_savi"],
                mean_ndre=result["mean_ndre"],
                mean_ndwi=result["mean_ndwi"],
                mean_bsi=result["mean_bsi"],
                rain_anomaly_pct=result["rain_anomaly_pct"],
                dry_days=result["dry_days"],
                lookback_days=lookback_days,
                area_ha=result["area_ha"],
                crop_type=crop_type,
                farmer_name=farmer_name,
                expected_value_per_ha_mwk=int(value_per_ha_mwk),
                observation_month=int(result.get("observation_month") or datetime.now(timezone.utc).month),
                climate_status=result.get("climate_status", ""),
                land_cover_label=land_cover_label,
            )

            m1, m2 = st.columns(2)
            m1.metric(
                "Drought Risk",
                f"{insights['drought_score']}/100",
                delta=insights["drought_level"],
                delta_color="inverse" if insights["drought_score"] >= 35 else "normal",
            )
            m2.metric(
                "Rain Anomaly",
                f"{result['rain_anomaly_pct']:+.1f}%",
                delta=f"{result['rain_mm']:.0f} vs {result['hist_rain_mm']:.0f} mm",
                delta_color="off",
            )

            m3, m4 = st.columns(2)
            m3.metric(
                "Median NDVI",
                f"{result['mean_ndvi']:.2f}",
                delta=insights["status"],
                delta_color=insights["severity_color"],
            )
            m4.metric(
                "Median SAVI",
                f"{result['mean_savi']:.2f}",
                delta=f"{result['dry_days']} dry d · {format_area_compact(result['area_ha'])}",
                delta_color="off",
            )

            s1_label = f"{result['s1_vv_db']:.2f} dB" if result.get("s1_vv_db") is not None else "N/A"
            st.caption(
                f"Stage: {insights['season_stage']} · {land_cover_label}: {result.get('crop_fraction', 0.0) * 100:.0f}% · "
                f"{result.get('climate_status', '')} · NDRE: {result['mean_ndre']:.2f} · "
                f"NDWI: {result['mean_ndwi']:.2f} · BSI: {result['mean_bsi']:.2f} · "
                f"S1 VV: {s1_label} · {result['buffer_note']}"
            )

            if insights["severity_color"] == "normal":
                st.success(f"**Crop Health: {insights['status']}** — {insights['summary']}")
            elif insights["severity_color"] == "off":
                st.warning(f"**Crop Health: {insights['status']}** — {insights['summary']}")
            else:
                st.error(f"**Crop Health: {insights['status']}** — {insights['summary']}")

            st.info(f"**Food Security Note ({insights['drought_level']}):** {insights['food_security_note']}")
            st.markdown(f"**Recommended Action:** {insights['action']}")

            e1, e2 = st.columns(2)
            e1.metric(
                "Est. Gross Value",
                format_mwk_compact(insights["total_value_mwk"]),
                help=f"Exact: MWK {insights['total_value_mwk']:,.0f}",
            )
            e2.metric(
                "Revenue at Risk",
                format_mwk_compact(insights["at_risk_mwk"]),
                delta=f"{insights['yield_retention_pct']}% yield retention",
                delta_color="off",
                help=f"Exact: MWK {insights['at_risk_mwk']:,.0f}",
            )

            st.divider()
            st.subheader("3. Chichewa SMS Alert Preview")
            st.caption(f"To: {farmer_phone} · Length: {len(insights['sms_chichewa'])} chars")
            centroid = result.get("centroid") or [LILONGWE_LAT, LILONGWE_LON]
            sms_key = f"sms_preview_{centroid[0]:.4f}_{centroid[1]:.4f}_{result['mean_ndvi']:.2f}"
            st.text_area(
                "Generated SMS Message (Editable before dispatch)",
                value=insights["sms_chichewa"],
                height=115,
                key=sms_key,
            )

            ts_df = result.get("timeseries")
            if isinstance(ts_df, pd.DataFrame) and not ts_df.empty:
                st.divider()
                st.subheader("Scene Trend (Lookback Window)")
                chart_cols = [c for c in ["NDVI", "SAVI", "NDWI"] if c in ts_df.columns]
                st.line_chart(ts_df.set_index("Date")[chart_cols], height=200)


if __name__ == "__main__":
    main()
