"""
AgroSat Lilongwe — Agricultural Satellite Monitoring Platform
Single-file Streamlit application for smallholder & commercial farm monitoring
in Lilongwe, Malawi using Google Earth Engine (Sentinel-2 SR Harmonized).
"""

from __future__ import annotations

import json
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
LOOKBACK_DAYS = 30
MAX_CLOUD_PERCENT = 50
MIN_BUFFER_PLOT_AREA_M2 = 4000.0
MIN_BUFFERED_REMAINING_M2 = 900.0
INWARD_BUFFER_METERS = -7.5

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
    page_title="AgroSat Lilongwe | Satellite Crop Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


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

        # Normalize escaped newlines in private keys if pasted into TOML
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

    # SCL classes: 1=Defective/Saturated, 3=Cloud Shadow, 8=Cloud Med Prob,
    # 9=Cloud High Prob, 10=Thin Cirrus
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
    tailored for Lilongwe / Central Region agricultural landscapes in Malawi:
      - NDVI: general canopy vigor (B8, B4)
      - NDRE: red-edge chlorophyll/nitrogen sensitivity (B8A, B5)
      - SAVI (L=0.5): reduces reddish/ferralitic & sandy-loam soil background bias
      - NDWI (Gao 1996, B8/B11): plant canopy water status
      - NDWI_SURFACE (McFeeters 1996, B3/B8): standing water / localized waterlogging
      - BSI: Bare Soil Index to flag exposed soil vs. crop canopy
    """
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")

    # Red-Edge NDVI (20m B8A and B5) — sensitive to chlorophyll/nitrogen before B4 saturates
    ndre = image.normalizedDifference(["B8A", "B5"]).rename("NDRE")

    # Soil-Adjusted Vegetation Index (Huete L = 0.5) for partial canopy & Lilongwe soils
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

    # Canopy moisture (NIR vs SWIR1) and surface water (Green vs NIR)
    ndwi_canopy = image.normalizedDifference(["B8", "B11"]).rename("NDWI")
    ndwi_surface = image.normalizedDifference(["B3", "B8"]).rename("NDWI_SURFACE")

    # Bare Soil Index: higher values indicate exposed/dry soil; lower values indicate vegetation
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


def get_analysis_geometry(geojson_geom: dict[str, Any]) -> tuple[ee.Geometry, str]:
    """
    Safely apply a small inward buffer (-7.5m, less than one 10m Sentinel-2 pixel)
    to reduce field-edge/path/bund mixing on larger fields while preserving
    smallholder plots so they don't collapse or lose valid pixels.
    """
    raw_geom = ee.Geometry(geojson_geom)
    area_m2 = float(raw_geom.area(maxError=1).getInfo() or 0.0)

    if area_m2 >= MIN_BUFFER_PLOT_AREA_M2:
        buffered = raw_geom.buffer(distance=INWARD_BUFFER_METERS, maxError=1)
        buffered_area = float(buffered.area(maxError=1).getInfo() or 0.0)
        if buffered_area >= MIN_BUFFERED_REMAINING_M2:
            return buffered, f"Inward {INWARD_BUFFER_METERS:.1f}m edge buffer applied"

    return raw_geom, "Full drawn boundary used (plot compact)"


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
    """Run GEE Sentinel-2 NDVI/NDRE/SAVI/NDWI/BSI analysis for the drawn farm polygon."""
    raw_aoi = ee.Geometry(geojson_geom)
    raw_area_ha = float(raw_aoi.area(maxError=1).divide(10_000).getInfo() or 0.0)
    analysis_aoi, buffer_note = get_analysis_geometry(geojson_geom)
    analysis_area_ha = float(analysis_aoi.area(maxError=1).divide(10_000).getInfo() or raw_area_ha)

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=lookback_days)

    collection = (
        ee.ImageCollection(S2_COLLECTION_ID)
        .filterBounds(analysis_aoi)
        .filterDate(
            start_dt.strftime("%Y-%m-%d"),
            (end_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .map(mask_malawi_s2_clouds)
        .map(add_enhanced_indices)
    )

    scene_count = int(collection.size().getInfo())
    if scene_count == 0:
        return {
            "has_data": False,
            "scene_count": 0,
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_date": end_dt.strftime("%Y-%m-%d"),
            "buffer_note": buffer_note,
        }

    # Median composite resists residual cloud/shadow outliers better than raw mean
    band_list = ["NDVI", "NDRE", "SAVI", "NDWI", "NDWI_SURFACE", "BSI"]
    composite = collection.select(band_list).median().clip(analysis_aoi)

    stats = composite.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=analysis_aoi,
        scale=10,
        maxPixels=1_000_000_000,
        bestEffort=True,
    ).getInfo()

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
            "buffer_note": buffer_note,
            "reason": "All pixels inside the farm boundary were masked by clouds, shadows, or invalid flags.",
        }

    # Generate Earth Engine map tile URL for the NDVI color layer
    vis_params = {
        "min": 0.0,
        "max": 0.85,
        "palette": NDVI_PALETTE,
    }
    map_id_dict = composite.select("NDVI").getMapId(vis_params)
    ndvi_tile_url = map_id_dict["tile_fetcher"].url_format

    # Per-scene time series inside the analyzed geometry
    def summarize_scene(img: ee.Image) -> ee.Feature:
        vals = img.select(["NDVI", "SAVI", "NDWI"]).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=analysis_aoi,
            scale=20,
            maxPixels=1_000_000_000,
            bestEffort=True,
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

    centroid_coords = raw_aoi.centroid(maxError=1).coordinates().getInfo()

    return {
        "has_data": True,
        "scene_count": scene_count,
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"),
        "area_ha": round(raw_area_ha, 2),
        "analyzed_area_ha": round(analysis_area_ha, 2),
        "buffer_note": buffer_note,
        "mean_ndvi": float(mean_ndvi),
        "mean_ndre": float(mean_ndre) if mean_ndre is not None else 0.0,
        "mean_savi": float(mean_savi) if mean_savi is not None else 0.0,
        "mean_ndwi": float(mean_ndwi) if mean_ndwi is not None else 0.0,
        "mean_ndwi_surface": float(mean_ndwi_surface) if mean_ndwi_surface is not None else 0.0,
        "mean_bsi": float(mean_bsi) if mean_bsi is not None else 0.0,
        "ndvi_tile_url": ndvi_tile_url,
        "centroid": [centroid_coords[1], centroid_coords[0]],
        "timeseries": ts_df,
    }


# ---------------------------------------------------------------------------
# 3. Agronomic Status, Economic Impact & Chichewa SMS Generator
# ---------------------------------------------------------------------------
def interpret_crop_health(
    mean_ndvi: float,
    mean_savi: float,
    mean_ndre: float,
    mean_ndwi: float,
    mean_bsi: float,
    area_ha: float,
    crop_type: str,
    farmer_name: str,
    expected_value_per_ha_mwk: int,
) -> dict[str, Any]:
    """
    Translate NDVI, SAVI, NDRE, NDWI, and BSI into plain-English crop health,
    geoscience context, MWK exposure, and Chichewa SMS advisory.
    """
    name_prefix = f"Bambo/Mai {farmer_name.strip()}, " if farmer_name.strip() else ""
    total_farm_value_mwk = area_ha * expected_value_per_ha_mwk

    # Exposed / bare soil heuristic: low canopy vigor + positive bare-soil index
    likely_bare_soil = mean_ndvi < 0.28 and mean_savi < 0.22 and mean_bsi > 0.02

    if mean_ndvi >= 0.55 and mean_savi >= 0.38:
        status = "Healthy"
        severity_color = "normal"
        ndre_note = (
            "Red-edge chlorophyll (NDRE) is strong."
            if mean_ndre >= 0.22
            else "Monitor red-edge/nitrogen status as canopy matures."
        )
        summary = (
            f"Vigorous canopy across {area_ha:.2f} ha of {crop_type} "
            f"(NDVI {mean_ndvi:.2f}, SAVI {mean_savi:.2f}). {ndre_note}"
        )
        action = (
            "Maintain routine weekly field scouting. Avoid unnecessary top-dressing "
            "where canopy and red-edge readings are already strong."
        )
        estimated_yield_retention = 0.95
        sms_chichewa = (
            f"AGROSAT LILONGWE: {name_prefix}munda wanu wa {crop_type} uli bwino kwambiri "
            f"(NDVI {mean_ndvi:.2f}). Mbeu zikukula mokwanira. Pitirizani kuyang'anira munda "
            "mwachizolowezi; musawononge ndalama pogula feteleza wowonjezera."
        )
    elif mean_ndvi >= 0.30 or (mean_ndvi >= 0.25 and not likely_bare_soil):
        status = "Moderate Stress"
        severity_color = "off"
        if mean_ndwi < 0.02:
            stress_hint = "Canopy moisture (NDWI) suggests mild water deficit."
        elif mean_ndre < 0.18:
            stress_hint = "Red-edge (NDRE) suggests possible nitrogen/chlorophyll stress."
        else:
            stress_hint = "Soil-adjusted vigor (SAVI) shows patchy or early-stage canopy."
        summary = (
            f"Moderate vigor across {area_ha:.2f} ha of {crop_type} "
            f"(NDVI {mean_ndvi:.2f}, SAVI {mean_savi:.2f}, NDRE {mean_ndre:.2f}). {stress_hint}"
        )
        action = (
            "Schedule an extension check within 5 days to inspect weeding, ridge moisture, "
            "and targeted fertilizer application on weaker zones."
        )
        estimated_yield_retention = 0.72
        sms_chichewa = (
            f"AGROSAT LILONGWE: Chenjezo! {name_prefix}munda wanu wa {crop_type} ukuonetsa "
            f"kuchepa kwa mphamvu (NDVI {mean_ndvi:.2f}). Chotsani udzu, yang'anani chinyezi "
            "m'nthaka ndipo thirani feteleza m'malo ofooka msanga."
        )
    else:
        status = "Severe Stress"
        severity_color = "inverse"
        soil_context = (
            "Bare/exposed soil signature (BSI) is elevated—confirm whether the field is "
            "pre-emergence/tilled or suffering severe crop loss."
            if likely_bare_soil
            else "Indicates severe drought stress, pest/Fall Armyworm damage, or bare ground."
        )
        summary = (
            f"Low vegetation index across {area_ha:.2f} ha of {crop_type} "
            f"(NDVI {mean_ndvi:.2f}, SAVI {mean_savi:.2f}). {soil_context}"
        )
        action = (
            "Verify on the ground with a Lilongwe EPA extension officer or agronomist, "
            "check crop stage vs. planting date, and document stress/damage."
        )
        estimated_yield_retention = 0.35
        sms_chichewa = (
            f"AGROSAT LILONGWE: CHENJEZO LALIKULU! {name_prefix}munda wanu wa {crop_type} uli pa "
            f"vuto lalikulu (NDVI {mean_ndvi:.2f}). Fufuzani mbozi kapena chilala lero lomwe "
            "ndipo lankhulani ndi mlangizi wa zaulimi (EPA)."
        )

    at_risk_mwk = max(0.0, total_farm_value_mwk * (1.0 - estimated_yield_retention))

    return {
        "status": status,
        "severity_color": severity_color,
        "summary": summary,
        "action": action,
        "sms_chichewa": sms_chichewa,
        "total_value_mwk": total_farm_value_mwk,
        "at_risk_mwk": at_risk_mwk,
        "yield_retention_pct": int(estimated_yield_retention * 100),
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
    """Create a Folium map centered on Lilongwe with polygon drawing & NDVI overlay."""
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
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=True,
    ).add_to(m)

    if farm_geojson:
        folium.GeoJson(
            data={"type": "Feature", "geometry": farm_geojson, "properties": {}},
            name="Drawn Farm Boundary",
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
        "Agricultural Satellite Monitoring & Smallholder Advisory Platform · "
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
            "Sentinel-2 Lookback Window (Days)",
            min_value=14,
            max_value=60,
            value=LOOKBACK_DAYS,
            step=5,
        )
        max_cloud_pct = st.slider(
            "Max Scene Cloud Cover (%)",
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
        st.subheader("1. Draw Farm Polygon (Lilongwe, Malawi)")
        st.markdown(
            "Use the **Polygon** or **Rectangle** tool on the top-left of the map to trace a "
            "farm boundary. Sentinel-2 (`COPERNICUS/S2_SR_HARMONIZED`) median indices are computed automatically."
        )

        result = st.session_state.get("analysis_result")
        map_center = (
            result["centroid"]
            if result and result.get("has_data") and result.get("centroid")
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
            height=540,
            use_container_width=True,
            key="agrosat_lilongwe_map",
            returned_objects=["last_active_drawing", "all_drawings"],
        )

        new_geom = extract_polygon_geometry(map_output)
        new_geom_key = json.dumps(new_geom, sort_keys=True) if new_geom else None
        if new_geom and new_geom_key != st.session_state.get("drawn_geojson_key"):
            st.session_state["drawn_geojson"] = new_geom
            st.session_state["drawn_geojson_key"] = new_geom_key
            with st.spinner("Querying Sentinel-2 & computing NDVI, SAVI, NDRE, NDWI, and BSI..."):
                st.session_state["analysis_result"] = run_farm_analysis(
                    geojson_geom=new_geom,
                    lookback_days=lookback_days,
                    max_cloud_pct=max_cloud_pct,
                )
            st.rerun()

    with col_insights:
        st.subheader("2. Satellite Analysis & Farmer Advisory")
        result = st.session_state.get("analysis_result")

        if not result:
            st.info(
                "Draw a polygon on the Lilongwe map to compute cloud-masked Sentinel-2 "
                "median NDVI, SAVI, NDRE, NDWI, and BSI, along with a Chichewa SMS alert."
            )
            st.markdown(
                "**Baseline Crop Health Guide:**\n"
                "- **Healthy:** `NDVI >= 0.55` and `SAVI >= 0.38` (vigorous canopy)\n"
                "- **Moderate Stress:** intermediate vigor or moisture/red-edge deficit\n"
                "- **Severe Stress:** low canopy vigor (checked against Bare Soil Index `BSI`)"
            )
        elif not result.get("has_data"):
            st.warning(
                f"No usable cloud-free Sentinel-2 pixels found between "
                f"**{result['start_date']}** and **{result['end_date']}** "
                f"(scenes checked: {result['scene_count']}). "
                "Try increasing the lookback window or maximum cloud cover in the sidebar."
            )
        else:
            insights = interpret_crop_health(
                mean_ndvi=result["mean_ndvi"],
                mean_savi=result["mean_savi"],
                mean_ndre=result["mean_ndre"],
                mean_ndwi=result["mean_ndwi"],
                mean_bsi=result["mean_bsi"],
                area_ha=result["area_ha"],
                crop_type=crop_type,
                farmer_name=farmer_name,
                expected_value_per_ha_mwk=int(value_per_ha_mwk),
            )

            # Top Metrics (2 per row so values/labels don't get cramped)
            m1, m2 = st.columns(2)
            m1.metric(
                "Median NDVI",
                f"{result['mean_ndvi']:.3f}",
                delta=insights["status"],
                delta_color=insights["severity_color"],
            )
            m2.metric(
                "Median SAVI (Soil-Adj.)",
                f"{result['mean_savi']:.3f}",
                help="Soil-Adjusted Vegetation Index (L=0.5) for Lilongwe soils",
            )

            m3, m4 = st.columns(2)
            m3.metric(
                "Median NDRE / NDWI",
                f"{result['mean_ndre']:.2f} / {result['mean_ndwi']:.2f}",
                help="NDRE (red-edge vigor) / NDWI (canopy moisture)",
            )
            m4.metric(
                "Plot Area",
                f"{result['area_ha']:.2f} ha",
                help=f"{result['buffer_note']} · {result['scene_count']} Sentinel-2 scenes",
            )

            st.caption(f"Sampling rule: {result['buffer_note']} · BSI: {result['mean_bsi']:.3f}")

            if insights["status"] == "Healthy":
                st.success(f"**Crop Health Status: {insights['status']}** — {insights['summary']}")
            elif insights["status"] == "Moderate Stress":
                st.warning(f"**Crop Health Status: {insights['status']}** — {insights['summary']}")
            else:
                st.error(f"**Crop Health Status: {insights['status']}** — {insights['summary']}")

            st.markdown(f"**Recommended Action:** {insights['action']}")

            e1, e2 = st.columns(2)
            e1.metric("Est. Gross Value", f"MWK {insights['total_value_mwk']:,.0f}")
            e2.metric(
                "Revenue at Risk",
                f"MWK {insights['at_risk_mwk']:,.0f}",
                delta=f"{insights['yield_retention_pct']}% yield retention est.",
                delta_color="off",
            )

            st.divider()
            st.subheader("3. Chichewa SMS Alert Preview")
            st.caption(f"To: {farmer_phone} · Length: {len(insights['sms_chichewa'])} chars")
            st.text_area(
                "Generated SMS Message (Editable before dispatch)",
                value=insights["sms_chichewa"],
                height=115,
                key="chichewa_sms_preview",
            )

            ts_df = result.get("timeseries")
            if isinstance(ts_df, pd.DataFrame) and not ts_df.empty:
                st.divider()
                st.subheader("Scene Trend (Lookback Window)")
                chart_cols = [c for c in ["NDVI", "SAVI", "NDWI"] if c in ts_df.columns]
                st.line_chart(ts_df.set_index("Date")[chart_cols], height=200)


if __name__ == "__main__":
    main()
