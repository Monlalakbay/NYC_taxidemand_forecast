# src/visualize/next_hour_map.py

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import datetime
from datetime import timezone

from src.utils.pipeline_logger import init_logger, log_pipeline_step

logger = init_logger(__name__)


# ---------------------------------------------------------------------
# Load latest evaluation summary
# ---------------------------------------------------------------------
def latest_eval_summary(base_dir: str) -> pd.DataFrame | None:
    """
    Load the most recent all_models_eval_test_*.csv from data/evaluation.
    This file contains per-district RMSE/MAE and the best_model column.
    """
    eval_dir = os.path.join(base_dir, "data", "evaluation")
    if not os.path.exists(eval_dir):
        logger.warning("Evaluation directory missing: %s", eval_dir)
        return None

    candidates = [
        f for f in os.listdir(eval_dir)
        if f.startswith("all_models_eval_test_") and f.endswith(".csv")
    ]
    if not candidates:
        logger.warning("No evaluation summary CSV found.")
        return None

    latest = sorted(candidates)[-1]
    path = os.path.join(eval_dir, latest)

    logger.info("Using evaluation summary: %s", latest)
    return pd.read_csv(path)


# ---------------------------------------------------------------------
# Helper: latest run folder for a given prefix
# ---------------------------------------------------------------------
def latest_run(base_dir: str, prefix: str, add_predictions: bool = True) -> str | None:
    """
    Returns the latest run folder under data/runs that starts with `prefix`.
    If add_predictions=True, returns the 'predictions' subfolder.
    """
    runs_dir = os.path.join(base_dir, "data", "runs")
    if not os.path.exists(runs_dir):
        return None

    runs = [d for d in os.listdir(runs_dir) if d.startswith(prefix)]
    if not runs:
        return None

    latest = sorted(runs)[-1]
    folder = os.path.join(runs_dir, latest)
    return os.path.join(folder, "predictions") if add_predictions else folder


# ---------------------------------------------------------------------
# Load prediction CSV for the best model of a district
# ---------------------------------------------------------------------
def load_pred_row(model_type: str, district: str, base_dir: str) -> pd.DataFrame | None:
    """
    Loads the last prediction row for a given district and model type,
    normalizes the prediction column name to 'pred_next_hour', and returns
    a tiny DataFrame with columns:
        ['district', 'model_type', 'pred_next_hour', 'y_true'].
    """
    # district string is already like "Manhattan CD 04"
    slug = district.replace(" ", "_")

    # Resolve file path based on model type
    if model_type == "rmse_rf":
        folder = latest_run(base_dir, "tree_preds_", add_predictions=True)
        path = os.path.join(folder, f"rf_preds_{slug}.csv") if folder else None

    elif model_type == "rmse_lgbm":
        folder = latest_run(base_dir, "tree_preds_", add_predictions=True)
        path = os.path.join(folder, f"lgbm_preds_{slug}.csv") if folder else None

    elif model_type == "rmse_conv":
        folder = latest_run(base_dir, "conv_lstm_", add_predictions=False)
        path = os.path.join(folder, f"conv_lstm_preds_{slug}.csv") if folder else None

    elif model_type == "rmse_hybrid":
        folder = latest_run(base_dir, "hybrid_", add_predictions=False)
        path = os.path.join(folder, f"hybrid_preds_{slug}.csv") if folder else None

    else:
        # Unknown model type, nothing to load
        return None

    if not path or not os.path.exists(path):
        logger.warning("Missing prediction file for district=%s (%s)", district, model_type)
        return None

    df = pd.read_csv(path)
    if df.empty:
        return None

    # Take the latest row (last prediction)
    row = df.tail(1).copy()
    row["district"] = district
    row["model_type"] = model_type.replace("rmse_", "").upper()

    # Normalize prediction column name
    if "y_pred_blend" in row.columns:
        row.rename(columns={"y_pred_blend": "pred_next_hour"}, inplace=True)
    elif "pred_ensemble" in row.columns:
        row.rename(columns={"pred_ensemble": "pred_next_hour"}, inplace=True)
    elif "y_pred" in row.columns:
        row.rename(columns={"y_pred": "pred_next_hour"}, inplace=True)

    return row[["district", "model_type", "pred_next_hour", "y_true"]]


# ---------------------------------------------------------------------
# Helper: convert boro_cd → "Manhattan CD 04" style name
# ---------------------------------------------------------------------
def boro_cd_to_name(code: int) -> str:
    """
    Convert NYC boro_cd (e.g., 104, 301) into a human-readable
    district name matching the model outputs, e.g.:

        101 → "Manhattan CD 01"
        205 → "Bronx CD 05"
        302 → "Brooklyn CD 02"
        412 → "Queens CD 12"
        501 → "Staten Island CD 01"
    """
    s = str(int(code))  # ensure string, e.g. "101"
    boro = int(s[0])
    num = int(s[1:])  # remaining digits

    borough_map = {
        1: "Manhattan",
        2: "Bronx",
        3: "Brooklyn",
        4: "Queens",
        5: "Staten Island",
    }
    borough = borough_map.get(boro, "Unknown")
    return f"{borough} CD {num:02d}"


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    raw_geo_dir = os.path.join(base_dir, "data", "raw", "district_nyc_data")

    logger.info("Building next-hour demand map (best-model-per-district)…")

    # ---------------------------------------------------------
    # Load evaluation results
    # ---------------------------------------------------------
    eval_df = latest_eval_summary(base_dir)
    if eval_df is None:
        raise SystemExit("Evaluation summary not found.")

    required_cols = {"district", "best_model"}
    if not required_cols.issubset(eval_df.columns):
        raise ValueError("Evaluation summary missing required columns.")

    # ---------------------------------------------------------
    # Collect predictions for each district's best model
    # ---------------------------------------------------------
    preds = []
    for _, row in eval_df.iterrows():
        district = row["district"]
        best_model = row["best_model"]  # e.g. "rmse_rf", "rmse_conv"

        pred_row = load_pred_row(best_model, district, base_dir)
        if pred_row is not None:
            rmse_col = best_model  # column name in eval_df
            pred_row["rmse"] = row[rmse_col] if rmse_col in row else np.nan
            preds.append(pred_row)

    if not preds:
        raise SystemExit("No valid predictions found for best models.")

    pred_df = pd.concat(preds, ignore_index=True)
    pred_df["pred_next_hour"] = pred_df["pred_next_hour"].astype(float)
    pred_df["y_true"] = pred_df["y_true"].astype(float)

    # ---------------------------------------------------------
    # Load polygons (NYC community districts)
    # ---------------------------------------------------------
    geo_files = [
        f for f in os.listdir(raw_geo_dir)
        if f.endswith((".geojson", ".shp", ".gpkg"))
    ]
    if not geo_files:
        raise FileNotFoundError(f"No geo files found in {raw_geo_dir}")

    geo_path = os.path.join(raw_geo_dir, geo_files[0])
    nyc_cds = gpd.read_file(geo_path).to_crs("EPSG:4326")

    # Create cd_name column from boro_cd to match model district naming
    if "boro_cd" not in nyc_cds.columns:
        raise KeyError("Expected 'boro_cd' column in geo file, but it is missing.")

    nyc_cds["cd_name"] = nyc_cds["boro_cd"].apply(boro_cd_to_name)

    # Compute centroids in a projected CRS for better accuracy
    nyc_cds_proj = nyc_cds.to_crs(3857)
    nyc_cds["centroid"] = nyc_cds_proj.geometry.centroid.to_crs(4326)
    nyc_cds["lat"] = nyc_cds["centroid"].y
    nyc_cds["lon"] = nyc_cds["centroid"].x

    # ---------------------------------------------------------
    # Merge predictions with centroids via cd_name
    # ---------------------------------------------------------
    merged = pred_df.merge(
        nyc_cds[["cd_name", "lat", "lon"]],
        left_on="district", right_on="cd_name",
        how="inner"
    )

    if merged.empty:
        raise SystemExit(
            "Merged GeoDataFrame is empty — district names from evaluation "
            "do not align with geo 'boro_cd' mapping."
        )

    # ---------------------------------------------------------
    # Scale circle radius based on predicted demand
    # ---------------------------------------------------------
    vmin, vmax = merged["pred_next_hour"].min(), merged["pred_next_hour"].max()
    merged["radius"] = 6 + (merged["pred_next_hour"] - vmin) * (28 - 6) / max(vmax - vmin, 1e-6)

    # Color coding by RMSE
    def rmse_color(rmse):
        if rmse < 20:
            return "#2ecc71"   # green: excellent
        elif rmse < 40:
            return "#f39c12"   # orange: moderate
        return "#e74c3c"       # red: needs improvement

    merged["color"] = merged["rmse"].apply(rmse_color)

    # ---------------------------------------------------------
    # Build Folium map
    # ---------------------------------------------------------
    m = folium.Map(location=[40.73, -73.97], zoom_start=11, tiles="cartodbpositron")

    # Add per-district circle markers with popup info
    for _, r in merged.iterrows():
        popup_html = (
            f"<b>{r['district']}</b><br>"
            f"Model: <b>{r['model_type']}</b><br>"
            f"Predicted next hour: <b>{r['pred_next_hour']:.1f}</b><br>"
            f"Last observed: {r['y_true']:.1f}<br>"
            f"RMSE: {r['rmse']:.2f} taxis/hr"
        )

        folium.CircleMarker(
            [r["lat"], r["lon"]],
            radius=float(r["radius"]),
            color=r["color"],
            fill=True,
            fill_color=r["color"],
            fill_opacity=0.8,
            weight=1,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(m)

    # ---------------------------------------------------------
    # Business Context Panel (once, fixed top-right)
    # ---------------------------------------------------------
    top3 = (
        merged.groupby("district")["rmse"]
        .mean()
        .reset_index()
        .sort_values("rmse")
        .head(3)
    )

    top_html = "<br>".join(
        f"{i+1}. {row['district']} — RMSE {row['rmse']:.1f} taxis/hr"
        for i, row in top3.iterrows()
    )

    info_html = f"""
    <div style="
        position: fixed; 
        top: 25px; right: 25px; 
        width: 300px; height: 320px; 
        border: 2px solid grey; 
        z-index: 9999; 
        background-color: white; 
        padding: 12px; 
        overflow-y: auto;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.25);
        font-size: 12px;
        line-height: 1.3;
        font-family: Arial, sans-serif;">
    <h4>🎯 Next Hour Forecast &amp; Metrics</h4>
    <p style="margin: 6px 0;">
        Hourly taxi demand per district to adjust vehicle and driver deployment.<br>
        <br>
        Highlights model accuracy and top-performing districts.
    </p><br>

    <b>🏆 Top 3 Best-Performing Districts (lowest RMSE):</b><br>
    {top_html}
    </div>
    """
    m.get_root().html.add_child(folium.Element(info_html))

    # ---------------------------------------------------------
    # Legend for RMSE colors
    # ---------------------------------------------------------
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; width: 210px; height: 140px;
                border:2px solid grey; z-index:9999; background-color:white; padding:10px;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.3); font-family: Arial; font-size: 13px;">
    <b>Forecast Accuracy (RMSE)</b><br>
    <span style="color:#2ecc71;">●</span> Excellent (&lt; 20 taxis/hr)<br>
    <span style="color:#f39c12;">●</span> Moderate (20–40 taxis/hr)<br>
    <span style="color:#e74c3c;">●</span> Needs improvement (&gt; 40 taxis/hr)<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # ---------------------------------------------------------
    # Save output
    # ---------------------------------------------------------
    maps_dir = os.path.join(base_dir, "data", "evaluation", "maps")
    os.makedirs(maps_dir, exist_ok=True)

    timestamp = datetime.datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_html = os.path.join(maps_dir, f"next_hour_accuracy_map_{timestamp}.html")

    m.save(out_html)
    logger.info("Map saved → %s", out_html)

    log_pipeline_step("Generate next-hour map", "success", details=out_html)


if __name__ == "__main__":
    main()
