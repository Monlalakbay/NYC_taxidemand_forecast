# src/data/preprocess.py

import os
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from glob import glob

from src.utils.pipeline_logger import init_logger, log_pipeline_step

logger = init_logger(__name__)


# ===========================
# Core Preprocessing Function
# ===========================
def preprocess_pickup_data(df: pd.DataFrame, nyc_cds: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Preprocess NYC taxi pickup data:
    - Adds district names via spatial join
    - Aggregates pickups by district and hour
    - Extracts temporal, cyclical, and lag-based features
    """
    df = df.copy()
    logger.info("Starting preprocessing on %d rows", len(df))

    # Normalize coordinate and datetime columns inside the main DF
    rename_map = {}
    for col in df.columns:
        lc = col.lower()
        if lc in ["lat", "latitude", "pickup_latitude"]:
            rename_map[col] = "Lat"
        elif lc in ["lon", "lng", "longitude", "pickup_longitude"]:
            rename_map[col] = "Lon"
        elif "date" in lc or "time" in lc:
            rename_map[col] = "pickup_datetime"

    df = df.rename(columns=rename_map)
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df = df.dropna(subset=["pickup_datetime", "Lat", "Lon"])

    geometry = [Point(xy) for xy in zip(df["Lon"], df["Lat"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Ensure district names
    if "cd_name" not in nyc_cds.columns:
        logger.info("Generating cd_name...")
        borough_map = {
            "1": "Manhattan",
            "2": "Bronx",
            "3": "Brooklyn",
            "4": "Queens",
            "5": "Staten Island",
        }
        id_col = "boro_cd" if "boro_cd" in nyc_cds.columns else None
        nyc_cds["cd_name"] = nyc_cds[id_col].astype(str).apply(
            lambda x: f"{borough_map.get(x[0], 'Unknown')} CD {x[1:]}"
        )

    # Spatial join
    gdf = gpd.sjoin(
        gdf,
        nyc_cds[["cd_name", "geometry"]],
        how="left",
        predicate="within"
    )

    df = pd.DataFrame(gdf.drop(columns="geometry")).dropna(subset=["cd_name"])

    # Aggregate pickups by district-hour
    df["pickup_hour"] = df["pickup_datetime"].dt.floor("h")
    df_hour = df.groupby(["cd_name", "pickup_hour"]).size().reset_index(name="pickups")
    df_hour = df_hour.rename(columns={"pickup_hour": "pickup_datetime"})

    df_hour["hour"] = df_hour["pickup_datetime"].dt.hour
    df_hour["pickup_density_cd"] = df_hour.groupby(["cd_name", "hour"])["pickups"].transform("mean")

    # Temporal features
    df_hour["weekday"] = df_hour["pickup_datetime"].dt.weekday
    df_hour["month"] = df_hour["pickup_datetime"].dt.month
    df_hour["day"] = df_hour["pickup_datetime"].dt.day
    df_hour["is_weekend"] = (df_hour["weekday"] >= 5).astype(int)

    # Cyclical encoding
    df_hour["sin_hour"] = np.sin(2 * np.pi * df_hour["hour"] / 24)
    df_hour["cos_hour"] = np.cos(2 * np.pi * df_hour["hour"] / 24)
    df_hour["sin_weekday"] = np.sin(2 * np.pi * df_hour["weekday"] / 7)
    df_hour["cos_weekday"] = np.cos(2 * np.pi * df_hour["weekday"] / 7)

    # Lag features
    df_hour = df_hour.sort_values(["cd_name", "pickup_datetime"])
    df_hour["lag_1"] = df_hour.groupby("cd_name")["pickups"].shift(1)
    df_hour["lag_2"] = df_hour.groupby("cd_name")["pickups"].shift(2)
    df_hour["lag_24"] = df_hour.groupby("cd_name")["pickups"].shift(24)

    df_hour["rolling_mean_3h"] = (
        df_hour.groupby("cd_name")["pickups"].transform(lambda x: x.rolling(3).mean())
    )
    df_hour["rolling_mean_24h"] = (
        df_hour.groupby("cd_name")["pickups"].transform(lambda x: x.rolling(24).mean())
    )

    df_hour["rolling_std_3h"] = (
        df_hour.groupby("cd_name")["pickups"].transform(lambda x: x.rolling(3).std())
    )
    df_hour["rolling_std_24h"] = (
        df_hour.groupby("cd_name")["pickups"].transform(lambda x: x.rolling(24).std())
    )

    df_hour = df_hour.dropna().reset_index(drop=True)

    logger.info("Preprocessing complete: %d rows", len(df_hour))
    return df_hour


# ===========================
# Script Entry Point
# ===========================
if __name__ == "__main__":
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cleaned = os.path.join(base, "data", "cleaned")
    district_dir = os.path.join(base, "data", "raw", "district_nyc_data")
    processed = os.path.join(base, "data", "processed")

    files = glob(os.path.join(cleaned, "*.csv"))
    df_list = []

    for f in files:
        try:
            temp = pd.read_csv(f)

            # Normalize timestamp column
            if "pickup_datetime" not in temp.columns:
                if "date/time" in temp.columns:
                    temp = temp.rename(columns={"date/time": "pickup_datetime"})
                elif "datetime" in temp.columns:
                    temp = temp.rename(columns={"datetime": "pickup_datetime"})
                elif "tpep_pickup_datetime" in temp.columns:
                    temp = temp.rename(columns={"tpep_pickup_datetime": "pickup_datetime"})
                else:
                    raise KeyError(
                        f"No valid datetime column found in {f}. "
                        f"Columns: {list(temp.columns)}"
                    )

            temp["pickup_datetime"] = pd.to_datetime(temp["pickup_datetime"], errors="coerce")

            # Normalize coordinate columns
            coord_rename = {}
            for col in temp.columns:
                lc = col.lower()
                if lc in ["lat", "latitude", "pickup_latitude"]:
                    coord_rename[col] = "Lat"
                elif lc in ["lon", "lng", "longitude", "pickup_longitude"]:
                    coord_rename[col] = "Lon"

            temp = temp.rename(columns=coord_rename)

            # Append valid rows
            if {"Lat", "Lon"}.issubset(temp.columns):
                temp["pickups"] = 1
                df_list.append(temp[["pickup_datetime", "Lat", "Lon", "pickups"]])
            else:
                logger.warning("Skipping %s — missing Lat/Lon columns after normalization.", f)

        except Exception:
            logger.exception("Failed reading %s", f)

    if not df_list:
        raise ValueError("No valid cleaned files found.")

    df_all = pd.concat(df_list, ignore_index=True)

    geo_files = [g for g in os.listdir(district_dir) if g.endswith((".geojson", ".shp"))]
    nyc_cds = gpd.read_file(os.path.join(district_dir, geo_files[0]))

    df_processed = preprocess_pickup_data(df_all, nyc_cds)
    os.makedirs(processed, exist_ok=True)

    out_path = os.path.join(processed, "nyc_pickups_features_hourly.csv")
    df_processed.to_csv(out_path, index=False)

    logger.info("Saved processed dataset: %s", out_path)
    log_pipeline_step("Preprocessing", "success", details=f"rows={len(df_processed)}")
