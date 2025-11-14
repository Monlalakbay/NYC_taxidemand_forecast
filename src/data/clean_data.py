# src/data/clean_data.py

import os
import pandas as pd
import logging
from datetime import datetime
from src.utils.pipeline_logger import init_logger, log_pipeline_step

logger = init_logger(__name__)

# ===========================
# Data cleaning functions
# ===========================
def clean_pickup_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw NYC taxi pickup data:
    - Standardizes columns
    - Ensures datetime format
    - Removes duplicates
    - Filters spatial outliers using IQR + NYC bounding box
    - Removes invalid pickup counts
    - Sorts chronologically
    """
    df = df.copy()
    logger.info("Cleaning DataFrame: %d rows before cleaning", len(df))

    try:
        # --- Standardize column names ---
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # --- Ensure datetime type ---
        if "pickup_datetime" in df.columns:
            df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
            df = df.dropna(subset=["pickup_datetime"])

        # --- Drop duplicates ---
        before_dup = len(df)
        df = df.drop_duplicates()
        logger.info("Removed %d duplicate rows", before_dup - len(df))

        # --- Remove spatial outliers (IQR-based) ---
        if {"lat", "lon"}.issubset(df.columns):
            Q1_lat, Q3_lat = df["lat"].quantile([0.25, 0.75])
            Q1_lon, Q3_lon = df["lon"].quantile([0.25, 0.75])
            IQR_lat = Q3_lat - Q1_lat
            IQR_lon = Q3_lon - Q1_lon

            before_iqr = len(df)
            df = df[
                df["lat"].between(Q1_lat - 1.5 * IQR_lat, Q3_lat + 1.5 * IQR_lat)
                & df["lon"].between(Q1_lon - 1.5 * IQR_lon, Q3_lon + 1.5 * IQR_lon)
                ]
            logger.info("Removed %d IQR spatial outliers", before_iqr - len(df))

            # --- Additional NYC bounding box filter (safety) ---
            before_box = len(df)
            df = df[
                (df["lat"].between(40.5, 41.0))
                & (df["lon"].between(-74.3, -73.7))
                ]
            logger.info("Removed %d outside NYC bounding box", before_box - len(df))

        # --- Remove zero or negative pickup counts (if aggregated) ---
        if "pickups" in df.columns:
            before_pickups = len(df)
            df = df[df["pickups"] > 0]
            logger.info("Removed %d rows with zero/negative pickups", before_pickups - len(df))

        # --- Handle missing categorical values ---
        for col in ["cd_name", "borough"]:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")

        # --- Sort chronologically ---
        if "pickup_datetime" in df.columns:
            df = df.sort_values("pickup_datetime")

        logger.info("Finished cleaning: %d rows after cleaning", len(df))
        return df

    except Exception as e:
        logger.exception("Error while cleaning data: %s", e)
        raise


def load_and_clean_all(raw_folder: str, output_folder: str) -> None:
    os.makedirs(output_folder, exist_ok=True)
    logger.info("Starting batch cleaning: %s -> %s", raw_folder, output_folder)

    for file in os.listdir(raw_folder):
        if file.endswith(".csv"):
            try:
                logger.info("Processing file: %s", file)
                df = pd.read_csv(os.path.join(raw_folder, file))
                df_clean = clean_pickup_data(df)

                out_name = file.replace(".csv", "_clean.csv")
                out_path = os.path.join(output_folder, out_name)
                df_clean.to_csv(out_path, index=False)
                logger.info("Saved cleaned file: %s", out_name)

            except Exception:
                logger.exception("Failed to clean %s", file)

    log_pipeline_step("Data cleaning", "success")


# ===========================
# Script entry point
# ===========================
if __name__ == "__main__":
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    raw = os.path.join(base, "data", "raw", "pickup_data")
    out = os.path.join(base, "data", "cleaned")
    load_and_clean_all(raw, out)
