# src/data/split_data.py

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler

from src.utils.pipeline_logger import init_logger, log_pipeline_step

logger = init_logger(__name__)


# ===========================
# Core Function
# ===========================
def split_and_scale_data(
    df: pd.DataFrame,
    time_col: str = "pickup_datetime",
    target_col: str = "pickups",
    test_size: float = 0.2,
    by_district: bool = False,
):
    """
    Split processed data into train and test sets chronologically.
    Optionally split within each district to prevent time overlap leaks.
    Scales numeric features using StandardScaler.

    Returns:
        train_df, test_df, scaler
    """
    logger.info("Splitting dataset: %d rows, %.0f%% test", len(df), test_size * 100)

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    df = df.sort_values([time_col, "cd_name"] if "cd_name" in df.columns else time_col)

    # --- Split logic ---
    if by_district and "cd_name" in df.columns:
        train_parts, test_parts = [], []
        for dist, group in df.groupby("cd_name"):
            split_idx = int(len(group) * (1 - test_size))
            train_parts.append(group.iloc[:split_idx])
            test_parts.append(group.iloc[split_idx:])
        train_df = pd.concat(train_parts)
        test_df = pd.concat(test_parts)
    else:
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

    # --- Identify feature columns ---
    feature_cols = [c for c in df.columns if c not in [time_col, target_col, "cd_name"]]

    # --- Handle NaNs before scaling ---
    X_train = train_df[feature_cols].fillna(train_df[feature_cols].mean())
    X_test = test_df[feature_cols].fillna(train_df[feature_cols].mean())

    # --- Scale features ---
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train)
    test_scaled = scaler.transform(X_test)

    # --- Reattach metadata ---
    train_out = pd.DataFrame(train_scaled, columns=feature_cols)
    test_out = pd.DataFrame(test_scaled, columns=feature_cols)

    train_out[time_col] = train_df[time_col].values
    train_out[target_col] = train_df[target_col].values
    if "cd_name" in train_df.columns:
        train_out["cd_name"] = train_df["cd_name"].values

    test_out[time_col] = test_df[time_col].values
    test_out[target_col] = test_df[target_col].values
    if "cd_name" in test_df.columns:
        test_out["cd_name"] = test_df["cd_name"].values

    logger.info("Split complete: Train=%d, Test=%d", len(train_out), len(test_out))
    return train_out, test_out, scaler


# ===========================
# Script Entry Point
# ===========================
if __name__ == "__main__":
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    processed_path = os.path.join(base, "data", "processed", "nyc_pickups_features_hourly.csv")
    df = pd.read_csv(processed_path)

    train_df, test_df, scaler = split_and_scale_data(df, by_district=True)

    out_dir = os.path.join(base, "data", "processed")
    os.makedirs(out_dir, exist_ok=True)

    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)

    logger.info("Saved train/test split.")
    log_pipeline_step("Split + scale", "success", details=f"train={len(train_df)}, test={len(test_df)}")
