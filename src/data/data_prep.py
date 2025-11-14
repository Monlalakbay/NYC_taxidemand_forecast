# src/data/data_prep.py

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from src.utils.pipeline_logger import init_logger, log_pipeline_step

logger = init_logger(__name__)

# ===========================
# Default Config
# ===========================
SEQ_LEN = 24  # default global sequence length


# ===========================
# Main Function
# ===========================
def prepare_conv_lstm_data(base_dir: str, use_only_top_k=10, seq_len=24):
    """
    Prepares Conv1D+LSTM input sequences with a configurable window length.

    Args:
        base_dir (str): Project base directory.
        use_only_top_k (int): Number of most active districts to include.
        seq_len (int): Number of timesteps to look back.

    Returns:
        dict: One entry per district, each containing scaled train/val/test splits,
              full feature list, and the scaler.
    """
    global SEQ_LEN
    SEQ_LEN = seq_len

    path = os.path.join(base_dir, "data", "processed", "nyc_pickups_features_hourly.csv")

    # Load
    df = pd.read_csv(path, parse_dates=["pickup_datetime"])
    df = df.sort_values(["cd_name", "pickup_datetime"])
    logger.info("Loaded dataset for ConvLSTM: %d rows", len(df))

    feature_cols = [
        "pickups",
        "lag_1", "lag_2", "lag_24",
        "rolling_mean_3h", "rolling_mean_24h",
        "rolling_std_3h", "rolling_std_24h",
        "pickup_density_cd",
        "sin_hour", "cos_hour",
        "sin_weekday", "cos_weekday",
        "is_weekend",
    ]

    # Safety check
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        raise KeyError(
            f"❌ Missing required features for ConvLSTM: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    logger.info("ConvLSTM feature list (%d features): %s", len(feature_cols), feature_cols)

    # ==========================================================
    # Select the top K most active districts
    # ==========================================================
    top_districts = df["cd_name"].value_counts().head(use_only_top_k).index
    logger.info("Using top districts for ConvLSTM: %s", list(top_districts))

    datasets = {}

    for district in top_districts:
        df_d = df[df["cd_name"] == district].copy()

        # ---- Scaling per district ----
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df_d[feature_cols].values)

        # ---- Create sliding windows ----
        X, y = [], []
        for i in range(seq_len, len(scaled)):
            X.append(scaled[i - seq_len : i])
            y.append(scaled[i, 0])  # pickups is first column → target

        X, y = np.array(X), np.array(y)

        # ---- Time-aware split ----
        n = len(X)
        n_train, n_val = int(0.8 * n), int(0.9 * n)

        datasets[district] = {
            "X_tr": X[:n_train],
            "y_tr": y[:n_train],
            "X_va": X[n_train:n_val],
            "y_va": y[n_train:n_val],
            "X_te": X[n_val:],
            "y_te": y[n_val:],
            "scaler": scaler,
            "feature_cols": feature_cols,
        }

        logger.info(
            "District %-20s → %5d samples | features=%d",
            district, n, len(feature_cols)
        )

    # Save scaler from last district
    joblib.dump(scaler, os.path.join(base_dir, "data", "processed", "scaler.pkl"))

    log_pipeline_step("ConvLSTM data prep", "success", details=f"districts={len(datasets)}")

    return datasets


# ===========================
# Script Entry Point
# ===========================
if __name__ == "__main__":
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    prepare_conv_lstm_data(base, use_only_top_k=10, seq_len=24)
