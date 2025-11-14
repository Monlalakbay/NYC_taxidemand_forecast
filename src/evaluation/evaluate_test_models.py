# src/evaluation/evaluate_test_models.py

import os
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model
from datetime import datetime, timezone

from src.utils.pipeline_logger import init_logger, log_pipeline_step

logger = init_logger(__name__)


# ------------------------------------------------------------
# Helper: find the latest run folder with given prefix
# ------------------------------------------------------------
def latest_run_folder(base_dir, prefix, add_predictions_subfolder=False):
    runs_dir = os.path.join(base_dir, "data", "runs")
    if not os.path.exists(runs_dir):
        logger.warning("No runs directory found.")
        return None

    candidates = [d for d in os.listdir(runs_dir) if d.startswith(prefix)]
    if not candidates:
        logger.warning("No folder found for prefix %s", prefix)
        return None

    latest = sorted(
        candidates,
        key=lambda d: os.path.getmtime(os.path.join(runs_dir, d))
    )[-1]

    folder = os.path.join(runs_dir, latest)
    if add_predictions_subfolder:
        folder = os.path.join(folder, "predictions")

    logger.info("Using latest run for %s → %s", prefix, latest)
    return folder


# ------------------------------------------------------------
# Helper: normalize district name to slug (same as training)
# ------------------------------------------------------------
def normalize_district_name(d: str) -> str:
    """
    Mirrors district_to_slug() in modeling_conv_lstm_train.py
    Example: "Manhattan CD 01" -> "manhattan_cd_01"
    """
    d = d.strip().lower().replace(" ", "_").replace("/", "_")
    while "__" in d:
        d = d.replace("__", "_")
    return d


# ------------------------------------------------------------
# Evaluate models
# ------------------------------------------------------------
def evaluate_test_models(base_dir, seq_len=24):
    """Evaluate RF, LGBM, ConvLSTM, and Hybrid models on test.csv."""
    start = time.time()
    logger.info("Evaluating all models on test.csv …")

    processed_dir = os.path.join(base_dir, "data", "processed")
    eval_dir = os.path.join(base_dir, "data", "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    # Load test.csv
    test_path = os.path.join(processed_dir, "test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset not found: {test_path}")

    df = pd.read_csv(test_path, parse_dates=["pickup_datetime"])
    logger.info("Loaded test dataset (%d rows)", len(df))

    # Load scaler (used for ConvLSTM inverse transform)
    scaler = joblib.load(os.path.join(processed_dir, "scaler.pkl"))

    top_districts = df["cd_name"].value_counts().head(10).index.tolist()
    features = [
        "pickups",
        "sin_hour", "cos_hour",
        "sin_weekday", "cos_weekday",
        "is_weekend",
        "lag_1", "lag_2", "lag_24",
        "rolling_mean_3h", "rolling_std_3h",
        "rolling_mean_24h", "rolling_std_24h",
        "pickup_density_cd"
    ]

    # Locate model prediction folders
    rf_dir = latest_run_folder(base_dir, "tree_preds_", add_predictions_subfolder=True)
    conv_dir = latest_run_folder(base_dir, "conv_lstm_")
    hybrid_dir = latest_run_folder(base_dir, "hybrid_")

    results = []

    # ------------------------------------------------------------
    # DISTRICT LOOP
    # ------------------------------------------------------------
    for district in top_districts:
        logger.info("Evaluating district: %s", district)

        df_d = df[df["cd_name"] == district].sort_values("pickup_datetime").copy()
        scaled_values = scaler.transform(df_d[features].values)

        # --- Prepare ConvLSTM sequences ---
        X_seq, y_true_scaled = [], []
        for i in range(seq_len, len(scaled_values)):
            X_seq.append(scaled_values[i - seq_len:i])
            y_true_scaled.append(scaled_values[i, 0])

        X_seq = np.array(X_seq)
        y_true_scaled = np.array(y_true_scaled)

        # Inverse transform for y_true
        dummy = np.zeros((len(y_true_scaled), scaler.n_features_in_))
        dummy[:, 0] = y_true_scaled
        y_true = scaler.inverse_transform(dummy)[:, 0]

        # --- Evaluate ConvLSTM ---
        rmse_conv, mae_conv = np.nan, np.nan
        if conv_dir:
            safe_district = normalize_district_name(district)
            model_path = os.path.join(conv_dir, f"conv_lstm_{safe_district}.keras")  # <-- changed to .keras

            if os.path.exists(model_path):
                logger.info("Loading ConvLSTM model from %s", model_path)
                model = load_model(model_path, compile=False)
                y_pred_scaled = model.predict(X_seq).flatten()
                dummy[:, 0] = y_pred_scaled
                y_pred = scaler.inverse_transform(dummy)[:, 0]
                rmse_conv = np.sqrt(mean_squared_error(y_true, y_pred))
                mae_conv = mean_absolute_error(y_true, y_pred)
            else:
                logger.warning("⚠ ConvLSTM model not found for %s → %s", district, model_path)

        # --- Helper for CSV predictions ---
        def eval_csv_prediction(path, col):
            if not path or not os.path.exists(path):
                return np.nan, np.nan
            dfp = pd.read_csv(path)
            return (
                np.sqrt(mean_squared_error(dfp["y_true"], dfp[col])),
                mean_absolute_error(dfp["y_true"], dfp[col])
            )

        rf_file = os.path.join(rf_dir, f"rf_preds_{district.replace(' ', '_')}.csv") if rf_dir else None
        lgbm_file = os.path.join(rf_dir, f"lgbm_preds_{district.replace(' ', '_')}.csv") if rf_dir else None
        hybrid_file = os.path.join(hybrid_dir, f"hybrid_preds_{district.replace(' ', '_')}.csv") if hybrid_dir else None

        rmse_rf, mae_rf = eval_csv_prediction(rf_file, "y_pred")
        rmse_lgbm, mae_lgbm = eval_csv_prediction(lgbm_file, "y_pred")
        rmse_hybrid, mae_hybrid = eval_csv_prediction(hybrid_file, "y_pred_blend")

        results.append({
            "district": district,
            "rmse_rf": rmse_rf, "mae_rf": mae_rf,
            "rmse_lgbm": rmse_lgbm, "mae_lgbm": mae_lgbm,
            "rmse_conv": rmse_conv, "mae_conv": mae_conv,
            "rmse_hybrid": rmse_hybrid, "mae_hybrid": mae_hybrid
        })

    # ------------------------------------
    # Save summary
    # ------------------------------------
    summary = pd.DataFrame(results)

    summary["best_model"] = summary[
        ["rmse_rf", "rmse_lgbm", "rmse_conv", "rmse_hybrid"]
    ].idxmin(axis=1)

    summary["mean_rmse"] = summary[
        ["rmse_rf", "rmse_lgbm", "rmse_conv", "rmse_hybrid"]
    ].mean(axis=1)

    out_path = os.path.join(
        eval_dir,
        f"all_models_eval_test_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    )
    summary.to_csv(out_path, index=False)
    logger.info("Evaluation complete → %s", out_path)

    # Pipeline logging
    duration = round((time.time() - start) / 60, 2)
    log_pipeline_step(
        "Unified evaluation on test dataset",
        "success",
        duration_min=duration,
        details=f"mean_rmse={summary['mean_rmse'].mean():.3f}"
    )

    return summary


# ===========================
# Script Entry Point
# ===========================
if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    evaluate_test_models(BASE_DIR)
