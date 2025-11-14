# src/models/modeling_conv_lstm_train.py

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from tensorflow.keras import layers, models, optimizers, callbacks, Input
from sklearn.metrics import mean_squared_error, mean_absolute_error  

from src.utils.pipeline_logger import init_logger, log_pipeline_step
from src.data.data_prep import prepare_conv_lstm_data, SEQ_LEN

logger = init_logger(__name__)


# ============================================================
# Helper: district name
# ============================================================
def district_to_slug(district: str) -> str:
    """
    Normalizes district names to a filesystem-safe, lowercase slug.
    Example: "Manhattan CD 01" -> "manhattan_cd_01"
    """
    d = district.strip().lower().replace(" ", "_").replace("/", "_")
    while "__" in d:
        d = d.replace("__", "_")
    return d


# ============================================================
# Build Conv1D + LSTM model (Optimized version)
# ============================================================
def build_conv_lstm(input_shape, learning_rate=1e-3):
    """
    Builds an optimized Conv1D + stacked LSTM model.
    - Two Conv1D blocks with BatchNorm for better feature extraction
    - Reduced recurrent_dropout (deprecated & slow on CPU)
    - Deeper dense head
    - More stable training with Adam + ReduceLROnPlateau
    """
    logger.info("Building Conv1D + LSTM model: input_shape=%s", input_shape)

    model = models.Sequential([
        Input(shape=input_shape),

        # ---- Conv Block 1 ----
        layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        # ---- Conv Block 2 ----
        layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        # ---- LSTM Blocks ----
        layers.LSTM(64, return_sequences=True, dropout=0.3),
        layers.LSTM(32, return_sequences=False, dropout=0.3),

        # ---- Dense Head ----
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),

        layers.Dense(1)
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )

    return model


# ============================================================
# Train Conv1D + LSTM per district
# ============================================================
def train_conv_lstm_per_district(base_dir, artifacts_dir=None, seq_len=24, epochs=25, batch_size=32):
    """
    Trains Conv1D+LSTM models per district using prepared sequences.
    Saves model weights, training history, and predictions.
    """
    start = time.time()
    datasets = prepare_conv_lstm_data(base_dir, use_only_top_k=10, seq_len=seq_len)
    results = []

    if artifacts_dir:
        os.makedirs(artifacts_dir, exist_ok=True)

    for district, pack in datasets.items():
        dist_slug = district_to_slug(district)  # <-- UNIFIED slug logic
        logger.info("Training ConvLSTM for district: %s", district)

        X_tr, y_tr = pack["X_tr"], pack["y_tr"]
        X_va, y_va = pack["X_va"], pack["y_va"]
        X_te, y_te = pack["X_te"], pack["y_te"]
        scaler = pack["scaler"]

        input_shape = (X_tr.shape[1], X_tr.shape[2])

        logger.info(
            "ConvLSTM input: seq_len=%d, n_features=%d",
            X_tr.shape[1], X_tr.shape[2]
        )

        # Log actual feature list if provided
        if "feature_cols" in pack:
            logger.info("ConvLSTM features: %s", pack["feature_cols"])
        else:
            logger.warning("⚠ No feature list provided in pack — cannot verify density feature.")

        model = build_conv_lstm(input_shape)

        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=3, factor=0.5, min_lr=1e-5
        )

        checkpoint = callbacks.ModelCheckpoint(
            os.path.join(artifacts_dir, f"conv_lstm_{dist_slug}.keras"),  # <-- stays .keras
            save_best_only=True,
            monitor="val_loss",
            verbose=0
        )

        cb = [early_stop, reduce_lr, checkpoint]

        # ---- Training ----
        hist = model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=cb
        )

        # ---- Predictions + inverse scaling ----
        y_pred_scaled = model.predict(X_te).flatten()

        dummy_pred = np.zeros((len(y_pred_scaled), scaler.n_features_in_))
        dummy_true = np.zeros_like(dummy_pred)

        dummy_pred[:, 0] = y_pred_scaled
        dummy_true[:, 0] = y_te

        y_pred = scaler.inverse_transform(dummy_pred)[:, 0]
        y_true = scaler.inverse_transform(dummy_true)[:, 0]

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)  # <-- ADDED
        logger.info("%s → RMSE=%.2f MAE=%.2f", district, rmse, mae)  # <-- UPDATED log

        # ---- Store metrics for summary ----
        results.append({
            "district": district,
            "rmse": rmse,
            "mae": mae  # <-- ADDED
        })

        # ---- Save outputs ----
        if artifacts_dir:
            pd.DataFrame(hist.history).to_csv(
                os.path.join(artifacts_dir, f"conv_lstm_history_{dist_slug}.csv"),
                index=False
            )

            pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(
                os.path.join(artifacts_dir, f"conv_lstm_preds_{dist_slug}.csv"),
                index=False
            )

    # ---- Summary ----
    # ---- Summary ----
    summary = pd.DataFrame(results)

    mean_rmse = summary["rmse"].mean()
    mean_mae = summary["mae"].mean()

    summary["mean_rmse"] = mean_rmse
    summary["mean_mae"] = mean_mae

    # SAVE SUMMARY
    if artifacts_dir:
        summary_path = os.path.join(artifacts_dir, "conv_lstm_summary.csv")
        summary.to_csv(summary_path, index=False)

    runtime = round((time.time() - start) / 60, 2)

    log_pipeline_step(
        "ConvLSTM training", "success",
        duration_min=runtime,
        details=(
            f"mean_rmse={mean_rmse:.2f}, "
            f"mean_mae={mean_mae:.2f}"
        )
    )

    return summary


# ===========================
# Script Entry Point
# ===========================
if __name__ == "__main__":
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # UTC timestamp (safe, modern)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base, "data", "runs", f"conv_lstm_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    summary = train_conv_lstm_per_district(base, artifacts_dir=run_dir)
