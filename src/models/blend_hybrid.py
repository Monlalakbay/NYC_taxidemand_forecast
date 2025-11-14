# src/models/blend_hybrid.py — Hybrid blending (RF + LGBM + ConvLSTM)

import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV, ElasticNetCV

from src.utils.pipeline_logger import init_logger, log_pipeline_step

logger = init_logger(__name__)


# ============================================================
# Utilities
# ============================================================
def _rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ============================================================
# Load ConvLSTM predictions
# ============================================================
def _latest_conv_run_dir(base_dir):
    """Returns the most recent conv_lstm_* run directory."""
    runs_dir = os.path.join(base_dir, "data", "runs")
    runs = sorted([d for d in os.listdir(runs_dir) if d.startswith("conv_lstm_")])
    if not runs:
        raise FileNotFoundError("❌ No conv_lstm_* runs found.")
    logger.info("🔍 Using latest ConvLSTM run: %s", runs[-1])
    return os.path.join(runs_dir, runs[-1])


def _load_conv_preds(conv_dir):
    """Loads all ConvLSTM predictions, normalized to lowercase district names."""
    files = glob.glob(os.path.join(conv_dir, "conv_lstm_preds_*.csv"))
    preds = {}
    for f in files:
        name = (
            os.path.basename(f)
            .replace("conv_lstm_preds_", "")
            .replace(".csv", "")
            .replace("_", " ")
            .lower()
        )
        preds[name] = pd.read_csv(f)
    logger.info("📦 Loaded ConvLSTM predictions for %d districts", len(preds))
    return preds


def _align_tail(a, b):
    """Return last aligned n rows based on smallest length."""
    n = min(len(a), len(b))
    return a[-n:], b[-n:]


# ============================================================
# Hybrid blending (RF + LGBM + ConvLSTM)
# ============================================================
def blend_with_conv_lstm(base_dir, tree_dir, out_dir=None):
    logger.info("Starting hybrid blending …")

    # Ensure tree_dir points to /predictions
    if os.path.basename(tree_dir).lower() != "predictions":
        tree_dir = os.path.join(tree_dir, "predictions")

    conv_dir = _latest_conv_run_dir(base_dir)
    _ensure_dir(out_dir)

    conv = _load_conv_preds(conv_dir)
    results = []

    # ----------------------------------------------------------
    # Load RF and LGBM predictions
    # ----------------------------------------------------------
    tree_preds = {}
    for model_type in ("rf", "lgbm"):
        pattern = os.path.join(tree_dir, f"{model_type}_preds_*.csv")
        for f in glob.glob(pattern):
            district = (
                os.path.basename(f)
                .replace(f"{model_type}_preds_", "")
                .replace(".csv", "")
                .replace("_", " ")
                .lower()
            )
            tree_preds.setdefault(district, {})[model_type] = pd.read_csv(f)

    # ----------------------------------------------------------
    # Blend per district
    # ----------------------------------------------------------
    for district, models in tree_preds.items():
        if "rf" not in models or "lgbm" not in models or district not in conv:
            continue

        df_rf = models["rf"]
        df_lgb = models["lgbm"]
        df_conv = conv[district]

        y_true1, y_rf = _align_tail(df_rf["y_true"].values, df_rf["y_pred"].values)
        y_true2, y_lgb = _align_tail(df_lgb["y_true"].values, df_lgb["y_pred"].values)
        y_true3, y_conv = _align_tail(df_conv["y_true"].values, df_conv["y_pred"].values)

        n = min(len(y_true1), len(y_true2), len(y_true3))
        if n == 0:
            continue

        y_true = y_true1[-n:]
        X = np.vstack([y_rf[-n:], y_lgb[-n:], y_conv[-n:]]).T

        # -----------------------------
        # Meta-model selection
        # -----------------------------
        if len(y_true) < 20:
            y_pred = X.mean(axis=1)
            rmse = _rmse(y_true, y_pred)
            method = "MeanBlend"

        else:
            split = max(10, int(0.6 * len(y_true)))
            X_val, y_val = X[:split], y_true[:split]
            X_hold, y_hold = X[split:], y_true[split:]

            ridge = RidgeCV(alphas=np.logspace(-3, 2, 20))
            ridge.fit(X_val, y_val)
            pred_ridge = ridge.predict(X_hold)
            rmse_ridge = _rmse(y_hold, pred_ridge)

            enet = ElasticNetCV(
                l1_ratio=[0.1, 0.3, 0.5, 0.7, 1.0],
                alphas=np.logspace(-3, 1, 15),
                cv=3,
                max_iter=20000,
            )
            enet.fit(X_val, y_val)
            pred_enet = enet.predict(X_hold)
            rmse_enet = _rmse(y_hold, pred_enet)

            if rmse_enet < rmse_ridge:
                rmse = rmse_enet
                y_pred = pred_enet
                method = "ElasticNet"
            else:
                rmse = rmse_ridge
                y_pred = pred_ridge
                method = "Ridge"

        # Save per-district predictions
        out_df = pd.DataFrame(
            {
                "y_true": y_hold if len(y_true) >= 20 else y_true,
                "y_pred_blend": y_pred,
                "y_pred_rf": y_rf[-len(y_pred):],
                "y_pred_lgbm": y_lgb[-len(y_pred):],
                "y_pred_conv": y_conv[-len(y_pred):],
            }
        )

        out_df.to_csv(
            os.path.join(out_dir, f"hybrid_preds_{district.replace(' ', '_')}.csv"),
            index=False,
        )

        results.append({"district": district, "rmse_blend": rmse, "method": method})

    summary = pd.DataFrame(results)
    summary.to_csv(os.path.join(out_dir, "hybrid_summary.csv"), index=False)

    logger.info("Hybrid blending completed.")
    return summary


# ===========================
# Script Entry Point
# ===========================
if __name__ == "__main__":
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    runs_root = os.path.join(base, "data", "runs")
    tree_runs = sorted([d for d in os.listdir(runs_root) if d.startswith("tree_preds_")])
    tree_dir = os.path.join(runs_root, tree_runs[-1])

    # Timestamp with UTC
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base, "data", "runs", f"hybrid_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    summary = blend_with_conv_lstm(base, tree_dir, out_dir=out_dir)

    log_pipeline_step("Hybrid blending", "success", details=f"districts={len(summary)}")
