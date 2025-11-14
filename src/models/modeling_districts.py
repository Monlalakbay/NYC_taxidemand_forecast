# src/models/modeling_districts.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from datetime import datetime

from src.utils.pipeline_logger import init_logger, log_pipeline_step

logger = init_logger(__name__)


# =========================================================
# Helpers
# =========================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def slugify(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


# =========================================================
# Load Data
# =========================================================

def load_data():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    train_path = os.path.join(base_dir, "data", "processed", "train.csv")
    test_path = os.path.join(base_dir, "data", "processed", "test.csv")

    logger.info("Loading train/test data …")
    train = pd.read_csv(train_path, parse_dates=["pickup_datetime"])
    test = pd.read_csv(test_path, parse_dates=["pickup_datetime"])

    logger.info("Loaded train=%d, test=%d", len(train), len(test))
    return train, test, base_dir


# =========================================================
# Single-Model Training per District
# =========================================================

def train_and_evaluate_district_single(district, train, test, base_features, model_type):
    logger.info("Single-model training for: %s (%s)", district, model_type)

    train_d = train[train["cd_name"] == district].copy().sort_values("pickup_datetime")
    test_d = test[test["cd_name"] == district].copy().sort_values("pickup_datetime")

    # Feature creation
    for df in [train_d, test_d]:
        df["lag_1"] = df["pickups"].shift(1)
        df["lag_2"] = df["pickups"].shift(2)
        df["lag_24"] = df["pickups"].shift(24)
        df["rolling_mean_3h"] = df["pickups"].shift(1).rolling(3, min_periods=1).mean()
        df["rolling_mean_24h"] = df["pickups"].shift(1).rolling(24, min_periods=1).mean()

    train_d = train_d.dropna(subset=["lag_1", "lag_2", "lag_24"])
    test_d = test_d.dropna(subset=["lag_1", "lag_2", "lag_24"])

    if train_d.empty or test_d.empty:
        logger.warning("Insufficient data for %s", district)
        return None

    features = base_features + [
    "lag_1", "lag_2", "lag_24",
    "rolling_mean_3h", "rolling_mean_24h",
    "rolling_std_3h", "rolling_std_24h",
    "pickup_density_cd"
    ]
    X_train, y_train = train_d[features], train_d["pickups"]
    X_test, y_test = test_d[features], test_d["pickups"]

    baseline_rmse = np.sqrt(mean_squared_error(y_test, test_d["lag_1"]))

    # Model selection
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "ridge":
        model = Ridge(alpha=1.0)
    elif model_type == "lasso":
        model = Lasso(alpha=0.1)
    elif model_type == "rf":
        model = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, n_jobs=-1, random_state=42)
    elif model_type == "lgbm":
        model = lgb.LGBMRegressor(
            objective="regression",
            learning_rate=0.03,
            num_leaves=64,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            n_estimators=5000,
            min_data_in_leaf=50,
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unsupported model_type={model_type}")

    # Train
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    corr = np.corrcoef(y_test, y_pred)[0, 1]

    logger.info("%s: RMSE=%.2f (baseline=%.2f) corr=%.3f", district, rmse, baseline_rmse, corr)

    return {
        "district": district,
        "model_rmse": rmse,
        "baseline_rmse": baseline_rmse,
        "improvement_%": 100 * (1 - rmse / baseline_rmse),
        "correlation": corr,
    }


# =========================================================
# Ensemble per District (RF + LGBM Soft Blend)
# =========================================================

def train_and_evaluate_district_ensemble(district, train, test, base_features, base_dir, run_id):
    logger.info("Training ENSEMBLE for district: %s", district)

    train_d = train[train["cd_name"] == district].copy().sort_values("pickup_datetime")
    test_d = test[test["cd_name"] == district].copy().sort_values("pickup_datetime")

    # Create features
    for df in [train_d, test_d]:
        df["lag_1"] = df["pickups"].shift(1)
        df["lag_2"] = df["pickups"].shift(2)
        df["lag_24"] = df["pickups"].shift(24)
        df["rolling_mean_3h"] = df["pickups"].shift(1).rolling(3, min_periods=1).mean()
        df["rolling_mean_24h"] = df["pickups"].shift(1).rolling(24, min_periods=1).mean()

    train_d = train_d.dropna(subset=["lag_1", "lag_2", "lag_24"])
    test_d = test_d.dropna(subset=["lag_1", "lag_2", "lag_24"])

    if train_d.empty or test_d.empty:
        logger.warning("Insufficient data for %s", district)
        return None

    features = base_features + ["lag_1", "lag_2", "lag_24", "rolling_mean_3h", "rolling_mean_24h"]
    X_train, y_train = train_d[features], train_d["pickups"]
    X_test, y_test = test_d[features], test_d["pickups"]

    # Validation split
    cutoff = train_d["pickup_datetime"].max() - pd.Timedelta(days=14)
    tr_idx = train_d["pickup_datetime"] < cutoff
    va_idx = ~tr_idx

    if va_idx.sum() < 100:  # fallback
        split = int(0.9 * len(train_d))
        tr_idx = np.zeros(len(train_d), dtype=bool)
        tr_idx[:split] = True
        va_idx = ~tr_idx

    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_va, y_va = X_train[va_idx], y_train[va_idx]

    # Random Forest
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    va_rf = rf.predict(X_va)

    # LightGBM
    lgbm = lgb.LGBMRegressor(
        objective="regression",
        learning_rate=0.03,
        num_leaves=64,
        min_data_in_leaf=50,
        n_estimators=5000,
        random_state=42,
        n_jobs=-1,
    )
    lgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="rmse",
             callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
    va_lgb = lgbm.predict(X_va, num_iteration=lgbm.best_iteration_)

    # Soft blending weights via validation RMSE
    rmse_rf = np.sqrt(mean_squared_error(y_va, va_rf))
    rmse_lgb = np.sqrt(mean_squared_error(y_va, va_lgb))
    w_rf = 1 / (rmse_rf + 1e-6)
    w_lgb = 1 / (rmse_lgb + 1e-6)
    w_rf, w_lgb = w_rf / (w_rf + w_lgb), w_lgb / (w_rf + w_lgb)

    # Final ensemble
    pred_rf = rf.predict(X_test)
    pred_lgb = lgbm.predict(X_test, num_iteration=lgbm.best_iteration_)
    pred_ens = w_rf * pred_rf + w_lgb * pred_lgb
    baseline_rmse = np.sqrt(mean_squared_error(y_test, test_d["lag_1"]))
    rmse_ens = np.sqrt(mean_squared_error(y_test, pred_ens))
    corr_ens = np.corrcoef(y_test, pred_ens)[0, 1]

    logger.info("%s → Ensemble RMSE=%.2f baseline=%.2f corr=%.3f", district, rmse_ens, baseline_rmse, corr_ens)

    # Save predictions
    pred_dir = os.path.join(base_dir, "data", "runs", run_id, "predictions")
    ensure_dir(pred_dir)

    slug = slugify(district)

    pd.DataFrame({"pickup_datetime": test_d["pickup_datetime"], "y_true": y_test, "y_pred": pred_rf}).to_csv(
        os.path.join(pred_dir, f"rf_preds_{slug}.csv"), index=False
    )
    pd.DataFrame({"pickup_datetime": test_d["pickup_datetime"], "y_true": y_test, "y_pred": pred_lgb}).to_csv(
        os.path.join(pred_dir, f"lgbm_preds_{slug}.csv"), index=False
    )
    pd.DataFrame({
        "pickup_datetime": test_d["pickup_datetime"],
        "y_true": y_test,
        "pred_rf": pred_rf,
        "pred_lgbm": pred_lgb,
        "pred_ensemble": pred_ens
    }).to_csv(os.path.join(pred_dir, f"ensemble_preds_{slug}.csv"), index=False)

    return {
        "district": district,
        "model_rmse": rmse_ens,
        "baseline_rmse": baseline_rmse,
        "improvement_%": 100 * (1 - rmse_ens / baseline_rmse),
        "correlation": corr_ens,
    }


# =========================================================
# Run All Districts (Single or Ensemble)
# =========================================================

def run_all_districts(train, test, base_dir, model_type="ensemble"):
    base_features = ["sin_hour", "cos_hour", "sin_weekday", "cos_weekday", "is_weekend"]
    top_districts = train["cd_name"].value_counts().head(10).index.tolist()

    run_id = f"tree_preds_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    results = []

    for district in top_districts:
        if model_type in ["linear", "lasso", "ridge", "rf", "lgbm"]:
            res = train_and_evaluate_district_single(district, train, test, base_features, model_type)
        else:
            res = train_and_evaluate_district_ensemble(district, train, test, base_features, base_dir, run_id)

        if res:
            results.append(res)

    summary = pd.DataFrame(results)
    summary["mean_rmse"] = summary["model_rmse"].mean()

    out_dir = os.path.join(base_dir, "data", "runs", run_id)
    ensure_dir(out_dir)
    summary.to_csv(os.path.join(out_dir, f"summary_{model_type}.csv"), index=False)

    logger.info("Saved %s summary", model_type)

    return summary, run_id


# ===========================
# Script entry point
# ===========================

if __name__ == "__main__":
    start = datetime.utcnow()

    train, test, base_dir = load_data()
    summary, run_id = run_all_districts(train, test, base_dir, model_type="ensemble")

    runtime = round((datetime.utcnow() - start).total_seconds() / 60, 2)
    log_pipeline_step("District modeling", "success", duration_min=runtime, details=f"run={run_id}")

    logger.info("Finished district modeling in %.2f minutes", runtime)
