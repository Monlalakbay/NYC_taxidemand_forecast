# src/data/run_pipeline.py — Unified Pipeline Runner

import os
import sys
import time
import subprocess
import threading
from tqdm import tqdm
from datetime import datetime, timezone

from src.utils.pipeline_logger import init_logger, log_pipeline_step

logger = init_logger(__name__)

HERE = os.path.dirname(__file__)
BASE_DIR = os.path.abspath(os.path.join(HERE, ".."))
ENABLE_HYBRID = True  # toggle for hybrid blend


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def abs_path(rel_path: str) -> str:
    """Create absolute path based on project root."""
    return os.path.join(BASE_DIR, rel_path.replace("/", os.sep))


# ------------------------------------------------------------
# Live spinner progress bar
# ------------------------------------------------------------
def run_with_progress(cmd, description):
    pbar = tqdm(total=0, desc=f"⏳ {description}", bar_format="{desc}")
    done = threading.Event()

    def spinner():
        symbols = ["|", "/", "-", "\\"]
        idx = 0
        while not done.is_set():
            pbar.set_description_str(f"⏳ {description} {symbols[idx % 4]}")
            idx += 1
            time.sleep(0.1)

    thread = threading.Thread(target=spinner)
    thread.start()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    finally:
        done.set()
        thread.join()
        pbar.close()

    return result


# ------------------------------------------------------------
# Run each submodule & log results
# ------------------------------------------------------------
def run_module(script_rel_path: str, description: str):
    script_abs = abs_path(script_rel_path)
    logger.info("=" * 80)
    logger.info(f"▶️  {description}")
    logger.info("=" * 80)

    start = time.time()
    status = "success"

    try:
        result = run_with_progress([sys.executable, script_abs], description)
        duration = round((time.time() - start) / 60, 2)

        # Show stdout/stderr
        if result.stdout:
            logger.info(result.stdout)
        if result.stderr:
            logger.warning("stderr:\n%s", result.stderr)

        if result.returncode != 0:
            status = f"error: returncode {result.returncode}"
            logger.error(f"❌ Step failed with code {result.returncode}")

        else:
            logger.info(f"✅ Completed {description} in {duration} min.")

    except Exception as e:
        duration = round((time.time() - start) / 60, 2)
        status = f"exception: {str(e)}"
        logger.exception(f"❌ {description} failed — {e}")

    # Unified log entry
    log_pipeline_step(description, status, duration_min=duration)

    if status.startswith("error") or status.startswith("exception"):
        sys.exit(1)


# ------------------------------------------------------------
# Forecast CSV Export
# ------------------------------------------------------------
def export_forecast_report(base_dir):
    logger.info("📈 Exporting latest forecast report...")

    eval_dir = os.path.join(base_dir, "data", "evaluation")
    runs_dir = os.path.join(base_dir, "data", "runs")

    eval_runs = sorted([d for d in os.listdir(eval_dir) if d.startswith("all_models_eval_")])
    if not eval_runs:
        logger.warning("⚠️ No evaluation summary found — skipping forecast export.")
        return

    latest_folder = eval_runs[-1]
    summary_file = os.path.join(eval_dir, latest_folder)

    if not os.path.exists(summary_file):
        logger.warning("⚠️ Summary file missing — skipping forecast export.")
        return

    import pandas as pd

    df = pd.read_csv(summary_file)

    out_path = os.path.join(eval_dir, "forecast_latest.csv")
    df.to_csv(out_path, index=False)

    logger.info(f"✅ Forecast exported to: {out_path}")
    log_pipeline_step("Export forecast report", "success", details=f"file={out_path}")


# ------------------------------------------------------------
# Pipeline Steps
# ------------------------------------------------------------
if __name__ == "__main__":
    logger.info(f"🚀 Starting pipeline (UTC: {utc_now()})")
    logger.info(f"📂 Project root: {BASE_DIR}")

    steps = [
        ("src/data/clean_data.py", "Cleaning raw taxi pickup files"),
        ("src/data/preprocess.py", "Preprocessing taxi pickup data"),
        ("src/data/split_data.py", "Splitting & scaling dataset"),
        ("src/models/modeling_districts.py", "Training ensemble (RF + LGBM)"),
        ("src/models/modeling_conv_lstm_train.py", "Training ConvLSTM models"),
    ]

    for script, desc in steps:
        run_module(script, desc)

    if ENABLE_HYBRID:
        run_module("src/models/blend_hybrid.py", "Hybrid blending (RF+LGBM+ConvLSTM)")

    run_module("src/evaluation/evaluate_test_models.py", "Unified evaluation on test dataset")
    run_module("src/visualize/next_hour_map.py", "Generate next-hour demand map")

    export_forecast_report(BASE_DIR)

    logger.info(f"🏁 Pipeline completed (UTC: {utc_now()})")
