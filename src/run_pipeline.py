# src/run_pipeline.py — Unified Pipeline Runner 

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
ENABLE_HYBRID = True  # toggle hybrid blending


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ------------------------------------------------------------
# Spinner / progress bar
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
# Run module
# ------------------------------------------------------------
def run_module(module_name: str, description: str):
    logger.info("=" * 80)
    logger.info(f"▶️  {description}")
    logger.info("=" * 80)

    start = time.time()
    status = "success"

    cmd = [sys.executable, "-m", module_name]
    result = run_with_progress(cmd, description)

    duration = round((time.time() - start) / 60, 2)

    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.warning("stderr:\n%s", result.stderr)

    if result.returncode != 0:
        status = f"error: returncode {result.returncode}"
        logger.error(f"❌ Step failed with code {result.returncode}")
        log_pipeline_step(description, status, duration_min=duration)
        sys.exit(1)

    logger.info(f"✅ Completed {description} in {duration} min.")
    log_pipeline_step(description, status, duration_min=duration)


# ------------------------------------------------------------
# Export forecast file
# ------------------------------------------------------------
def export_forecast_report(base_dir):
    logger.info("📈 Exporting latest forecast report...")

    eval_dir = os.path.join(base_dir, "data", "evaluation")

    summary_files = sorted(
        f for f in os.listdir(eval_dir)
        if f.endswith(".csv") and "all_models_eval" in f
    )

    if not summary_files:
        logger.warning("⚠️ No evaluation summary found — skipping export.")
        return

    latest = summary_files[-1]
    src_path = os.path.join(eval_dir, latest)
    out_path = os.path.join(eval_dir, "forecast_latest.csv")

    import pandas as pd
    df = pd.read_csv(src_path)
    df.to_csv(out_path, index=False)

    logger.info(f"✅ Forecast exported: {out_path}")
    log_pipeline_step("Export forecast report", "success", details=f"file={out_path}")


# ------------------------------------------------------------
# Pipeline Steps
# ------------------------------------------------------------
if __name__ == "__main__":
    logger.info(f"🚀 Starting pipeline (UTC: {utc_now()})")
    logger.info(f"📂 Project root: {BASE_DIR}")

    steps = [
        ("src.data.clean_data", "Cleaning raw taxi pickup files"),
        ("src.data.preprocess", "Preprocessing taxi pickup data"),
        ("src.data.split_data", "Splitting & scaling dataset"),
        ("src.models.modeling_districts", "Training ensemble (RF + LGBM)"),
        ("src.models.modeling_conv_lstm_train", "Training ConvLSTM models"),
    ]

    for module, desc in steps:
        run_module(module, desc)

    if ENABLE_HYBRID:
        run_module("src.models.blend_hybrid", "Hybrid blending (RF + LGBM + ConvLSTM)")

    run_module("src.evaluation.evaluate_test_models", "Unified evaluation on test dataset")
    run_module("src.visualize.next_hour_map", "Generate next-hour demand map")

    export_forecast_report(BASE_DIR)

    logger.info(f"🏁 Pipeline completed (UTC: {utc_now()})")
