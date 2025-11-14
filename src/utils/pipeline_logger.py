# src/utils/pipeline_logger.py

import os
import csv
import logging
from datetime import datetime, timezone

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG_FILE = os.path.join(BASE_DIR, "data", "pipeline_log.csv")


def utc_now():
    """Return current UTC timestamp in human-readable string format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def init_logger(module_name: str):
    """
    Create a unified console logger for any module.
    Usage: logger = init_logger(__name__)
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

    return logger


def log_pipeline_step(step: str, status: str, duration_min: float = None, details: str = None):
    """
    Append a standardized entry to the unified pipeline log (UTC).
    """
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    entry = {
        "timestamp_utc": utc_now(),
        "step": step,
        "status": status,
        "duration_min": duration_min if duration_min is not None else "",
        "details": details if details else "",
    }

    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=entry.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(entry)
