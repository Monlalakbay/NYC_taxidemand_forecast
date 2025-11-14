import os
import requests
import pandas as pd
from datetime import datetime
from src.utils.pipeline_logger import init_logger, log_pipeline_step

logger = init_logger(__name__)

def fetch_latest_data(api_url, save_dir):
    """Fetches latest taxi trip data from API and saves as a CSV or Parquet."""
    logger.info("Fetching from API: %s", api_url)

    # Make API call
    resp = requests.get(api_url)

    if resp.status_code != 200:
        logger.error("API error: %s", resp.status_code)
        raise ConnectionError(f"API failed ({resp.status_code})")

    # Convert JSON → DataFrame (assuming JSON response)
    df = pd.DataFrame(resp.json())

    # Save file
    os.makedirs(save_dir, exist_ok=True)

    fname = f"fhv_tripdata_{datetime.utcnow().strftime('%Y%m%d')}.csv"
    path = os.path.join(save_dir, fname)
    df.to_csv(path, index=False)

    logger.info("Saved API dataset: %s", path)
    log_pipeline_step("Fetch API data", "success", details=f"file={fname}")

    return path


# ===========================
# Script Entry Point
# ===========================
if __name__ == "__main__":
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    raw_dir = os.path.join(base, "data", "raw", "pickup_data")

    # Example placeholder API
    API_URL = "https://data.city/api/v1/fhv_tripdata?month=2025-11"
    fetch_latest_data(API_URL, raw_dir)
