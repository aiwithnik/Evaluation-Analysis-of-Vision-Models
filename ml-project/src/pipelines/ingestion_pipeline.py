"""
Ingestion pipeline.

Since data is already present on disk,
this pipeline only validates the raw data location.
"""

from src.utils.logger import setup_logger
from src.ingestion.load_data import get_raw_data_path


def main():
    logger = setup_logger("ingestion")
    logger.info("Starting ingestion pipeline")

    raw_path = get_raw_data_path()
    logger.info(f"Raw data found at: {raw_path}")

    logger.info("Ingestion pipeline completed successfully")
