"""
Ingestion package interface.
Exposes the public functions used by the pipeline.
"""

from .load_data import get_raw_data_path

__all__ = ["get_raw_data_path"]
