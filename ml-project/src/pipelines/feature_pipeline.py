"""
Run feature engineering pipeline:
- Load interim data
- Convert grayscale → RGB
- Normalize + convert to tensors
- Save to data/processed
"""

from src.utils.logger import setup_logger
from src.features.rbg_converter import convert_grayscale_to_rgb
from src.features.normalize import normalize_and_save
from src.config import load_yaml, PATHS_YAML


def main():
    logger = setup_logger("feature_engineering")
    logger.info("Starting feature engineering pipeline")

    try:
        # Load paths
        paths = load_yaml(PATHS_YAML)

        interim_root = paths["data"]["interim"]
        processed_root = paths["data"]["processed"]

        logger.info(f"Loading interim data from: {interim_root}")

        # Step 1: Grayscale → RGB
        rgb_data = convert_grayscale_to_rgb(interim_root)
        logger.info("Grayscale to RGB conversion completed")

        # Step 2: Normalize + save tensors
        normalize_and_save(
            rgb_data=rgb_data,
            output_root=processed_root,
        )
        logger.info("Normalization and tensor saving completed")

        logger.info("Feature engineering completed successfully")

    except Exception:
        logger.exception("Feature engineering failed")
        raise


if __name__ == "__main__":
    main()
