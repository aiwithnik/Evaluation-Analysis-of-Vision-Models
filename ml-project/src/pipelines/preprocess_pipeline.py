"""
Run pipeline up to preprocessing to validate correctness.
"""

from src.utils.logger import setup_logger
from src.ingestion import get_raw_data_path
from src.preprocessing import standardize_images, split_and_save
from src.config import load_yaml, CONFIG_YAML, PATHS_YAML


def main():
    logger = setup_logger("preprocessing_check")
    logger.info("Starting preprocessing pipeline check")

    try:
        # Load configs
        config = load_yaml(CONFIG_YAML)
        paths = load_yaml(PATHS_YAML)

        logger.info("Configs loaded successfully")

        # Ingestion
        raw_path = get_raw_data_path()
        logger.info(f"Raw data path located: {raw_path}")

        # Standardization
        images, labels = standardize_images(raw_path)
        logger.info(f"Standardized images loaded: {len(images)}")

        # Split + Save
        split_cfg = config["data_split"]

        split_and_save(
            images=images,
            labels=labels,
            output_root=paths["data"]["interim"],
            train_size=split_cfg["train_size"],
            val_size=split_cfg["val_size"],
            test_size=split_cfg["test_size"],
            random_seed=config["random_seed"],
        )

        logger.info("Preprocessing completed successfully")

    except Exception as e:
        logger.exception("Preprocessing failed")
        raise e


if __name__ == "__main__":
    main()
