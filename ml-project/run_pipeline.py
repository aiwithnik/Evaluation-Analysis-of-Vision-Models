"""
Run the complete ML pipeline end-to-end.
"""

from src.utils.logger import setup_logger
from src.pipelines import (
    run_ingestion,
    run_preprocessing,
    run_feature_engineering,
    run_training,
    run_evaluation,
)


def main():
    logger = setup_logger("full_pipeline")
    logger.info("Starting full ML pipeline")

    try:
        logger.info("Step 1: Ingestion")
        run_ingestion()

        logger.info("Step 2: Preprocessing")
        run_preprocessing()

        logger.info("Step 3: Feature Engineering")
        run_feature_engineering()

        logger.info("Step 4: Training")
        run_training()

        logger.info("Step 5: Evaluation")
        run_evaluation()

        logger.info("Full pipeline completed successfully")

    except Exception:
        logger.exception("Pipeline failed")
        raise


if __name__ == "__main__":
    main()
