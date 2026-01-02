import json

from src.utils.logger import setup_logger
from src.models.train import train_model
from src.config import load_yaml, PATHS_YAML, MODEL_PARAMS_YAML


def main():
    logger = setup_logger("training")
    logger.info("Starting training pipeline")

    # Load configs
    paths = load_yaml(PATHS_YAML)
    params = load_yaml(MODEL_PARAMS_YAML)

    train_cfg = params["training"]
    optim_cfg = params["optimizer"]

    # Load class mapping
    with open("data/processed/class_to_idx.json") as f:
        class_to_idx = json.load(f)

    num_classes = len(class_to_idx)

    logger.info("Launching training")

    train_model(
        train_dir=f"{paths['data']['processed']}/train",
        val_dir=f"{paths['data']['processed']}/val",
        num_classes=num_classes,
        epochs=train_cfg["epochs"],
        batch_size=train_cfg["batch_size"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=optim_cfg["weight_decay"],
        device=train_cfg["device"],
    )

    logger.info("Training pipeline completed successfully")


if __name__ == "__main__":
    main()
