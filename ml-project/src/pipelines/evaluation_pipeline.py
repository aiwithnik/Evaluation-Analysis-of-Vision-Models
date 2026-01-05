import json
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from src.utils.logger import setup_logger
from src.config import load_yaml, PATHS_YAML, MODEL_PARAMS_YAML
from src.models.dataset_loader import TensorDataset
from src.models.model import build_efficientnet
from src.models.registry import load_model

from src.evaluation import (
    validate_evaluation_inputs,
    compute_classification_metrics,
    plot_confusion_matrix,
)

from src.evaluation.reporting import (
    create_report_dir,
    save_metrics,
    save_report_metadata,
)


def main():
    logger = setup_logger("evaluation")
    logger.info("Starting evaluation pipeline")

    # ------------------------------------------------------------------
    # Load configs
    # ------------------------------------------------------------------
    paths = load_yaml(PATHS_YAML)
    params = load_yaml(MODEL_PARAMS_YAML)

    device = params["training"]["device"]
    batch_size = params["training"]["batch_size"]

    # ------------------------------------------------------------------
    # Load class mapping
    # ------------------------------------------------------------------
    with open("data/processed/class_to_idx.json") as f:
        class_to_idx = json.load(f)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    model_registry_root = paths["models"]["root"]
    test_dir = f"{paths['data']['processed']}/test"

    validate_evaluation_inputs(model_registry_root, test_dir)
    logger.info("Evaluation inputs validated")

    # ------------------------------------------------------------------
    # Create report directory
    # ------------------------------------------------------------------
    report_dir = create_report_dir()
    logger.info(f"Writing evaluation report to {report_dir}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    test_ds = TensorDataset(test_dir)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_efficientnet(num_classes=len(class_names))
    model = load_model(model)
    model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device).float()
            labels = labels.to(device)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_prob.extend(probs.cpu().tolist())

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    results = compute_classification_metrics(y_true, y_pred)
    save_metrics(report_dir, results)

    logger.info(f"Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Macro Precision: {results['macro_precision']:.4f}")
    logger.info(f"Macro Recall: {results['macro_recall']:.4f}")
    logger.info(f"Macro F1: {results['macro_f1']:.4f}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        save_path=report_dir / "confusion_matrix.png",
    )

    # ------------------------------------------------------------------
    # Report metadata
    # ------------------------------------------------------------------
    save_report_metadata(
        report_dir,
        model_version="latest",
        evaluation_type="baseline_clean",
        params={
            "dataset_split": "test",
            "num_classes": len(class_names),
        },
    )

    logger.info("Evaluation pipeline completed successfully")


if __name__ == "__main__":
    main()
