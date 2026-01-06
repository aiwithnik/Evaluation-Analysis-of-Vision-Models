"""
Unified experiment runner.

Responsibilities:
- Load interim test data (image space)
- Apply degradations / scenarios
- Run model inference
- Compute metrics + plots
- Save everything under experiments/expXXX/
"""

from pathlib import Path
from PIL import Image
import json
import random
import numpy as np
import torch
import torch.nn.functional as F

from tests.ingestion import get_interim_split_path
from tests.composite import apply_scenario
from tests.utils.logger import setup_logger
from tests.utils.experiments import (
    create_experiment_dir,
    finalize_experiment,
)

from tests.evaluation.metrics import (
    compute_classification_metrics,
    compute_roc_auc_metrics,
)
from tests.evaluation.plots import (
    plot_confusion_matrix,
    plot_roc_curves,
)

from tests.ingestion.image_folder_dataset import ImageFolderDataset
from src.models.model import build_efficientnet
from src.models.registry import load_model
from src.config import load_yaml, PATHS_YAML, MODEL_PARAMS_YAML


# ---------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------
from tests.scenarios.b1_blur import scenario_b1_blur
from tests.scenarios.b3_low_light import scenario_b3_low_light
from tests.scenarios.b5_fog import scenario_b5_fog
from tests.scenarios.c1_night import scenario_c1_night
from tests.scenarios.c2_fog_motion import scenario_c2_fog_motion
from tests.scenarios.c5_glare import scenario_c5_glare
from tests.scenarios.d5_extreme import scenario_d3_extreme

SCENARIO_REGISTRY = {
    "B1": scenario_b1_blur,
    "B3": scenario_b3_low_light,
    "B5": scenario_b5_fog,
    "C1": scenario_c1_night,
    "C2": scenario_c2_fog_motion,
    "C5": scenario_c5_glare,
    "D3": scenario_d3_extreme,
}


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------
def run_experiment(
    *,
    scenario_code: str,
    scenario_params: dict,
    experiment_name: str,
    family: str,
    description: str,
    seed: int = 42,
):
    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------
    set_seed(seed)

    paths = load_yaml(PATHS_YAML)
    params = load_yaml(MODEL_PARAMS_YAML)

    device = params["training"]["device"]
    batch_size = params["training"]["batch_size"]

    exp_dir = create_experiment_dir(
        Path("experiments"),
        experiment_name=experiment_name,
        family=family,
        scenario=scenario_code,
        model_version="latest",
        degradations=[scenario_params],
        execution={
            "seed": seed,
            "batch_size": batch_size,
            "device": device,
        },
        description=description,
    )

    logger = setup_logger("experiment", exp_dir / "logs")
    logger.info(f"Starting experiment {exp_dir.name}")
    logger.info(f"Scenario {scenario_code} | Params {scenario_params}")

    # ------------------------------------------------------------
    # Sample saving setup
    # ------------------------------------------------------------
    MAX_SAMPLES_TO_SAVE = 30
    saved_samples = 5

    samples_dir = exp_dir / "samples"
    clean_dir = samples_dir / "clean"
    degraded_dir = samples_dir / "degraded"
    clean_dir.mkdir(parents=True, exist_ok=True)
    degraded_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Load data (IMAGE SPACE)
    # ------------------------------------------------------------
    test_path = get_interim_split_path("test")
    dataset = ImageFolderDataset(test_path)

    # Load class names
    with open("data/processed/class_to_idx.json") as f:
        class_to_idx = json.load(f)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------
    model = build_efficientnet(num_classes=len(class_names))
    model = load_model(model)
    model.to(device)
    model.eval()

    scenario_fn = SCENARIO_REGISTRY[scenario_code]

    # ------------------------------------------------------------
    # Evaluation loop (manual batching)
    # ------------------------------------------------------------
    y_true, y_pred, y_prob = [], [], []

    for start in range(0, len(dataset), batch_size):
        imgs_np = []
        labels = []

        for idx in range(start, min(start + batch_size, len(dataset))):
            img, label = dataset[idx]  # uint8 HWC

            degraded_img = apply_scenario(
                img,
                scenario_fn=scenario_fn,
                params=scenario_params,
            )

            # Save sample images (limited, deterministic)
            if saved_samples < MAX_SAMPLES_TO_SAVE:
                img_id = f"img_{saved_samples:04d}.png"
                Image.fromarray(img).save(clean_dir / img_id)
                Image.fromarray(degraded_img).save(degraded_dir / img_id)
                saved_samples += 1

            imgs_np.append(degraded_img)
            labels.append(label)

        # Convert to tensor: (N, H, W, C) → (N, C, H, W)
        imgs_np = np.stack(imgs_np, axis=0)
        imgs = torch.from_numpy(imgs_np).float()
        imgs = imgs.permute(0, 3, 1, 2) / 255.0
        imgs = imgs.to(device)

        labels = torch.tensor(labels, device=device)

        with torch.no_grad():
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        y_prob.extend(probs.cpu().tolist())

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------
    metrics = compute_classification_metrics(y_true, y_pred)
    auc_metrics = compute_roc_auc_metrics(
        y_true,
        y_prob,
        num_classes=len(class_names),
    )

    all_metrics = {**metrics, **auc_metrics}

    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)

    # ------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------
    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        save_path=exp_dir / "confusion_matrix.png",
    )

    plot_roc_curves(
        y_true,
        y_prob,
        class_names,
        save_path=exp_dir / "roc_curves.png",
    )

    # ------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------
    finalize_experiment(exp_dir)
    logger.info("Experiment completed successfully")


# ---------------------------------------------------------------------
# Example run
# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_experiment(
        scenario_code="D3",
        scenario_params={
            "gamma": 0.15,
            "fog_strength": 0.9,
            "noise_std": 0.12
        },
        experiment_name="d_extreme_gamma0.15_fog0.9_noise0.12",
        family="D",
        description="extreme robustness gamma=0.15, fog=0.9, noise=0.12",
    )
