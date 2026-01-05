from pathlib import Path


def validate_evaluation_inputs(model_registry_root, test_data_dir):
    model_registry_root = Path(model_registry_root)
    test_data_dir = Path(test_data_dir)

    # Check registry root
    if not model_registry_root.exists():
        raise FileNotFoundError(f"Model registry not found: {model_registry_root}")

    # Check for latest model
    latest_model = model_registry_root / "latest" / "model.pt"

    # Check for versioned models
    versioned_models = list(
        model_registry_root.glob("v*/model.pt")
    )

    if not latest_model.exists() and not versioned_models:
        raise FileNotFoundError(
            "No model.pt found in registry.\n"
            "Expected one of:\n"
            "- model/registry/latest/model.pt\n"
            "- model/registry/vXXX/model.pt"
        )

    # Check test data
    if not test_data_dir.exists():
        raise FileNotFoundError(f"Test data directory not found: {test_data_dir}")

    test_files = list(test_data_dir.glob("*.pt"))
    if len(test_files) == 0:
        raise RuntimeError("No test tensor files found")

    return True
