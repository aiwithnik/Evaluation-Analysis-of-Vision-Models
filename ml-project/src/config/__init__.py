"""
This file tells Python that the `config` directory is a package.

In simple words:
- It allows other files to IMPORT things from `src.config`
- Without this file, Python may treat this folder as just a normal directory

Even if this file is empty, its presence matters.
"""


# Optional but useful:
# We import helper functions here so that they can be accessed
# directly from `src.config` instead of long import paths.


from pathlib import Path
import yaml


def load_yaml(file_path: str):
    """
    Load a YAML configuration file and return it as a Python dictionary.

    Parameters:
    ----------
    file_path : str
        Path to the YAML file (example: 'src/config/config.yaml')

    Returns:
    -------
    dict
        Parsed YAML content as a Python dictionary
    """
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


# These constants are OPTIONAL.
# They just make paths easier to manage and less error-prone.

CONFIG_DIR = Path(__file__).parent
CONFIG_YAML = CONFIG_DIR / "config.yaml"
PATHS_YAML = CONFIG_DIR / "paths.yaml"
MODEL_PARAMS_YAML = CONFIG_DIR / "model_params.yaml"
