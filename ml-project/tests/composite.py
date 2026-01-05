from typing import Callable
import numpy as np


def apply_scenario(
    image: np.ndarray,
    scenario_fn: Callable,
    *,
    params: dict,
) -> np.ndarray:
    """
    Apply a degradation scenario to a single image.

    Parameters
    ----------
    image : np.ndarray
        Input image (uint8).
    scenario_fn : Callable
        Scenario function (e.g., scenario_b1_blur).
    params : dict
        Keyword arguments required by the scenario.

    Returns
    -------
    np.ndarray
        Degraded image.
    """
    if not isinstance(params, dict):
        raise TypeError("params must be a dictionary")

    try:
        degraded = scenario_fn(image, **params)
    except TypeError as e:
        raise TypeError(
            f"Scenario parameter mismatch: {e}"
        ) from e

    if not isinstance(degraded, np.ndarray):
        raise TypeError(
            "Scenario must return a numpy.ndarray"
        )

    return degraded
