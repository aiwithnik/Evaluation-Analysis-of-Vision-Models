import numpy as np


def apply_fog(
    image: np.ndarray,
    *,
    fog_strength: float,
    atmospheric_light: float = 1.0,
) -> np.ndarray:
    """
    Apply fog / haze degradation using atmospheric scattering model.

    Parameters
    ----------
    image : np.ndarray
        Input image in uint8 format (H, W, C).
    fog_strength : float
        Fog density in [0, 1].
        0.0 = no fog, 1.0 = very dense fog.
    atmospheric_light : float
        Intensity of atmospheric light (default = 1.0 → white fog).

    Returns
    -------
    np.ndarray
        Fog-degraded image in uint8 format.
    """
    if not (0.0 <= fog_strength <= 1.0):
        raise ValueError("fog_strength must be in [0, 1]")

    if image.dtype != np.uint8:
        raise TypeError("Expected uint8 image")

    # Normalize image
    img = image.astype(np.float32) / 255.0

    # Transmission factor (lower = denser fog)
    transmission = 1.0 - fog_strength

    # Atmospheric light (white/gray fog)
    A = np.ones_like(img) * atmospheric_light

    # Apply scattering model
    foggy = img * transmission + A * (1.0 - transmission)

    foggy = (foggy * 255.0).clip(0, 255)

    return foggy.astype(np.uint8)
