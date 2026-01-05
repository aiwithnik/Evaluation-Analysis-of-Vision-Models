import numpy as np


def apply_gaussian_noise(
    image: np.ndarray,
    *,
    std: float,
) -> np.ndarray:
    """
    Apply additive Gaussian noise to an image.

    Parameters
    ----------
    image : np.ndarray
        Input image in uint8 format (H, W, C).
    std : float
        Standard deviation of noise in [0, 1] scale.
        Typical values: 0.02 – 0.15

    Returns
    -------
    np.ndarray
        Noisy image in uint8 format.
    """
    if std <= 0:
        raise ValueError("std must be > 0")

    if image.dtype != np.uint8:
        raise TypeError("Expected uint8 image")

    # Normalize to [0, 1]
    img = image.astype(np.float32) / 255.0

    # Gaussian noise
    noise = np.random.normal(
        loc=0.0,
        scale=std,
        size=img.shape,
    )

    noisy = img + noise
    noisy = np.clip(noisy, 0.0, 1.0)

    # Back to uint8
    noisy = (noisy * 255.0).astype(np.uint8)

    return noisy
