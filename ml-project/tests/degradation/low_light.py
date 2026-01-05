import numpy as np


def apply_low_light(image: np.ndarray, gamma: float) -> np.ndarray:
    """
    Simulate low-light conditions using gamma correction.

    Parameters
    ----------
    image : np.ndarray
        Input image in uint8 format with values in [0, 255].
        Shape: (H, W, C) or (H, W)
    gamma : float
        Gamma value (< 1 darkens the image).
        Typical values: 0.1 – 0.6

    Returns
    -------
    np.ndarray
        Low-light image with same shape and dtype as input.
    """
    if gamma <= 0:
        raise ValueError("Gamma must be > 0")

    if image.dtype != np.uint8:
        raise TypeError(
            f"Expected uint8 image, got {image.dtype}"
        )

    # Normalize to [0, 1]
    img_norm = image.astype(np.float32) / 255.0

    # Apply gamma correction
    img_low_light = np.power(img_norm, gamma)

    # Rescale back to [0, 255]
    img_low_light = (img_low_light * 255.0).clip(0, 255)

    return img_low_light.astype(np.uint8)
