import numpy as np


def apply_glare(
    image: np.ndarray,
    *,
    center: tuple[int, int],
    radius: int,
    intensity: float,
) -> np.ndarray:
    """
    Apply glare / lens flare effect as localized overexposure.

    Parameters
    ----------
    image : np.ndarray
        Input image in uint8 format (H, W, C).
    center : tuple[int, int]
        (x, y) coordinates of glare center.
    radius : int
        Radius of glare region in pixels.
    intensity : float
        Glare intensity multiplier (>1 increases brightness).

    Returns
    -------
    np.ndarray
        Image with glare applied.
    """
    if image.dtype != np.uint8:
        raise TypeError("Expected uint8 image")

    h, w = image.shape[:2]
    cx, cy = center

    if not (0 <= cx < w and 0 <= cy < h):
        raise ValueError("Glare center must be within image bounds")

    if radius <= 0:
        raise ValueError("Radius must be > 0")

    if intensity <= 1.0:
        raise ValueError("Intensity must be > 1.0")

    # Create glare mask
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    mask = np.clip(1.0 - (dist / radius), 0.0, 1.0)
    mask = mask[..., None]  # for broadcasting

    # Normalize image
    img = image.astype(np.float32)

    # Apply glare
    glare = img + intensity * mask * 255.0
    glare = np.clip(glare, 0, 255)

    return glare.astype(np.uint8)
