import numpy as np
import cv2


def apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur to an image.

    Parameters
    ----------
    image : np.ndarray
        Input image as a NumPy array (H, W, C) or (H, W).
        Expected dtype: uint8 or float32.
    sigma : float
        Standard deviation of the Gaussian kernel.

    Returns
    -------
    np.ndarray
        Blurred image with the same shape and dtype as input.
    """
    if sigma <= 0:
        raise ValueError("Sigma must be > 0")

    if image.ndim not in {2, 3}:
        raise ValueError(
            f"Expected 2D or 3D image array, got shape {image.shape}"
        )

    # Kernel size derived from sigma (OpenCV convention)
    ksize = int(2 * round(3 * sigma) + 1)

    blurred = cv2.GaussianBlur(
        image,
        ksize=(ksize, ksize),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )

    return blurred
