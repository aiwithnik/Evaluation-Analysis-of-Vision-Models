import numpy as np
import cv2


def apply_motion_blur(
    image: np.ndarray,
    *,
    kernel_size: int,
    angle: float,
) -> np.ndarray:
    """
    Apply motion blur to an image using a linear kernel.

    Parameters
    ----------
    image : np.ndarray
        Input image in uint8 format (H, W, C).
    kernel_size : int
        Length of the motion blur kernel (odd integer >= 3).
    angle : float
        Direction of motion blur in degrees (0 = horizontal).

    Returns
    -------
    np.ndarray
        Motion-blurred image in uint8 format.
    """
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer >= 3")

    if image.dtype != np.uint8:
        raise TypeError("Expected uint8 image")

    # Create motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0

    # Rotate kernel to desired angle
    rotation_matrix = cv2.getRotationMatrix2D(
        (kernel_size / 2, kernel_size / 2),
        angle,
        1.0,
    )
    kernel = cv2.warpAffine(
        kernel,
        rotation_matrix,
        (kernel_size, kernel_size),
    )

    kernel /= kernel.sum()

    # Apply convolution
    blurred = cv2.filter2D(
        image,
        -1,
        kernel,
        borderType=cv2.BORDER_REFLECT_101,
    )

    return blurred
