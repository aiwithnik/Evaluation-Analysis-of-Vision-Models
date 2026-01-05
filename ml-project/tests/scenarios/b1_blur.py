from tests.degradation.blur import apply_gaussian_blur


def scenario_b1_blur(image, *, sigma: float):
    """
    B1: Gaussian blur scenario.
    """
    return apply_gaussian_blur(image, sigma=sigma)
