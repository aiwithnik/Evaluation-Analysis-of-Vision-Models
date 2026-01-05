from tests.degradation.low_light import apply_low_light
from tests.degradation.noise import apply_gaussian_noise


def scenario_c1_night(
    image,
    *,
    gamma: float,
    noise_std: float,
):
    """
    C1: Night driving scenario.

    Low-light illumination followed by sensor noise.
    """
    image = apply_low_light(image, gamma=gamma)
    image = apply_gaussian_noise(image, std=noise_std)

    return image
