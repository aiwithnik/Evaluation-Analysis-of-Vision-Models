from tests.degradation.low_light import apply_low_light
from tests.degradation.fog import apply_fog
from tests.degradation.noise import apply_gaussian_noise


def scenario_d3_extreme(
    image,
    *,
    gamma: float,
    fog_strength: float,
    noise_std: float,
):
    """
    D3: Extreme degradation scenario.

    Combines:
    - severe low-light
    - dense fog
    - strong sensor noise

    This scenario is intended to probe model failure boundaries.
    """
    # 1. Illumination collapse
    image = apply_low_light(image, gamma=gamma)

    # 2. Atmospheric scattering
    image = apply_fog(image, fog_strength=fog_strength)

    # 3. Sensor noise amplification
    image = apply_gaussian_noise(image, std=noise_std)

    return image
