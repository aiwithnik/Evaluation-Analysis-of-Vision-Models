from tests.degradation.low_light import apply_low_light


def scenario_b3_low_light(image, *, gamma: float):
    """
    B3: Low-light robustness scenario.

    Applies gamma-based illumination reduction.
    """
    return apply_low_light(image, gamma=gamma)
