from tests.degradation.fog import apply_fog


def scenario_b5_fog(image, *, fog_strength: float):
    """
    B5: Fog / haze robustness scenario.

    Applies atmospheric scattering-based fog.
    """
    return apply_fog(
        image,
        fog_strength=fog_strength,
        atmospheric_light=1.0,
    )
