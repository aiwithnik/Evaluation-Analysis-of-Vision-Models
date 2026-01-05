from tests.degradation.glare import apply_glare


def scenario_c5_glare(
    image,
    *,
    center: tuple[int, int],
    radius: int,
    intensity: float,
):
    """
    C5: Glare / lens flare robustness scenario.

    Simulates localized sensor saturation from strong light sources.
    """
    return apply_glare(
        image,
        center=center,
        radius=radius,
        intensity=intensity,
    )
