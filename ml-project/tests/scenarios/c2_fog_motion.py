from tests.degradation.fog import apply_fog
from tests.degradation.motion_blur import apply_motion_blur


def scenario_c2_fog_motion(
    image,
    *,
    fog_strength: float,
    kernel_size: int,
    angle: float,
):
    """
    C2: Fog + motion blur scenario.

    Simulates poor visibility with camera motion.
    """
    image = apply_fog(image, fog_strength=fog_strength)
    image = apply_motion_blur(
        image,
        kernel_size=kernel_size,
        angle=angle,
    )

    return image
