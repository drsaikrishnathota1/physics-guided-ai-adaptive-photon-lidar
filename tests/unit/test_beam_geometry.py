from src.physics.beam_geometry import (
    BeamParameters,
    beam_spot_area_m2,
    target_intercept_fraction,
)


def test_beam_spot_increases_with_range() -> None:
    beam = BeamParameters(full_angle_divergence_mrad=0.5)

    assert beam_spot_area_m2(2000.0, beam) > beam_spot_area_m2(
        500.0,
        beam,
    )


def test_intercept_fraction_is_bounded() -> None:
    fraction = target_intercept_fraction(
        range_m=1000.0,
        projected_target_area_m2=1.0,
        beam=BeamParameters(),
    )

    assert 0.0 <= fraction <= 1.0
