"""Integration tests for expected LiDAR physical trends."""

from src.physics.photon_return_model import (
    AtmosphereParameters,
    DetectorParameters,
    LaserParameters,
    ReceiverParameters,
    TargetParameters,
    expected_detected_signal_photons,
)


def calculate_signal(
    range_m: float,
    reflectivity: float,
    extinction: float,
) -> float:
    """Calculate expected signal photons for a test condition."""
    return expected_detected_signal_photons(
        laser=LaserParameters(emitted_photons=1e12),
        receiver=ReceiverParameters(),
        detector=DetectorParameters(),
        atmosphere=AtmosphereParameters(
            extinction_coefficient_per_km=extinction
        ),
        target=TargetParameters(
            range_m=range_m,
            reflectivity=reflectivity,
        ),
    )


def test_signal_decreases_with_range() -> None:
    near_signal = calculate_signal(
        range_m=500.0,
        reflectivity=0.20,
        extinction=0.10,
    )

    far_signal = calculate_signal(
        range_m=2000.0,
        reflectivity=0.20,
        extinction=0.10,
    )

    assert near_signal > far_signal


def test_signal_decreases_with_extinction() -> None:
    clear_signal = calculate_signal(
        range_m=1000.0,
        reflectivity=0.20,
        extinction=0.05,
    )

    obscured_signal = calculate_signal(
        range_m=1000.0,
        reflectivity=0.20,
        extinction=1.00,
    )

    assert clear_signal > obscured_signal


def test_signal_increases_with_reflectivity() -> None:
    low_reflectivity_signal = calculate_signal(
        range_m=1000.0,
        reflectivity=0.05,
        extinction=0.10,
    )

    high_reflectivity_signal = calculate_signal(
        range_m=1000.0,
        reflectivity=0.60,
        extinction=0.10,
    )

    assert high_reflectivity_signal > low_reflectivity_signal
