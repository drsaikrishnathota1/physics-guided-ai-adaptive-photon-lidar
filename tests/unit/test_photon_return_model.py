"""Unit tests for the single-photon LiDAR forward model."""

import numpy as np
import pytest

from src.physics.photon_return_model import (
    AtmosphereParameters,
    DetectorParameters,
    HistogramParameters,
    LaserParameters,
    ReceiverParameters,
    TargetParameters,
    expected_detected_signal_photons,
    gaussian_instrument_response,
    round_trip_time,
    simulate_photon_return,
    two_way_atmospheric_transmission,
)


def test_round_trip_time_is_positive() -> None:
    value = round_trip_time(1000.0)

    assert value > 0.0
    assert np.isclose(value, 2.0 * 1000.0 / 299_792_458.0)


def test_atmospheric_transmission_decreases_with_range() -> None:
    near = two_way_atmospheric_transmission(
        range_m=500.0,
        extinction_coefficient_per_km=0.2,
    )

    far = two_way_atmospheric_transmission(
        range_m=2000.0,
        extinction_coefficient_per_km=0.2,
    )

    assert 0.0 < far < near <= 1.0


def test_signal_photons_decrease_with_range() -> None:
    common = {
        "laser": LaserParameters(emitted_photons=1e12),
        "receiver": ReceiverParameters(),
        "detector": DetectorParameters(),
        "atmosphere": AtmosphereParameters(),
    }

    near = expected_detected_signal_photons(
        **common,
        target=TargetParameters(range_m=500.0),
    )

    far = expected_detected_signal_photons(
        **common,
        target=TargetParameters(range_m=1500.0),
    )

    assert near > far >= 0.0


def test_instrument_response_is_normalized() -> None:
    response = gaussian_instrument_response(
        target_bin=100,
        histogram=HistogramParameters(
            num_bins=256,
            bin_width_ns=1.0,
        ),
        detector=DetectorParameters(),
        laser=LaserParameters(),
    )

    assert np.isclose(response.sum(), 1.0)
    assert int(np.argmax(response)) == 100


def test_simulation_is_reproducible_with_fixed_seed() -> None:
    parameters = {
        "laser": LaserParameters(emitted_photons=1e12),
        "receiver": ReceiverParameters(),
        "detector": DetectorParameters(),
        "atmosphere": AtmosphereParameters(),
        "target": TargetParameters(range_m=1000.0),
        "histogram": HistogramParameters(
            num_bins=4096,
            bin_width_ns=5.0,
        ),
    }

    first = simulate_photon_return(
        **parameters,
        rng=np.random.default_rng(42),
    )

    second = simulate_photon_return(
        **parameters,
        rng=np.random.default_rng(42),
    )

    assert np.array_equal(first.photon_counts, second.photon_counts)
    assert first.target_bin == second.target_bin


def test_out_of_window_target_raises_error() -> None:
    with pytest.raises(ValueError):
        simulate_photon_return(
            laser=LaserParameters(),
            receiver=ReceiverParameters(),
            detector=DetectorParameters(),
            atmosphere=AtmosphereParameters(),
            target=TargetParameters(range_m=10_000.0),
            histogram=HistogramParameters(
                num_bins=100,
                bin_width_ns=1.0,
            ),
            rng=np.random.default_rng(42),
        )
