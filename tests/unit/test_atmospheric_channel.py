import numpy as np

from src.physics.atmospheric_channel import (
    AtmosphericChannel,
    distributed_backscatter_histogram,
    two_way_transmission,
)


def test_transmission_decreases_with_range() -> None:
    assert two_way_transmission(2000.0, 0.2) < two_way_transmission(
        500.0,
        0.2,
    )


def test_backscatter_profile_is_finite() -> None:
    values = distributed_backscatter_histogram(
        num_bins=512,
        bin_width_ns=5.0,
        total_transmitted_photons=1.0e10,
        transmitter_efficiency=0.7,
        receiver_aperture_area_m2=0.01,
        receiver_efficiency=0.5,
        channel=AtmosphericChannel(),
    )

    assert values.shape == (512,)
    assert np.all(np.isfinite(values))
    assert np.all(values >= 0.0)
