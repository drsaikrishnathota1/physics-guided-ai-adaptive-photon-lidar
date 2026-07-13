"""Atmospheric attenuation and distributed optical backscatter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


SPEED_OF_LIGHT_M_S = 299_792_458.0


@dataclass(frozen=True)
class AtmosphericChannel:
    """Homogeneous atmospheric optical channel."""

    extinction_coefficient_per_km: float = 0.10
    volume_backscatter_coefficient_per_m: float = 1.0e-7
    background_photons_per_bin_per_pulse: float = 1.0e-5

    def __post_init__(self) -> None:
        if self.extinction_coefficient_per_km < 0:
            raise ValueError(
                "extinction_coefficient_per_km cannot be negative."
            )

        if self.volume_backscatter_coefficient_per_m < 0:
            raise ValueError(
                "volume_backscatter_coefficient_per_m cannot be negative."
            )

        if self.background_photons_per_bin_per_pulse < 0:
            raise ValueError(
                "background_photons_per_bin_per_pulse cannot be negative."
            )


def two_way_transmission(
    range_m: float | NDArray[np.float64],
    extinction_coefficient_per_km: float,
) -> float | NDArray[np.float64]:
    """Return two-way Beer-Lambert transmission."""
    ranges = np.asarray(range_m, dtype=np.float64)

    if np.any(ranges < 0):
        raise ValueError("range_m cannot contain negative values.")

    if extinction_coefficient_per_km < 0:
        raise ValueError(
            "extinction_coefficient_per_km cannot be negative."
        )

    transmission = np.exp(
        -2.0
        * extinction_coefficient_per_km
        * ranges
        / 1000.0
    )

    if transmission.ndim == 0:
        return float(transmission)

    return transmission


def histogram_bin_ranges_m(
    num_bins: int,
    bin_width_ns: float,
) -> NDArray[np.float64]:
    """Return the one-way range associated with each histogram-bin center."""
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")

    if bin_width_ns <= 0:
        raise ValueError("bin_width_ns must be positive.")

    bin_width_s = bin_width_ns * 1.0e-9
    times_s = (np.arange(num_bins, dtype=np.float64) + 0.5) * bin_width_s

    return SPEED_OF_LIGHT_M_S * times_s / 2.0


def distributed_backscatter_histogram(
    *,
    num_bins: int,
    bin_width_ns: float,
    total_transmitted_photons: float,
    transmitter_efficiency: float,
    receiver_aperture_area_m2: float,
    receiver_efficiency: float,
    channel: AtmosphericChannel,
) -> NDArray[np.float64]:
    """Calculate expected distributed atmospheric backscatter.

    A simplified monostatic elastic-backscatter relationship is used:

        N_b(r) proportional to
            N_tx * beta * dr * A_r * T²(r) / (4*pi*r²)

    The near-field range is clamped to avoid a singularity.
    """
    if total_transmitted_photons < 0:
        raise ValueError(
            "total_transmitted_photons cannot be negative."
        )

    ranges_m = histogram_bin_ranges_m(
        num_bins=num_bins,
        bin_width_ns=bin_width_ns,
    )

    effective_ranges_m = np.maximum(ranges_m, 1.0)

    range_increment_m = (
        SPEED_OF_LIGHT_M_S
        * bin_width_ns
        * 1.0e-9
        / 2.0
    )

    transmission = two_way_transmission(
        effective_ranges_m,
        channel.extinction_coefficient_per_km,
    )

    expected = (
        total_transmitted_photons
        * transmitter_efficiency
        * channel.volume_backscatter_coefficient_per_m
        * range_increment_m
        * receiver_aperture_area_m2
        * receiver_efficiency
        * transmission
        / (4.0 * np.pi * effective_ranges_m**2)
    )

    return np.asarray(expected, dtype=np.float64)
