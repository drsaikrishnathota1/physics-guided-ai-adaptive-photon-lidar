"""Physics-grounded photon-return model for pulsed single-photon LiDAR.

The module simulates time-resolved photon-count histograms using:

- inverse-square geometric spreading;
- two-way Beer-Lambert atmospheric attenuation;
- target reflectivity;
- receiver aperture and optical efficiency;
- detector photon-detection efficiency;
- Gaussian instrument-response timing jitter;
- solar/background photons;
- detector dark counts;
- Poisson photon-arrival statistics.

This is an initial research simulator. Parameter values must later be justified
using experimental literature or manufacturer specifications.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


SPEED_OF_LIGHT_M_S = 299_792_458.0


@dataclass(frozen=True)
class LaserParameters:
    """Transmitter parameters."""

    wavelength_nm: float = 1550.0
    pulse_width_ns: float = 1.0
    emitted_photons: float = 100_000.0


@dataclass(frozen=True)
class ReceiverParameters:
    """Optical receiver parameters."""

    aperture_diameter_m: float = 0.10
    optical_efficiency: float = 0.55


@dataclass(frozen=True)
class DetectorParameters:
    """Single-photon detector parameters."""

    photon_detection_efficiency: float = 0.35
    timing_jitter_ps: float = 80.0
    dark_count_rate_hz: float = 500.0


@dataclass(frozen=True)
class AtmosphereParameters:
    """Atmospheric propagation and background parameters."""

    extinction_coefficient_per_km: float = 0.10
    background_photons_per_bin: float = 0.02
    backscatter_photons_per_bin: float = 0.0


@dataclass(frozen=True)
class TargetParameters:
    """Target optical properties."""

    range_m: float = 1000.0
    reflectivity: float = 0.20
    projected_area_m2: float = 1.0


@dataclass(frozen=True)
class HistogramParameters:
    """Time-of-flight histogram parameters."""

    num_bins: int = 512
    bin_width_ns: float = 5.0
    integration_time_s: float = 0.01


@dataclass(frozen=True)
class PhotonReturnResult:
    """Output of one simulated LiDAR observation."""

    photon_counts: NDArray[np.int64]
    expected_counts: NDArray[np.float64]
    expected_signal_counts: NDArray[np.float64]
    expected_background_counts: NDArray[np.float64]
    target_bin: int
    round_trip_time_s: float
    expected_detected_signal_photons: float


def receiver_aperture_area(aperture_diameter_m: float) -> float:
    """Return the circular receiver aperture area in square metres."""
    if aperture_diameter_m <= 0:
        raise ValueError("aperture_diameter_m must be positive.")

    radius_m = aperture_diameter_m / 2.0
    return float(np.pi * radius_m**2)


def two_way_atmospheric_transmission(
    range_m: float,
    extinction_coefficient_per_km: float,
) -> float:
    """Calculate two-way Beer-Lambert atmospheric transmission.

    The extinction coefficient is specified per kilometre. The signal travels
    from the sensor to the target and back, producing the factor of two.
    """
    if range_m < 0:
        raise ValueError("range_m cannot be negative.")

    if extinction_coefficient_per_km < 0:
        raise ValueError("extinction_coefficient_per_km cannot be negative.")

    range_km = range_m / 1000.0

    return float(
        np.exp(-2.0 * extinction_coefficient_per_km * range_km)
    )


def round_trip_time(range_m: float) -> float:
    """Return target round-trip time of flight in seconds."""
    if range_m < 0:
        raise ValueError("range_m cannot be negative.")

    return 2.0 * range_m / SPEED_OF_LIGHT_M_S


def time_of_flight_bin(
    range_m: float,
    histogram: HistogramParameters,
) -> int:
    """Map target range to the closest histogram bin."""
    time_s = round_trip_time(range_m)
    bin_width_s = histogram.bin_width_ns * 1e-9
    target_bin = int(np.rint(time_s / bin_width_s))

    if target_bin < 0 or target_bin >= histogram.num_bins:
        raise ValueError(
            "Target round-trip time falls outside the histogram window. "
            f"Computed bin={target_bin}, available bins={histogram.num_bins}."
        )

    return target_bin


def expected_detected_signal_photons(
    laser: LaserParameters,
    receiver: ReceiverParameters,
    detector: DetectorParameters,
    atmosphere: AtmosphereParameters,
    target: TargetParameters,
) -> float:
    """Estimate the expected number of detected target-return photons.

    The simplified link-budget model is:

        N_detected =
            N_tx
            × target_reflectivity
            × projected_target_area
            × receiver_aperture_area
            / (pi × range^2)
            × two_way_atmospheric_transmission
            × optical_efficiency
            × photon_detection_efficiency

    This formulation is intentionally transparent for early validation.
    Later versions may include beam divergence, incidence angle, receiver field
    of view, speckle, target BRDF and wavelength-dependent atmospheric effects.
    """
    if laser.emitted_photons < 0:
        raise ValueError("emitted_photons cannot be negative.")

    if target.range_m <= 0:
        raise ValueError("target.range_m must be positive.")

    if not 0.0 <= target.reflectivity <= 1.0:
        raise ValueError("target.reflectivity must be between 0 and 1.")

    if target.projected_area_m2 <= 0:
        raise ValueError("target.projected_area_m2 must be positive.")

    if not 0.0 <= receiver.optical_efficiency <= 1.0:
        raise ValueError("receiver.optical_efficiency must be between 0 and 1.")

    if not 0.0 <= detector.photon_detection_efficiency <= 1.0:
        raise ValueError(
            "detector.photon_detection_efficiency must be between 0 and 1."
        )

    aperture_area_m2 = receiver_aperture_area(
        receiver.aperture_diameter_m
    )

    geometric_coupling = (
        target.projected_area_m2
        * aperture_area_m2
        / (np.pi * target.range_m**2)
    )

    atmospheric_transmission = two_way_atmospheric_transmission(
        range_m=target.range_m,
        extinction_coefficient_per_km=(
            atmosphere.extinction_coefficient_per_km
        ),
    )

    expected_photons = (
        laser.emitted_photons
        * target.reflectivity
        * geometric_coupling
        * atmospheric_transmission
        * receiver.optical_efficiency
        * detector.photon_detection_efficiency
    )

    return float(max(expected_photons, 0.0))


def gaussian_instrument_response(
    target_bin: int,
    histogram: HistogramParameters,
    detector: DetectorParameters,
    laser: LaserParameters,
) -> NDArray[np.float64]:
    """Create a normalized Gaussian instrument-response function.

    The temporal spread combines laser pulse width and detector timing jitter
    using a root-sum-square approximation.
    """
    if laser.pulse_width_ns <= 0:
        raise ValueError("laser.pulse_width_ns must be positive.")

    if detector.timing_jitter_ps < 0:
        raise ValueError("detector.timing_jitter_ps cannot be negative.")

    pulse_sigma_ns = laser.pulse_width_ns / 2.355
    detector_sigma_ns = detector.timing_jitter_ps * 1e-3 / 2.355

    combined_sigma_ns = float(
        np.sqrt(pulse_sigma_ns**2 + detector_sigma_ns**2)
    )

    sigma_bins = max(
        combined_sigma_ns / histogram.bin_width_ns,
        1e-6,
    )

    bins = np.arange(histogram.num_bins, dtype=np.float64)

    response = np.exp(
        -0.5 * ((bins - target_bin) / sigma_bins) ** 2
    )

    response_sum = float(response.sum())

    if response_sum <= 0:
        raise RuntimeError("Instrument response could not be normalized.")

    return response / response_sum


def expected_background_histogram(
    atmosphere: AtmosphereParameters,
    detector: DetectorParameters,
    histogram: HistogramParameters,
) -> NDArray[np.float64]:
    """Calculate expected background and dark-count photons per bin."""
    if atmosphere.background_photons_per_bin < 0:
        raise ValueError("background_photons_per_bin cannot be negative.")

    if atmosphere.backscatter_photons_per_bin < 0:
        raise ValueError("backscatter_photons_per_bin cannot be negative.")

    if detector.dark_count_rate_hz < 0:
        raise ValueError("dark_count_rate_hz cannot be negative.")

    dark_counts_total = (
        detector.dark_count_rate_hz * histogram.integration_time_s
    )

    dark_counts_per_bin = dark_counts_total / histogram.num_bins

    expected_per_bin = (
        atmosphere.background_photons_per_bin
        + atmosphere.backscatter_photons_per_bin
        + dark_counts_per_bin
    )

    return np.full(
        histogram.num_bins,
        expected_per_bin,
        dtype=np.float64,
    )


def simulate_photon_return(
    laser: LaserParameters,
    receiver: ReceiverParameters,
    detector: DetectorParameters,
    atmosphere: AtmosphereParameters,
    target: TargetParameters,
    histogram: HistogramParameters,
    rng: Optional[np.random.Generator] = None,
) -> PhotonReturnResult:
    """Simulate one time-resolved single-photon LiDAR observation."""
    if rng is None:
        rng = np.random.default_rng()

    target_bin = time_of_flight_bin(
        range_m=target.range_m,
        histogram=histogram,
    )

    signal_photons = expected_detected_signal_photons(
        laser=laser,
        receiver=receiver,
        detector=detector,
        atmosphere=atmosphere,
        target=target,
    )

    impulse_response = gaussian_instrument_response(
        target_bin=target_bin,
        histogram=histogram,
        detector=detector,
        laser=laser,
    )

    expected_signal = signal_photons * impulse_response

    expected_background = expected_background_histogram(
        atmosphere=atmosphere,
        detector=detector,
        histogram=histogram,
    )

    expected_total = expected_signal + expected_background

    photon_counts = rng.poisson(expected_total).astype(np.int64)

    return PhotonReturnResult(
        photon_counts=photon_counts,
        expected_counts=expected_total,
        expected_signal_counts=expected_signal,
        expected_background_counts=expected_background,
        target_bin=target_bin,
        round_trip_time_s=round_trip_time(target.range_m),
        expected_detected_signal_photons=signal_photons,
    )


def main() -> None:
    """Run a reproducible demonstration simulation."""
    rng = np.random.default_rng(42)

    result = simulate_photon_return(
        laser=LaserParameters(),
        receiver=ReceiverParameters(),
        detector=DetectorParameters(),
        atmosphere=AtmosphereParameters(),
        target=TargetParameters(range_m=1000.0),
        histogram=HistogramParameters(
            num_bins=2048,
            bin_width_ns=5.0,
        ),
        rng=rng,
    )

    print("Target bin:", result.target_bin)
    print("Round-trip time (microseconds):", result.round_trip_time_s * 1e6)
    print(
        "Expected detected signal photons:",
        result.expected_detected_signal_photons,
    )
    print("Observed total photon counts:", int(result.photon_counts.sum()))


if __name__ == "__main__":
    main()
