"""Integrated journal-grade pulsed single-photon LiDAR simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.physics.atmospheric_channel import (
    AtmosphericChannel,
    distributed_backscatter_histogram,
    two_way_transmission,
)
from src.physics.beam_geometry import (
    BeamParameters,
    incidence_factor,
    target_intercept_fraction,
)
from src.physics.detector_nonidealities import (
    SPADParameters,
    expected_spad_counts,
    sample_spad_histogram,
)
from src.physics.photon_return_model import (
    SPEED_OF_LIGHT_M_S,
    gaussian_instrument_response,
)
from src.physics.pulse_accounting import PulseSchedule


@dataclass(frozen=True)
class JournalLaser:
    """Laser and pulse parameters."""

    wavelength_nm: float = 1550.0
    pulse_width_ns: float = 1.0
    beam: BeamParameters = BeamParameters()


@dataclass(frozen=True)
class JournalReceiver:
    """Receiver optical parameters."""

    aperture_diameter_m: float = 0.10
    optical_efficiency: float = 0.55

    def __post_init__(self) -> None:
        if self.aperture_diameter_m <= 0:
            raise ValueError(
                "aperture_diameter_m must be positive."
            )

        if not 0.0 <= self.optical_efficiency <= 1.0:
            raise ValueError(
                "optical_efficiency must be between 0 and 1."
            )

    @property
    def aperture_area_m2(self) -> float:
        radius_m = self.aperture_diameter_m / 2.0

        return float(np.pi * radius_m**2)


@dataclass(frozen=True)
class JournalTarget:
    """Extended diffuse target parameters."""

    range_m: float = 1000.0
    reflectivity: float = 0.20
    projected_area_m2: float = 1.0
    incidence_angle_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.range_m <= 0:
            raise ValueError("range_m must be positive.")

        if not 0.0 <= self.reflectivity <= 1.0:
            raise ValueError(
                "reflectivity must be between 0 and 1."
            )

        if self.projected_area_m2 <= 0:
            raise ValueError(
                "projected_area_m2 must be positive."
            )


@dataclass(frozen=True)
class JournalHistogram:
    """Time-of-flight histogram configuration."""

    num_bins: int = 4096
    bin_width_ns: float = 5.0

    def __post_init__(self) -> None:
        if self.num_bins <= 0:
            raise ValueError("num_bins must be positive.")

        if self.bin_width_ns <= 0:
            raise ValueError("bin_width_ns must be positive.")

    @property
    def maximum_unambiguous_range_m(self) -> float:
        return (
            SPEED_OF_LIGHT_M_S
            * self.num_bins
            * self.bin_width_ns
            * 1.0e-9
            / 2.0
        )


@dataclass(frozen=True)
class JournalSimulationResult:
    """Outputs of the integrated simulation."""

    observed_counts: NDArray[np.int64]
    expected_detector_counts: NDArray[np.float64]
    incident_signal_photons: NDArray[np.float64]
    incident_backscatter_photons: NDArray[np.float64]
    incident_background_photons: NDArray[np.float64]
    target_bin: int
    expected_target_return_photons: float
    total_transmitted_photons: float
    intercepted_fraction: float


def expected_target_return_photons(
    *,
    schedule: PulseSchedule,
    laser: JournalLaser,
    receiver: JournalReceiver,
    target: JournalTarget,
    atmosphere: AtmosphericChannel,
) -> tuple[float, float]:
    """Calculate target-return photons incident on the detector."""
    intercepted_fraction = target_intercept_fraction(
        range_m=target.range_m,
        projected_target_area_m2=target.projected_area_m2,
        beam=laser.beam,
    )

    angular_factor = incidence_factor(
        target.incidence_angle_deg
    )

    atmospheric_factor = float(
        two_way_transmission(
            target.range_m,
            atmosphere.extinction_coefficient_per_km,
        )
    )

    diffuse_receiver_coupling = (
        receiver.aperture_area_m2
        / (np.pi * target.range_m**2)
    )

    expected_photons = (
        schedule.total_transmitted_photons
        * laser.beam.transmitter_efficiency
        * intercepted_fraction
        * target.reflectivity
        * angular_factor
        * atmospheric_factor
        * diffuse_receiver_coupling
        * receiver.optical_efficiency
    )

    return float(expected_photons), intercepted_fraction


def target_bin(
    target_range_m: float,
    histogram: JournalHistogram,
) -> int:
    """Map target range to the nearest time-of-flight bin."""
    round_trip_time_s = (
        2.0 * target_range_m / SPEED_OF_LIGHT_M_S
    )

    index = int(
        np.rint(
            round_trip_time_s
            / (histogram.bin_width_ns * 1.0e-9)
        )
    )

    if not 0 <= index < histogram.num_bins:
        raise ValueError(
            "Target falls outside the unambiguous histogram range."
        )

    return index


def simulate_journal_grade_return(
    *,
    schedule: PulseSchedule,
    laser: JournalLaser,
    receiver: JournalReceiver,
    detector: SPADParameters,
    atmosphere: AtmosphericChannel,
    target: JournalTarget,
    histogram: JournalHistogram,
    rng: np.random.Generator,
) -> JournalSimulationResult:
    """Simulate target, atmospheric, background, and detector response."""
    index = target_bin(
        target_range_m=target.range_m,
        histogram=histogram,
    )

    expected_target_photons, intercepted_fraction = (
        expected_target_return_photons(
            schedule=schedule,
            laser=laser,
            receiver=receiver,
            target=target,
            atmosphere=atmosphere,
        )
    )

    from src.physics.photon_return_model import (
        DetectorParameters,
        HistogramParameters,
        LaserParameters,
    )

    response = gaussian_instrument_response(
        target_bin=index,
        histogram=HistogramParameters(
            num_bins=histogram.num_bins,
            bin_width_ns=histogram.bin_width_ns,
        ),
        detector=DetectorParameters(
            photon_detection_efficiency=detector.detection_efficiency,
            timing_jitter_ps=detector.timing_jitter_ps,
            dark_count_rate_hz=detector.dark_count_rate_hz,
        ),
        laser=LaserParameters(
            wavelength_nm=laser.wavelength_nm,
            pulse_width_ns=laser.pulse_width_ns,
            emitted_photons=schedule.total_transmitted_photons,
        ),
    )

    incident_signal = expected_target_photons * response

    incident_backscatter = distributed_backscatter_histogram(
        num_bins=histogram.num_bins,
        bin_width_ns=histogram.bin_width_ns,
        total_transmitted_photons=(
            schedule.total_transmitted_photons
        ),
        transmitter_efficiency=(
            laser.beam.transmitter_efficiency
        ),
        receiver_aperture_area_m2=(
            receiver.aperture_area_m2
        ),
        receiver_efficiency=receiver.optical_efficiency,
        channel=atmosphere,
    )

    incident_background = np.full(
        histogram.num_bins,
        atmosphere.background_photons_per_bin_per_pulse
        * schedule.pulses_per_dwell,
        dtype=np.float64,
    )

    incident_total = (
        incident_signal
        + incident_backscatter
        + incident_background
    )

    expected_detector = expected_spad_counts(
        incident_total,
        pulses_per_dwell=schedule.pulses_per_dwell,
        bin_width_ns=histogram.bin_width_ns,
        detector=detector,
    )

    observed = sample_spad_histogram(
        incident_total,
        pulses_per_dwell=schedule.pulses_per_dwell,
        bin_width_ns=histogram.bin_width_ns,
        detector=detector,
        rng=rng,
    )

    return JournalSimulationResult(
        observed_counts=observed,
        expected_detector_counts=expected_detector,
        incident_signal_photons=incident_signal,
        incident_backscatter_photons=incident_backscatter,
        incident_background_photons=incident_background,
        target_bin=index,
        expected_target_return_photons=expected_target_photons,
        total_transmitted_photons=(
            schedule.total_transmitted_photons
        ),
        intercepted_fraction=intercepted_fraction,
    )
