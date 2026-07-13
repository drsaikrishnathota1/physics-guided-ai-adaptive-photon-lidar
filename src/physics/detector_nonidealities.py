"""Approximate SPAD efficiency, pile-up, dead time, and afterpulsing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SPADParameters:
    """Single-photon avalanche diode parameters."""

    detection_efficiency: float = 0.35
    dark_count_rate_hz: float = 500.0
    dead_time_ns: float = 50.0
    afterpulse_probability: float = 0.01
    timing_jitter_ps: float = 80.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.detection_efficiency <= 1.0:
            raise ValueError(
                "detection_efficiency must be between 0 and 1."
            )

        if self.dark_count_rate_hz < 0:
            raise ValueError("dark_count_rate_hz cannot be negative.")

        if self.dead_time_ns < 0:
            raise ValueError("dead_time_ns cannot be negative.")

        if not 0.0 <= self.afterpulse_probability <= 1.0:
            raise ValueError(
                "afterpulse_probability must be between 0 and 1."
            )

        if self.timing_jitter_ps < 0:
            raise ValueError("timing_jitter_ps cannot be negative.")


def primary_detection_probability(
    incident_photons_per_pulse: NDArray[np.float64],
    *,
    bin_width_ns: float,
    detector: SPADParameters,
) -> NDArray[np.float64]:
    """Return binary SPAD click probability for each bin and pulse."""
    if np.any(incident_photons_per_pulse < 0):
        raise ValueError(
            "incident_photons_per_pulse cannot be negative."
        )

    dark_lambda_per_bin = (
        detector.dark_count_rate_hz
        * bin_width_ns
        * 1.0e-9
    )

    detected_lambda = (
        detector.detection_efficiency
        * incident_photons_per_pulse
        + dark_lambda_per_bin
    )

    return 1.0 - np.exp(-detected_lambda)


def apply_dead_time_approximation(
    expected_primary_counts: NDArray[np.float64],
    *,
    pulses_per_dwell: int,
    bin_width_ns: float,
    dead_time_ns: float,
) -> NDArray[np.float64]:
    """Apply a sequential expected-count dead-time approximation.

    Counts in earlier bins reduce detector availability in subsequent bins
    within the dead-time recovery window.
    """
    if pulses_per_dwell <= 0:
        raise ValueError("pulses_per_dwell must be positive.")

    if bin_width_ns <= 0:
        raise ValueError("bin_width_ns must be positive.")

    output = np.zeros_like(expected_primary_counts, dtype=np.float64)

    recovery_bins = int(np.ceil(dead_time_ns / bin_width_ns))

    if recovery_bins <= 0:
        return expected_primary_counts.astype(np.float64, copy=True)

    for index, expected_count in enumerate(expected_primary_counts):
        start = max(0, index - recovery_bins)

        recent_fraction = (
            output[start:index].sum()
            / pulses_per_dwell
        )

        available_fraction = float(
            np.clip(1.0 - recent_fraction, 0.0, 1.0)
        )

        output[index] = expected_count * available_fraction

    return output


def expected_spad_counts(
    incident_photons_total: NDArray[np.float64],
    *,
    pulses_per_dwell: int,
    bin_width_ns: float,
    detector: SPADParameters,
) -> NDArray[np.float64]:
    """Return expected SPAD counts after pile-up and detector effects."""
    if pulses_per_dwell <= 0:
        raise ValueError("pulses_per_dwell must be positive.")

    incident_per_pulse = (
        incident_photons_total / pulses_per_dwell
    )

    click_probability = primary_detection_probability(
        incident_per_pulse,
        bin_width_ns=bin_width_ns,
        detector=detector,
    )

    expected_primary = (
        pulses_per_dwell * click_probability
    )

    dead_time_corrected = apply_dead_time_approximation(
        expected_primary,
        pulses_per_dwell=pulses_per_dwell,
        bin_width_ns=bin_width_ns,
        dead_time_ns=detector.dead_time_ns,
    )

    expected_afterpulse = np.zeros_like(dead_time_corrected)

    if dead_time_corrected.size > 1:
        expected_afterpulse[1:] = (
            detector.afterpulse_probability
            * dead_time_corrected[:-1]
        )

    return np.minimum(
        dead_time_corrected + expected_afterpulse,
        float(pulses_per_dwell),
    )


def sample_spad_histogram(
    incident_photons_total: NDArray[np.float64],
    *,
    pulses_per_dwell: int,
    bin_width_ns: float,
    detector: SPADParameters,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Sample an approximate aggregate SPAD histogram."""
    expected = expected_spad_counts(
        incident_photons_total,
        pulses_per_dwell=pulses_per_dwell,
        bin_width_ns=bin_width_ns,
        detector=detector,
    )

    probabilities = np.clip(
        expected / pulses_per_dwell,
        0.0,
        1.0,
    )

    return rng.binomial(
        pulses_per_dwell,
        probabilities,
    ).astype(np.int64)
