"""Sequential adaptive photon-allocation environment."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, lgamma, log
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from src.physics.atmospheric_channel import AtmosphericChannel
from src.physics.detector_nonidealities import SPADParameters
from src.physics.journal_grade_model import (
    JournalHistogram,
    JournalLaser,
    JournalReceiver,
    JournalTarget,
    simulate_journal_grade_return,
)
from src.physics.pulse_accounting import PulseSchedule


EPSILON = 1.0e-12


@dataclass(frozen=True)
class RegionDefinition:
    """Physical properties of one candidate sensing region."""

    range_m: float
    reflectivity_if_target: float
    projected_area_m2: float = 1.0
    incidence_angle_deg: float = 0.0


@dataclass(frozen=True)
class AdaptiveEnvironmentConfig:
    """Sequential sensing configuration."""

    total_photon_budget: float = 5.0e12
    photons_per_action: float = 2.5e11
    pulses_per_action: int = 1000
    initial_target_probability: float = 0.10
    detection_threshold: float = 0.90
    rejection_threshold: float = 0.05
    target_window_half_width_bins: int = 2
    maximum_steps: int = 100

    def __post_init__(self) -> None:
        if self.total_photon_budget <= 0:
            raise ValueError("total_photon_budget must be positive.")

        if self.photons_per_action <= 0:
            raise ValueError("photons_per_action must be positive.")

        if self.pulses_per_action <= 0:
            raise ValueError("pulses_per_action must be positive.")

        if self.photons_per_action > self.total_photon_budget:
            raise ValueError(
                "photons_per_action cannot exceed total_photon_budget."
            )

        if not 0.0 < self.initial_target_probability < 1.0:
            raise ValueError(
                "initial_target_probability must be between zero and one."
            )

        if not 0.0 < self.detection_threshold < 1.0:
            raise ValueError(
                "detection_threshold must be between zero and one."
            )

        if self.maximum_steps <= 0:
            raise ValueError("maximum_steps must be positive.")


@dataclass(frozen=True)
class AdaptiveObservation:
    """Current state supplied to a photon-allocation policy."""

    posterior_probabilities: NDArray[np.float64]
    measurements_per_region: NDArray[np.int64]
    photons_used_per_region: NDArray[np.float64]
    remaining_photons: float
    current_step: int


@dataclass(frozen=True)
class AdaptiveStepResult:
    """Result of one sequential photon-allocation action."""

    observation: AdaptiveObservation
    selected_region: int
    allocated_photons: float
    observed_window_counts: int
    expected_counts_h0: float
    expected_counts_h1: float
    posterior_before: float
    posterior_after: float
    terminated: bool
    termination_reason: Optional[str]


def poisson_log_probability(
    observed_count: int,
    expected_count: float,
) -> float:
    """Return the Poisson log probability."""
    if observed_count < 0:
        raise ValueError("observed_count cannot be negative.")

    expected = max(float(expected_count), EPSILON)

    return (
        observed_count * log(expected)
        - expected
        - lgamma(observed_count + 1.0)
    )


def bayesian_binary_update(
    prior_probability: float,
    observed_count: int,
    expected_count_h0: float,
    expected_count_h1: float,
) -> float:
    """Update target probability using Poisson likelihoods."""
    prior = float(
        np.clip(prior_probability, EPSILON, 1.0 - EPSILON)
    )

    log_likelihood_h0 = poisson_log_probability(
        observed_count,
        expected_count_h0,
    )

    log_likelihood_h1 = poisson_log_probability(
        observed_count,
        expected_count_h1,
    )

    log_prior_odds = log(prior / (1.0 - prior))

    log_posterior_odds = (
        log_prior_odds
        + log_likelihood_h1
        - log_likelihood_h0
    )

    if log_posterior_odds >= 0:
        posterior = 1.0 / (
            1.0 + exp(-log_posterior_odds)
        )
    else:
        exponential = exp(log_posterior_odds)
        posterior = exponential / (1.0 + exponential)

    return float(
        np.clip(posterior, EPSILON, 1.0 - EPSILON)
    )


class AdaptiveLidarEnvironment:
    """Sequential single-photon LiDAR allocation environment."""

    def __init__(
        self,
        *,
        regions: list[RegionDefinition],
        target_region_index: int,
        config: AdaptiveEnvironmentConfig,
        laser: JournalLaser,
        receiver: JournalReceiver,
        detector: SPADParameters,
        atmosphere: AtmosphericChannel,
        histogram: JournalHistogram,
        seed: int = 42,
    ) -> None:
        if not regions:
            raise ValueError("At least one region is required.")

        if target_region_index != -1 and not 0 <= target_region_index < len(regions):
            raise ValueError(
                "target_region_index must be -1 or a valid region index."
            )

        self.regions = regions
        self.target_region_index = target_region_index
        self.config = config
        self.laser = laser
        self.receiver = receiver
        self.detector = detector
        self.atmosphere = atmosphere
        self.histogram = histogram
        self.rng = np.random.default_rng(seed)

        self._posterior_probabilities = np.full(
            len(regions),
            config.initial_target_probability,
            dtype=np.float64,
        )

        self._measurements_per_region = np.zeros(
            len(regions),
            dtype=np.int64,
        )

        self._photons_used_per_region = np.zeros(
            len(regions),
            dtype=np.float64,
        )

        self._remaining_photons = config.total_photon_budget
        self._current_step = 0

    @property
    def number_of_regions(self) -> int:
        """Return number of candidate sensing regions."""
        return len(self.regions)

    def observation(self) -> AdaptiveObservation:
        """Return a defensive copy of current environment state."""
        return AdaptiveObservation(
            posterior_probabilities=(
                self._posterior_probabilities.copy()
            ),
            measurements_per_region=(
                self._measurements_per_region.copy()
            ),
            photons_used_per_region=(
                self._photons_used_per_region.copy()
            ),
            remaining_photons=float(self._remaining_photons),
            current_step=self._current_step,
        )

    def _measurement_schedule(
        self,
        allocated_photons: float,
    ) -> PulseSchedule:
        repetition_rate_hz = 100_000.0

        return PulseSchedule(
            photons_per_pulse=(
                allocated_photons
                / self.config.pulses_per_action
            ),
            repetition_rate_hz=repetition_rate_hz,
            dwell_time_s=(
                self.config.pulses_per_action
                / repetition_rate_hz
            ),
            active_gate_width_ns=(
                self.histogram.num_bins
                * self.histogram.bin_width_ns
            ),
        )

    def _simulate_hypothesis(
        self,
        *,
        region_index: int,
        target_present: bool,
        allocated_photons: float,
    ):
        region = self.regions[region_index]

        target = JournalTarget(
            range_m=region.range_m,
            reflectivity=(
                region.reflectivity_if_target
                if target_present
                else 0.0
            ),
            projected_area_m2=region.projected_area_m2,
            incidence_angle_deg=region.incidence_angle_deg,
        )

        return simulate_journal_grade_return(
            schedule=self._measurement_schedule(
                allocated_photons
            ),
            laser=self.laser,
            receiver=self.receiver,
            detector=self.detector,
            atmosphere=self.atmosphere,
            target=target,
            histogram=self.histogram,
            rng=self.rng,
        )

    def expected_window_counts(
        self,
        *,
        region_index: int,
        target_present: bool,
        allocated_photons: Optional[float] = None,
    ) -> float:
        """Return expected detector counts near target range."""
        photons = (
            self.config.photons_per_action
            if allocated_photons is None
            else float(allocated_photons)
        )

        result = self._simulate_hypothesis(
            region_index=region_index,
            target_present=target_present,
            allocated_photons=photons,
        )

        half_width = self.config.target_window_half_width_bins
        start = max(0, result.target_bin - half_width)
        end = min(
            self.histogram.num_bins,
            result.target_bin + half_width + 1,
        )

        return float(
            result.expected_detector_counts[start:end].sum()
        )

    def predicted_signal_to_background_ratio(
        self,
        region_index: int,
    ) -> float:
        """Return predicted target contrast for one region."""
        expected_h0 = self.expected_window_counts(
            region_index=region_index,
            target_present=False,
        )

        expected_h1 = self.expected_window_counts(
            region_index=region_index,
            target_present=True,
        )

        return (
            max(expected_h1 - expected_h0, 0.0)
            / max(expected_h0, EPSILON)
        )

    def step(
        self,
        region_index: int,
        allocated_photons: Optional[float] = None,
    ) -> AdaptiveStepResult:
        """Measure one region and update its target posterior."""
        if not 0 <= region_index < self.number_of_regions:
            raise ValueError("region_index is out of range.")

        if self._remaining_photons <= 0:
            raise RuntimeError("Photon budget is exhausted.")

        requested_photons = (
            self.config.photons_per_action
            if allocated_photons is None
            else float(allocated_photons)
        )

        if requested_photons <= 0:
            raise ValueError("allocated_photons must be positive.")

        actual_photons = min(
            requested_photons,
            self._remaining_photons,
        )

        target_present = (
            region_index == self.target_region_index
        )

        actual_result = self._simulate_hypothesis(
            region_index=region_index,
            target_present=target_present,
            allocated_photons=actual_photons,
        )

        h0_result = self._simulate_hypothesis(
            region_index=region_index,
            target_present=False,
            allocated_photons=actual_photons,
        )

        h1_result = self._simulate_hypothesis(
            region_index=region_index,
            target_present=True,
            allocated_photons=actual_photons,
        )

        half_width = self.config.target_window_half_width_bins
        start = max(
            0,
            actual_result.target_bin - half_width,
        )
        end = min(
            self.histogram.num_bins,
            actual_result.target_bin + half_width + 1,
        )

        observed_window_counts = int(
            actual_result.observed_counts[start:end].sum()
        )

        expected_window_h0 = float(
            h0_result.expected_detector_counts[start:end].sum()
        )

        expected_window_h1 = float(
            h1_result.expected_detector_counts[start:end].sum()
        )

        posterior_before = float(
            self._posterior_probabilities[region_index]
        )

        posterior_after = bayesian_binary_update(
            prior_probability=posterior_before,
            observed_count=observed_window_counts,
            expected_count_h0=expected_window_h0,
            expected_count_h1=expected_window_h1,
        )

        self._posterior_probabilities[region_index] = (
            posterior_after
        )

        self._measurements_per_region[region_index] += 1
        self._photons_used_per_region[region_index] += (
            actual_photons
        )

        self._remaining_photons -= actual_photons
        self._current_step += 1

        termination_reason: Optional[str] = None

        if (
            self.target_region_index >= 0
            and self._posterior_probabilities[
                self.target_region_index
            ] >= self.config.detection_threshold
        ):
            termination_reason = "target_detected"

        elif self._remaining_photons <= 0:
            termination_reason = "photon_budget_exhausted"

        elif self._current_step >= self.config.maximum_steps:
            termination_reason = "maximum_steps_reached"

        return AdaptiveStepResult(
            observation=self.observation(),
            selected_region=region_index,
            allocated_photons=actual_photons,
            observed_window_counts=observed_window_counts,
            expected_counts_h0=expected_window_h0,
            expected_counts_h1=expected_window_h1,
            posterior_before=posterior_before,
            posterior_after=posterior_after,
            terminated=termination_reason is not None,
            termination_reason=termination_reason,
        )
