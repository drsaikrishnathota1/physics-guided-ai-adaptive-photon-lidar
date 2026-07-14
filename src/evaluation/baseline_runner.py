"""Utilities for evaluating photon-allocation policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from src.environment.adaptive_lidar_env import (
    AdaptiveLidarEnvironment,
)
from src.policies.base_policy import PhotonAllocationPolicy


@dataclass(frozen=True)
class EpisodeResult:
    """Summary of one adaptive sensing episode."""

    policy: str
    seed: int
    target_region_index: Optional[int]
    target_present: bool
    detected: bool
    false_alarm: bool
    termination_reason: str
    steps: int
    photons_used: float
    target_posterior: Optional[float]
    maximum_posterior: float
    maximum_non_target_posterior: float
    declared_region_index: int


def run_episode(
    *,
    policy: PhotonAllocationPolicy,
    environment: AdaptiveLidarEnvironment,
    seed: int,
) -> EpisodeResult:
    """Run a sensing policy until detection or budget termination."""
    termination_reason = "unknown"

    while True:
        observation = environment.observation()

        region_index = policy.select_region(
            observation,
            environment,
        )

        result = environment.step(region_index)

        if result.terminated:
            termination_reason = (
                result.termination_reason
                or "unknown"
            )
            break

    final_observation = environment.observation()
    posterior = final_observation.posterior_probabilities

    maximum_index = int(np.argmax(posterior))
    maximum_posterior = float(posterior[maximum_index])

    target_index_raw = environment.target_region_index
    target_present = target_index_raw >= 0

    if target_present:
        target_index: Optional[int] = target_index_raw
        target_posterior: Optional[float] = float(
            posterior[target_index_raw]
        )

        non_target_posteriors = np.delete(
            posterior,
            target_index_raw,
        )

        maximum_non_target = (
            float(np.max(non_target_posteriors))
            if non_target_posteriors.size
            else 0.0
        )

        detected = (
            target_posterior
            >= environment.config.detection_threshold
        )

        false_alarm = (
            maximum_index != target_index_raw
            and maximum_posterior
            >= environment.config.detection_threshold
        )
    else:
        target_index = None
        target_posterior = None
        maximum_non_target = maximum_posterior
        detected = False

        false_alarm = (
            maximum_posterior
            >= environment.config.detection_threshold
        )

        if false_alarm:
            termination_reason = "false_alarm"

    photons_used = (
        environment.config.total_photon_budget
        - final_observation.remaining_photons
    )

    return EpisodeResult(
        policy=policy.name,
        seed=seed,
        target_region_index=target_index,
        target_present=target_present,
        detected=bool(detected),
        false_alarm=bool(false_alarm),
        termination_reason=termination_reason,
        steps=final_observation.current_step,
        photons_used=float(photons_used),
        target_posterior=target_posterior,
        maximum_posterior=maximum_posterior,
        maximum_non_target_posterior=maximum_non_target,
        declared_region_index=maximum_index,
    )


def episode_result_to_dict(
    result: EpisodeResult,
) -> dict:
    """Convert an episode result to a JSON-safe dictionary."""
    return asdict(result)
