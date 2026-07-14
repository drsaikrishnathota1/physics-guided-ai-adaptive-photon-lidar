"""Utilities for evaluating photon-allocation policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.environment.adaptive_lidar_env import (
    AdaptiveLidarEnvironment,
)
from src.policies.base_policy import PhotonAllocationPolicy


@dataclass(frozen=True)
class EpisodeResult:
    """Summary of one adaptive sensing episode."""

    policy: str
    seed: int
    target_region_index: int
    detected: bool
    termination_reason: str
    steps: int
    photons_used: float
    target_posterior: float
    maximum_non_target_posterior: float


def run_episode(
    *,
    policy: PhotonAllocationPolicy,
    environment: AdaptiveLidarEnvironment,
    seed: int,
) -> EpisodeResult:
    """Run a policy until detection or termination."""
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

    target_index = environment.target_region_index

    target_posterior = float(
        final_observation.posterior_probabilities[
            target_index
        ]
    )

    non_target_posteriors = [
        float(probability)
        for index, probability in enumerate(
            final_observation.posterior_probabilities
        )
        if index != target_index
    ]

    maximum_non_target = (
        max(non_target_posteriors)
        if non_target_posteriors
        else 0.0
    )

    photons_used = (
        environment.config.total_photon_budget
        - final_observation.remaining_photons
    )

    return EpisodeResult(
        policy=policy.name,
        seed=seed,
        target_region_index=target_index,
        detected=(
            termination_reason == "target_detected"
        ),
        termination_reason=termination_reason,
        steps=final_observation.current_step,
        photons_used=float(photons_used),
        target_posterior=target_posterior,
        maximum_non_target_posterior=float(
            maximum_non_target
        ),
    )


def episode_result_to_dict(
    result: EpisodeResult,
) -> dict:
    """Convert an episode result to a JSON-safe dictionary."""
    return asdict(result)
