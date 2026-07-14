"""Uniform round-robin photon allocation."""

from __future__ import annotations

from src.environment.adaptive_lidar_env import (
    AdaptiveLidarEnvironment,
    AdaptiveObservation,
)
from src.policies.base_policy import PhotonAllocationPolicy


class UniformPolicy(PhotonAllocationPolicy):
    """Cycle evenly through all candidate regions."""

    name = "uniform"

    def select_region(
        self,
        observation: AdaptiveObservation,
        environment: AdaptiveLidarEnvironment,
    ) -> int:
        return (
            observation.current_step
            % environment.number_of_regions
        )
