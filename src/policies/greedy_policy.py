"""Greedy posterior-probability allocation."""

from __future__ import annotations

import numpy as np

from src.environment.adaptive_lidar_env import (
    AdaptiveLidarEnvironment,
    AdaptiveObservation,
)
from src.policies.base_policy import PhotonAllocationPolicy


class GreedyPolicy(PhotonAllocationPolicy):
    """Measure the region with highest current target posterior."""

    name = "greedy"

    def select_region(
        self,
        observation: AdaptiveObservation,
        environment: AdaptiveLidarEnvironment,
    ) -> int:
        del environment

        return int(
            np.argmax(observation.posterior_probabilities)
        )
