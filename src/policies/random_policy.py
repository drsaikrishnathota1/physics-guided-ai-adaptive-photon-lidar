"""Random photon-allocation baseline."""

from __future__ import annotations

import numpy as np

from src.environment.adaptive_lidar_env import (
    AdaptiveLidarEnvironment,
    AdaptiveObservation,
)
from src.policies.base_policy import PhotonAllocationPolicy


class RandomPolicy(PhotonAllocationPolicy):
    """Select candidate regions uniformly at random."""

    name = "random"

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def select_region(
        self,
        observation: AdaptiveObservation,
        environment: AdaptiveLidarEnvironment,
    ) -> int:
        del observation

        return int(
            self.rng.integers(
                0,
                environment.number_of_regions,
            )
        )
