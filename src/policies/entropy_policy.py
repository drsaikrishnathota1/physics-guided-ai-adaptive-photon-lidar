"""Posterior-entropy photon allocation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.environment.adaptive_lidar_env import (
    AdaptiveLidarEnvironment,
    AdaptiveObservation,
)
from src.policies.base_policy import PhotonAllocationPolicy


def binary_entropy(
    probabilities: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return binary entropy in natural-log units."""
    clipped = np.clip(
        probabilities,
        1.0e-12,
        1.0 - 1.0e-12,
    )

    return -(
        clipped * np.log(clipped)
        + (1.0 - clipped) * np.log(1.0 - clipped)
    )


class EntropyPolicy(PhotonAllocationPolicy):
    """Measure the region with maximum posterior uncertainty."""

    name = "entropy"

    def select_region(
        self,
        observation: AdaptiveObservation,
        environment: AdaptiveLidarEnvironment,
    ) -> int:
        del environment

        entropy = binary_entropy(
            observation.posterior_probabilities
        )

        return int(np.argmax(entropy))
