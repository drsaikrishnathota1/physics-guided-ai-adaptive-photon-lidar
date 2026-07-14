"""Approximate Bayesian information-gain allocation."""

from __future__ import annotations

import numpy as np

from src.environment.adaptive_lidar_env import (
    AdaptiveLidarEnvironment,
    AdaptiveObservation,
)
from src.policies.base_policy import PhotonAllocationPolicy
from src.policies.entropy_policy import binary_entropy


class InformationGainPolicy(PhotonAllocationPolicy):
    """Combine uncertainty, optical contrast, and exploration."""

    name = "information_gain"

    def select_region(
        self,
        observation: AdaptiveObservation,
        environment: AdaptiveLidarEnvironment,
    ) -> int:
        uncertainty = binary_entropy(
            observation.posterior_probabilities
        )

        optical_contrast = np.asarray(
            [
                environment.predicted_signal_to_background_ratio(
                    region_index
                )
                for region_index in range(
                    environment.number_of_regions
                )
            ],
            dtype=np.float64,
        )

        exploration_bonus = 1.0 / np.sqrt(
            observation.measurements_per_region + 1.0
        )

        score = (
            uncertainty
            * np.log1p(optical_contrast)
            * exploration_bonus
        )

        return int(np.argmax(score))
