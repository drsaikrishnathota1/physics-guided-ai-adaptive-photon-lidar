"""Ablation policies for the physics-guided allocation score."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.environment.adaptive_lidar_env import (
    AdaptiveLidarEnvironment,
    AdaptiveObservation,
)
from src.policies.base_policy import PhotonAllocationPolicy
from src.policies.entropy_policy import binary_entropy


@dataclass
class InformationAblationPolicy(PhotonAllocationPolicy):
    """Configurable components of the information-guided score."""

    use_optical_contrast: bool = True
    use_exploration: bool = True
    name: str = "information_ablation"

    def select_region(
        self,
        observation: AdaptiveObservation,
        environment: AdaptiveLidarEnvironment,
    ) -> int:
        uncertainty = binary_entropy(
            observation.posterior_probabilities
        )

        score = uncertainty.copy()

        if self.use_optical_contrast:
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

            score *= np.log1p(optical_contrast)

        if self.use_exploration:
            score *= 1.0 / np.sqrt(
                observation.measurements_per_region + 1.0
            )

        return int(np.argmax(score))


def entropy_only_policy() -> InformationAblationPolicy:
    """Return posterior-entropy-only allocation."""
    return InformationAblationPolicy(
        use_optical_contrast=False,
        use_exploration=False,
        name="ablation_entropy_only",
    )


def entropy_contrast_policy() -> InformationAblationPolicy:
    """Return entropy multiplied by predicted optical contrast."""
    return InformationAblationPolicy(
        use_optical_contrast=True,
        use_exploration=False,
        name="ablation_entropy_contrast",
    )


def full_information_policy() -> InformationAblationPolicy:
    """Return the complete physics-guided heuristic."""
    return InformationAblationPolicy(
        use_optical_contrast=True,
        use_exploration=True,
        name="ablation_full_score",
    )
