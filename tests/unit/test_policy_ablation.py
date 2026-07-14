import numpy as np

from src.environment.adaptive_lidar_env import AdaptiveObservation
from src.policies.information_ablation_policy import (
    entropy_contrast_policy,
    entropy_only_policy,
    full_information_policy,
)


class StubEnvironment:
    """Minimal environment for policy-component tests."""

    number_of_regions = 3

    @staticmethod
    def predicted_signal_to_background_ratio(
        region_index: int,
    ) -> float:
        return [0.5, 2.0, 8.0][region_index]


def observation() -> AdaptiveObservation:
    return AdaptiveObservation(
        posterior_probabilities=np.asarray(
            [0.50, 0.50, 0.50],
            dtype=np.float64,
        ),
        measurements_per_region=np.asarray(
            [0, 0, 4],
            dtype=np.int64,
        ),
        photons_used_per_region=np.asarray(
            [0.0, 0.0, 1.0],
            dtype=np.float64,
        ),
        remaining_photons=1.0,
        current_step=0,
    )


def test_entropy_only_uses_first_tie() -> None:
    selected = entropy_only_policy().select_region(
        observation(),
        StubEnvironment(),
    )

    assert selected == 0


def test_entropy_contrast_prefers_highest_contrast() -> None:
    selected = entropy_contrast_policy().select_region(
        observation(),
        StubEnvironment(),
    )

    assert selected == 2


def test_full_score_includes_exploration_penalty() -> None:
    selected = full_information_policy().select_region(
        observation(),
        StubEnvironment(),
    )

    assert selected in {1, 2}
