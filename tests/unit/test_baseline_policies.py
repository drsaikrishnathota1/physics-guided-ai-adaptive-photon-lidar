from src.environment.adaptive_lidar_env import (
    AdaptiveEnvironmentConfig,
    AdaptiveLidarEnvironment,
    RegionDefinition,
)
from src.physics.atmospheric_channel import AtmosphericChannel
from src.physics.detector_nonidealities import SPADParameters
from src.physics.journal_grade_model import (
    JournalHistogram,
    JournalLaser,
    JournalReceiver,
)
from src.policies.entropy_policy import EntropyPolicy
from src.policies.greedy_policy import GreedyPolicy
from src.policies.information_gain_policy import (
    InformationGainPolicy,
)
from src.policies.random_policy import RandomPolicy
from src.policies.uniform_policy import UniformPolicy


def make_environment() -> AdaptiveLidarEnvironment:
    return AdaptiveLidarEnvironment(
        regions=[
            RegionDefinition(500.0, 0.20),
            RegionDefinition(1000.0, 0.20),
            RegionDefinition(1500.0, 0.20),
        ],
        target_region_index=1,
        config=AdaptiveEnvironmentConfig(
            total_photon_budget=1.0e12,
            photons_per_action=1.0e11,
        ),
        laser=JournalLaser(),
        receiver=JournalReceiver(),
        detector=SPADParameters(),
        atmosphere=AtmosphericChannel(),
        histogram=JournalHistogram(),
        seed=42,
    )


def test_all_policies_return_valid_region() -> None:
    environment = make_environment()
    observation = environment.observation()

    policies = [
        UniformPolicy(),
        RandomPolicy(seed=42),
        GreedyPolicy(),
        EntropyPolicy(),
        InformationGainPolicy(),
    ]

    for policy in policies:
        selected = policy.select_region(
            observation,
            environment,
        )

        assert 0 <= selected < environment.number_of_regions
