from src.environment.adaptive_lidar_env import (
    AdaptiveEnvironmentConfig,
    AdaptiveLidarEnvironment,
    RegionDefinition,
)
from src.evaluation.baseline_runner import run_episode
from src.physics.atmospheric_channel import AtmosphericChannel
from src.physics.detector_nonidealities import SPADParameters
from src.physics.journal_grade_model import (
    JournalHistogram,
    JournalLaser,
    JournalReceiver,
)
from src.policies.uniform_policy import UniformPolicy


def make_absent_environment() -> AdaptiveLidarEnvironment:
    """Construct a compact target-absent test environment."""
    return AdaptiveLidarEnvironment(
        regions=[
            RegionDefinition(500.0, 0.20),
            RegionDefinition(1000.0, 0.18),
        ],
        target_region_index=-1,
        config=AdaptiveEnvironmentConfig(
            total_photon_budget=5.0e11,
            photons_per_action=2.5e11,
            pulses_per_action=100,
            detection_threshold=0.90,
            maximum_steps=2,
        ),
        laser=JournalLaser(),
        receiver=JournalReceiver(),
        detector=SPADParameters(),
        atmosphere=AtmosphericChannel(
            extinction_coefficient_per_km=0.10,
            volume_backscatter_coefficient_per_m=1.0e-7,
            background_photons_per_bin_per_pulse=1.0e-5,
        ),
        histogram=JournalHistogram(),
        seed=42,
    )


def test_target_absent_episode_completes() -> None:
    result = run_episode(
        policy=UniformPolicy(),
        environment=make_absent_environment(),
        seed=42,
    )

    assert result.target_present is False
    assert result.target_region_index is None
    assert result.target_posterior is None
    assert result.detected is False
    assert result.steps <= 2


def test_target_absent_result_has_false_alarm_field() -> None:
    result = run_episode(
        policy=UniformPolicy(),
        environment=make_absent_environment(),
        seed=43,
    )

    assert isinstance(result.false_alarm, bool)
    assert 0.0 <= result.maximum_posterior <= 1.0
