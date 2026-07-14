from src.environment.adaptive_lidar_env import (
    AdaptiveEnvironmentConfig,
    AdaptiveLidarEnvironment,
    RegionDefinition,
    bayesian_binary_update,
)
from src.physics.atmospheric_channel import AtmosphericChannel
from src.physics.detector_nonidealities import SPADParameters
from src.physics.journal_grade_model import (
    JournalHistogram,
    JournalLaser,
    JournalReceiver,
)


def make_environment() -> AdaptiveLidarEnvironment:
    return AdaptiveLidarEnvironment(
        regions=[
            RegionDefinition(
                range_m=500.0,
                reflectivity_if_target=0.20,
            ),
            RegionDefinition(
                range_m=1000.0,
                reflectivity_if_target=0.20,
            ),
        ],
        target_region_index=1,
        config=AdaptiveEnvironmentConfig(
            total_photon_budget=1.0e12,
            photons_per_action=1.0e11,
            pulses_per_action=1000,
            maximum_steps=10,
        ),
        laser=JournalLaser(),
        receiver=JournalReceiver(),
        detector=SPADParameters(),
        atmosphere=AtmosphericChannel(),
        histogram=JournalHistogram(),
        seed=42,
    )


def test_bayesian_update_favors_target_hypothesis() -> None:
    posterior = bayesian_binary_update(
        prior_probability=0.10,
        observed_count=10,
        expected_count_h0=1.0,
        expected_count_h1=10.0,
    )

    assert posterior > 0.10


def test_environment_step_consumes_photons() -> None:
    environment = make_environment()

    before = environment.observation()
    result = environment.step(0)
    after = result.observation

    assert after.remaining_photons < before.remaining_photons
    assert after.current_step == 1
    assert after.measurements_per_region[0] == 1


def test_predicted_snr_is_nonnegative() -> None:
    environment = make_environment()

    assert (
        environment.predicted_signal_to_background_ratio(0)
        >= 0.0
    )
