import numpy as np

from src.physics.atmospheric_channel import AtmosphericChannel
from src.physics.detector_nonidealities import SPADParameters
from src.physics.journal_grade_model import (
    JournalHistogram,
    JournalLaser,
    JournalReceiver,
    JournalTarget,
    expected_target_return_photons,
    simulate_journal_grade_return,
)
from src.physics.pulse_accounting import PulseSchedule


def common_parameters() -> dict:
    return {
        "schedule": PulseSchedule(),
        "laser": JournalLaser(),
        "receiver": JournalReceiver(),
        "detector": SPADParameters(),
        "atmosphere": AtmosphericChannel(),
        "histogram": JournalHistogram(),
    }


def test_target_return_decreases_with_range() -> None:
    parameters = common_parameters()

    near, _ = expected_target_return_photons(
        schedule=parameters["schedule"],
        laser=parameters["laser"],
        receiver=parameters["receiver"],
        target=JournalTarget(range_m=500.0),
        atmosphere=parameters["atmosphere"],
    )

    far, _ = expected_target_return_photons(
        schedule=parameters["schedule"],
        laser=parameters["laser"],
        receiver=parameters["receiver"],
        target=JournalTarget(range_m=2000.0),
        atmosphere=parameters["atmosphere"],
    )

    assert near > far


def test_integrated_model_runs() -> None:
    result = simulate_journal_grade_return(
        **common_parameters(),
        target=JournalTarget(range_m=1000.0),
        rng=np.random.default_rng(42),
    )

    assert result.observed_counts.shape == (4096,)
    assert result.target_bin > 0
    assert result.total_transmitted_photons > 0
