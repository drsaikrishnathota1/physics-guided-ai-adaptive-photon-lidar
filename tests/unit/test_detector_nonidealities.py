import numpy as np

from src.physics.detector_nonidealities import (
    SPADParameters,
    expected_spad_counts,
)


def test_expected_spad_counts_are_bounded() -> None:
    pulses = 1000
    incident = np.full(100, 1.0e6)

    expected = expected_spad_counts(
        incident,
        pulses_per_dwell=pulses,
        bin_width_ns=5.0,
        detector=SPADParameters(),
    )

    assert np.all(expected >= 0.0)
    assert np.all(expected <= pulses)
