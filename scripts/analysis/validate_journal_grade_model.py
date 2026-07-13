"""Validate upgraded single-photon LiDAR physics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.physics.atmospheric_channel import AtmosphericChannel
from src.physics.detector_nonidealities import SPADParameters
from src.physics.journal_grade_model import (
    JournalHistogram,
    JournalLaser,
    JournalReceiver,
    JournalTarget,
    simulate_journal_grade_return,
)
from src.physics.pulse_accounting import PulseSchedule


def main() -> None:
    output_directory = Path(
        "results/processed/journal_grade_validation"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    ranges = [250.0, 500.0, 1000.0, 2000.0, 3000.0]
    rows = []

    for range_m in ranges:
        result = simulate_journal_grade_return(
            schedule=PulseSchedule(),
            laser=JournalLaser(),
            receiver=JournalReceiver(),
            detector=SPADParameters(),
            atmosphere=AtmosphericChannel(),
            target=JournalTarget(range_m=range_m),
            histogram=JournalHistogram(),
            rng=np.random.default_rng(42),
        )

        rows.append(
            {
                "range_m": range_m,
                "target_bin": result.target_bin,
                "expected_target_return_photons": (
                    result.expected_target_return_photons
                ),
                "total_expected_detector_counts": float(
                    result.expected_detector_counts.sum()
                ),
                "total_observed_counts": int(
                    result.observed_counts.sum()
                ),
                "intercepted_fraction": (
                    result.intercepted_fraction
                ),
            }
        )

    summary = {
        "experiment": "journal_grade_physics_validation_v1",
        "histogram_bins": 4096,
        "bin_width_ns": 5.0,
        "maximum_range_m": (
            JournalHistogram().maximum_unambiguous_range_m
        ),
        "rows": rows,
    }

    output_path = output_directory / "summary.json"

    output_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
