"""Compare five non-AI photon-allocation policies."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import numpy as np

from src.environment.adaptive_lidar_env import (
    AdaptiveEnvironmentConfig,
    AdaptiveLidarEnvironment,
    RegionDefinition,
)
from src.evaluation.baseline_runner import (
    episode_result_to_dict,
    run_episode,
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


OUTPUT_DIRECTORY = Path(
    "results/processed/baseline_comparison"
)


def git_commit() -> str:
    """Return the current Git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def build_environment(
    *,
    target_region_index: int,
    seed: int,
) -> AdaptiveLidarEnvironment:
    """Construct one shared baseline-evaluation environment."""
    regions = [
        RegionDefinition(500.0, 0.20),
        RegionDefinition(1000.0, 0.18),
        RegionDefinition(1500.0, 0.15),
        RegionDefinition(2000.0, 0.12),
        RegionDefinition(2500.0, 0.10),
        RegionDefinition(3000.0, 0.08),
    ]

    return AdaptiveLidarEnvironment(
        regions=regions,
        target_region_index=target_region_index,
        config=AdaptiveEnvironmentConfig(
            total_photon_budget=5.0e12,
            photons_per_action=2.5e11,
            pulses_per_action=1000,
            initial_target_probability=0.10,
            detection_threshold=0.90,
            maximum_steps=20,
        ),
        laser=JournalLaser(),
        receiver=JournalReceiver(),
        detector=SPADParameters(),
        atmosphere=AtmosphericChannel(
            extinction_coefficient_per_km=0.10,
            volume_backscatter_coefficient_per_m=1.0e-7,
            background_photons_per_bin_per_pulse=1.0e-5,
        ),
        histogram=JournalHistogram(
            num_bins=4096,
            bin_width_ns=5.0,
        ),
        seed=seed,
    )


def make_policy(
    name: str,
    seed: int,
):
    """Construct a policy by name."""
    if name == "uniform":
        return UniformPolicy()

    if name == "random":
        return RandomPolicy(seed=seed)

    if name == "greedy":
        return GreedyPolicy()

    if name == "entropy":
        return EntropyPolicy()

    if name == "information_gain":
        return InformationGainPolicy()

    raise ValueError(f"Unknown policy: {name}")


def main() -> None:
    """Run baseline-policy comparison."""
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    policy_names = [
        "uniform",
        "random",
        "greedy",
        "entropy",
        "information_gain",
    ]

    seeds = [11, 22, 33, 44, 55]
    target_regions = [0, 1, 2, 3, 4, 5]

    episode_rows: list[dict] = []

    for policy_name in policy_names:
        for seed in seeds:
            for target_region_index in target_regions:
                environment = build_environment(
                    target_region_index=target_region_index,
                    seed=seed,
                )

                policy = make_policy(
                    policy_name,
                    seed,
                )

                result = run_episode(
                    policy=policy,
                    environment=environment,
                    seed=seed,
                )

                row = episode_result_to_dict(result)
                episode_rows.append(row)

                print(
                    policy_name,
                    seed,
                    target_region_index,
                    result.detected,
                    result.steps,
                    result.photons_used,
                )

    csv_path = OUTPUT_DIRECTORY / "episodes.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=episode_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(episode_rows)

    summary: dict[str, dict[str, float]] = {}

    for policy_name in policy_names:
        policy_rows = [
            row
            for row in episode_rows
            if row["policy"] == policy_name
        ]

        detected_rows = [
            row
            for row in policy_rows
            if row["detected"]
        ]

        detection_rate = (
            len(detected_rows)
            / len(policy_rows)
        )

        mean_photons_used = float(
            np.mean(
                [
                    row["photons_used"]
                    for row in policy_rows
                ]
            )
        )

        mean_photons_when_detected = (
            float(
                np.mean(
                    [
                        row["photons_used"]
                        for row in detected_rows
                    ]
                )
            )
            if detected_rows
            else float("nan")
        )

        mean_steps = float(
            np.mean(
                [
                    row["steps"]
                    for row in policy_rows
                ]
            )
        )

        false_focus_rate = float(
            np.mean(
                [
                    row[
                        "maximum_non_target_posterior"
                    ]
                    >= 0.90
                    for row in policy_rows
                ]
            )
        )

        summary[policy_name] = {
            "episodes": float(len(policy_rows)),
            "detection_rate": detection_rate,
            "mean_photons_used": mean_photons_used,
            "mean_photons_when_detected": (
                mean_photons_when_detected
            ),
            "mean_steps": mean_steps,
            "false_focus_rate": false_focus_rate,
        }

    output_summary = {
        "experiment": "baseline_policy_comparison_v1",
        "git_commit": git_commit(),
        "seeds": seeds,
        "target_regions": target_regions,
        "summary": summary,
    }

    summary_path = OUTPUT_DIRECTORY / "summary.json"

    summary_path.write_text(
        json.dumps(output_summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(output_summary, indent=2))
    print(f"Saved: {csv_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
