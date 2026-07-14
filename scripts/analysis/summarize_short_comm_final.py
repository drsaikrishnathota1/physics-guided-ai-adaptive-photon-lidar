"""Summarize final short-communication experiment results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


Z_95 = 1.959963984540054


def wilson_interval(
    successes: int,
    trials: int,
) -> tuple[float, float]:
    """Return a 95% Wilson confidence interval."""
    if trials <= 0:
        return float("nan"), float("nan")

    proportion = successes / trials
    denominator = 1.0 + (Z_95**2 / trials)

    center = (
        proportion
        + Z_95**2 / (2.0 * trials)
    ) / denominator

    half_width = (
        Z_95
        * np.sqrt(
            proportion * (1.0 - proportion) / trials
            + Z_95**2 / (4.0 * trials**2)
        )
        / denominator
    )

    return (
        float(max(0.0, center - half_width)),
        float(min(1.0, center + half_width)),
    )


def summarize_group(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate detection, false-alarm and resource metrics."""
    present = frame[frame["target_present"]]
    absent = frame[~frame["target_present"]]

    detections = int(present["detected"].sum())
    present_trials = int(len(present))

    false_alarms = int(absent["false_alarm"].sum())
    absent_trials = int(len(absent))

    detection_rate = (
        detections / present_trials
        if present_trials
        else float("nan")
    )

    false_alarm_rate = (
        false_alarms / absent_trials
        if absent_trials
        else float("nan")
    )

    detection_ci_low, detection_ci_high = (
        wilson_interval(
            detections,
            present_trials,
        )
    )

    false_alarm_ci_low, false_alarm_ci_high = (
        wilson_interval(
            false_alarms,
            absent_trials,
        )
    )

    true_negatives = absent_trials - false_alarms
    balanced_accuracy = (
        0.5
        * (
            detection_rate
            + true_negatives / absent_trials
        )
        if present_trials and absent_trials
        else float("nan")
    )

    return {
        "episodes": int(len(frame)),
        "target_present_episodes": present_trials,
        "target_absent_episodes": absent_trials,
        "detections": detections,
        "false_alarms": false_alarms,
        "detection_rate": float(detection_rate),
        "detection_ci95_low": detection_ci_low,
        "detection_ci95_high": detection_ci_high,
        "false_alarm_rate": float(false_alarm_rate),
        "false_alarm_ci95_low": false_alarm_ci_low,
        "false_alarm_ci95_high": false_alarm_ci_high,
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "mean_photons_used": float(
            frame["photons_used"].mean()
        ),
        "median_photons_used": float(
            frame["photons_used"].median()
        ),
        "mean_steps": float(
            frame["steps"].mean()
        ),
        "mean_photons_detected_cases": (
            float(
                present.loc[
                    present["detected"],
                    "photons_used",
                ].mean()
            )
            if detections
            else float("nan")
        ),
    }


def grouped_summary(
    frame: pd.DataFrame,
    grouping_columns: list[str],
) -> pd.DataFrame:
    """Summarize results by specified columns."""
    rows: list[dict[str, Any]] = []

    group_key: str | list[str]
    group_key = (
        grouping_columns[0]
        if len(grouping_columns) == 1
        else grouping_columns
    )

    for keys, group in frame.groupby(
        group_key,
        dropna=False,
        sort=True,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {
            column: value
            for column, value in zip(
                grouping_columns,
                keys,
            )
        }

        row.update(summarize_group(group))
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    """Create all final experiment summaries."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiment/final_short_comm.yaml"
        ),
    )

    args = parser.parse_args()

    with args.config.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    output_directory = Path(
        config["outputs"]["directory"]
    )

    episodes_path = (
        output_directory
        / config["outputs"]["episodes_file"]
    )

    frame = pd.read_csv(episodes_path)

    for column in [
        "target_present",
        "detected",
        "false_alarm",
    ]:
        frame[column] = (
            frame[column]
            .astype(str)
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                }
            )
        )

    overall = grouped_summary(
        frame,
        ["policy"],
    )

    range_summary = grouped_summary(
        frame[frame["target_present"]],
        [
            "policy",
            "atmosphere",
            "budget_name",
            "target_range_m",
        ],
    )

    budget_summary = grouped_summary(
        frame,
        [
            "policy",
            "budget_name",
        ],
    )

    atmosphere_summary = grouped_summary(
        frame,
        [
            "policy",
            "atmosphere",
        ],
    )

    ablation_frame = frame[
        frame["policy"].str.startswith(
            "ablation_"
        )
    ]

    ablation_summary = grouped_summary(
        ablation_frame,
        [
            "policy",
            "atmosphere",
            "budget_name",
        ],
    )

    overall.to_csv(
        output_directory / "policy_summary.csv",
        index=False,
    )

    range_summary.to_csv(
        output_directory
        / config["outputs"]["range_summary_file"],
        index=False,
    )

    budget_summary.to_csv(
        output_directory
        / config["outputs"]["budget_summary_file"],
        index=False,
    )

    atmosphere_summary.to_csv(
        output_directory
        / config["outputs"][
            "atmosphere_summary_file"
        ],
        index=False,
    )

    ablation_summary.to_csv(
        output_directory
        / config["outputs"]["ablation_summary_file"],
        index=False,
    )

    summary_payload = {
        "experiment": config["experiment"]["name"],
        "episodes": int(len(frame)),
        "git_commits": sorted(
            frame["git_commit"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        "policy_summary": overall.to_dict(
            orient="records"
        ),
    }

    summary_path = (
        output_directory
        / config["outputs"]["summary_file"]
    )

    summary_path.write_text(
        json.dumps(
            summary_payload,
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )

    print(overall.to_string(index=False))
    print()
    print(f"Saved summaries to: {output_directory}")


if __name__ == "__main__":
    main()
