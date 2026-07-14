"""Run the final short-communication photon-allocation experiment."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

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
from src.policies.information_ablation_policy import (
    entropy_contrast_policy,
    entropy_only_policy,
    full_information_policy,
)
from src.policies.information_gain_policy import InformationGainPolicy
from src.policies.random_policy import RandomPolicy
from src.policies.uniform_policy import UniformPolicy


REGIONS = [
    RegionDefinition(500.0, 0.20),
    RegionDefinition(1000.0, 0.18),
    RegionDefinition(1500.0, 0.15),
    RegionDefinition(2000.0, 0.12),
    RegionDefinition(2500.0, 0.10),
    RegionDefinition(3000.0, 0.08),
]


def current_git_commit() -> str:
    """Return repository commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def load_config(path: Path) -> dict[str, Any]:
    """Load experiment YAML configuration."""
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping.")

    return config


def make_policy(name: str, seed: int):
    """Construct one allocation policy."""
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

    if name == "ablation_entropy_only":
        return entropy_only_policy()

    if name == "ablation_entropy_contrast":
        return entropy_contrast_policy()

    if name == "ablation_full_score":
        return full_information_policy()

    raise ValueError(f"Unknown policy: {name}")


def build_environment(
    *,
    target_region_index: int,
    seed: int,
    total_photon_budget: float,
    evaluation: dict[str, Any],
    atmosphere_config: dict[str, Any],
) -> AdaptiveLidarEnvironment:
    """Build one reproducible sensing environment."""
    maximum_budget_steps = int(
        total_photon_budget
        // float(evaluation["photons_per_action"])
    )

    configured_steps = int(evaluation["maximum_steps"])

    return AdaptiveLidarEnvironment(
        regions=REGIONS,
        target_region_index=target_region_index,
        config=AdaptiveEnvironmentConfig(
            total_photon_budget=total_photon_budget,
            photons_per_action=float(
                evaluation["photons_per_action"]
            ),
            pulses_per_action=int(
                evaluation["pulses_per_action"]
            ),
            initial_target_probability=0.10,
            detection_threshold=float(
                evaluation["detection_threshold"]
            ),
            maximum_steps=min(
                configured_steps,
                maximum_budget_steps,
            ),
        ),
        laser=JournalLaser(),
        receiver=JournalReceiver(),
        detector=SPADParameters(),
        atmosphere=AtmosphericChannel(
            extinction_coefficient_per_km=float(
                atmosphere_config[
                    "extinction_coefficient_per_km"
                ]
            ),
            volume_backscatter_coefficient_per_m=float(
                atmosphere_config[
                    "volume_backscatter_coefficient_per_m"
                ]
            ),
            background_photons_per_bin_per_pulse=float(
                atmosphere_config[
                    "background_photons_per_bin_per_pulse"
                ]
            ),
        ),
        histogram=JournalHistogram(
            num_bins=4096,
            bin_width_ns=5.0,
        ),
        seed=seed,
    )


def deterministic_episode_seed(
    *,
    base_seed: int,
    atmosphere_index: int,
    budget_index: int,
    target_index: int,
) -> int:
    """Create a matched seed shared by all policies."""
    return int(
        base_seed
        + atmosphere_index * 100_000
        + budget_index * 10_000
        + (target_index + 1) * 100
    )


def main() -> None:
    """Run all policies over the final experimental matrix."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiment/final_short_comm.yaml"
        ),
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a very small validation matrix.",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    evaluation = config["evaluation"]
    atmosphere_configs = config["atmospheres"]

    policy_names = list(config["policies"])
    policy_names.extend(config.get("ablations", []))

    seeds = list(evaluation["seeds"])
    target_indices = list(
        evaluation["target_region_indices"]
    )

    if evaluation.get("include_target_absent", False):
        target_indices.append(
            int(evaluation["target_absent_index"])
        )

    photon_budgets = dict(
        evaluation["photon_budgets"]
    )

    if args.smoke_test:
        seeds = seeds[:1]
        target_indices = [
            target_indices[0],
            int(evaluation["target_absent_index"]),
        ]
        photon_budgets = {
            "medium": photon_budgets["medium"]
        }
        atmosphere_configs = {
            "clear": atmosphere_configs["clear"]
        }
        policy_names = [
            "uniform",
            "information_gain",
            "ablation_full_score",
        ]

    output_directory = Path(
        config["outputs"]["directory"]
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    episodes_path = (
        output_directory
        / config["outputs"]["episodes_file"]
    )

    total_episodes = (
        len(policy_names)
        * len(seeds)
        * len(target_indices)
        * len(photon_budgets)
        * len(atmosphere_configs)
    )

    rows: list[dict[str, Any]] = []

    progress = tqdm(
        total=total_episodes,
        desc="Final photon-allocation experiment",
    )

    for policy_index, policy_name in enumerate(
        policy_names
    ):
        for atmosphere_index, (
            atmosphere_name,
            atmosphere_config,
        ) in enumerate(atmosphere_configs.items()):
            for budget_index, (
                budget_name,
                budget_value,
            ) in enumerate(photon_budgets.items()):
                total_budget = float(budget_value)

                for base_seed in seeds:
                    for target_index in target_indices:
                        episode_seed = (
                            deterministic_episode_seed(
                                base_seed=int(base_seed),
                                atmosphere_index=(
                                    atmosphere_index
                                ),
                                budget_index=budget_index,
                                target_index=int(
                                    target_index
                                ),
                            )
                        )

                        environment = build_environment(
                            target_region_index=int(
                                target_index
                            ),
                            seed=episode_seed,
                            total_photon_budget=(
                                total_budget
                            ),
                            evaluation=evaluation,
                            atmosphere_config=(
                                atmosphere_config
                            ),
                        )

                        policy = make_policy(
                            policy_name,
                            episode_seed,
                        )

                        result = run_episode(
                            policy=policy,
                            environment=environment,
                            seed=episode_seed,
                        )

                        row = episode_result_to_dict(
                            result
                        )

                        row.update(
                            {
                                "base_seed": int(
                                    base_seed
                                ),
                                "episode_seed": (
                                    episode_seed
                                ),
                                "atmosphere": (
                                    atmosphere_name
                                ),
                                "budget_name": (
                                    budget_name
                                ),
                                "total_photon_budget": (
                                    total_budget
                                ),
                                "target_range_m": (
                                    REGIONS[
                                        int(target_index)
                                    ].range_m
                                    if int(target_index)
                                    >= 0
                                    else None
                                ),
                                "git_commit": (
                                    current_git_commit()
                                ),
                            }
                        )

                        rows.append(row)
                        progress.update(1)

    progress.close()

    if not rows:
        raise RuntimeError(
            "The experiment generated no episode rows."
        )

    with episodes_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Completed episodes: {len(rows)}")
    print(f"Saved: {episodes_path}")


if __name__ == "__main__":
    main()
