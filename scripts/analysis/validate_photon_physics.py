"""Validate expected physical trends in the single-photon LiDAR model.

This script evaluates:

1. Signal photons versus range.
2. Signal photons versus atmospheric extinction.
3. Signal photons versus target reflectivity.
4. Monte Carlo agreement between expected and observed photon counts.

Outputs are saved as CSV files and publication-ready figures.
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.physics.photon_return_model import (
    AtmosphereParameters,
    DetectorParameters,
    HistogramParameters,
    LaserParameters,
    ReceiverParameters,
    TargetParameters,
    expected_detected_signal_photons,
    simulate_photon_return,
)


OUTPUT_DIRECTORY = Path("results/processed/physics_validation")
FIGURE_DIRECTORY = Path("figures/simulation")


def get_git_commit() -> str:
    """Return the current Git commit hash when available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def save_rows(path: Path, rows: list[dict[str, float]]) -> None:
    """Save a list of dictionaries to a CSV file."""
    if not rows:
        raise ValueError("No rows were provided.")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def evaluate_range_sweep() -> list[dict[str, float]]:
    """Evaluate detected signal photons across target ranges."""
    rows: list[dict[str, float]] = []

    laser = LaserParameters(emitted_photons=1e12)
    receiver = ReceiverParameters()
    detector = DetectorParameters()
    atmosphere = AtmosphereParameters(
        extinction_coefficient_per_km=0.10
    )

    for range_m in np.linspace(100.0, 3000.0, 60):
        expected_signal = expected_detected_signal_photons(
            laser=laser,
            receiver=receiver,
            detector=detector,
            atmosphere=atmosphere,
            target=TargetParameters(
                range_m=float(range_m),
                reflectivity=0.20,
            ),
        )

        rows.append(
            {
                "range_m": float(range_m),
                "expected_signal_photons": expected_signal,
            }
        )

    return rows


def evaluate_extinction_sweep() -> list[dict[str, float]]:
    """Evaluate detected signal photons across extinction conditions."""
    rows: list[dict[str, float]] = []

    laser = LaserParameters(emitted_photons=1e12)
    receiver = ReceiverParameters()
    detector = DetectorParameters()
    target = TargetParameters(
        range_m=1000.0,
        reflectivity=0.20,
    )

    for extinction in np.linspace(0.0, 2.0, 50):
        expected_signal = expected_detected_signal_photons(
            laser=laser,
            receiver=receiver,
            detector=detector,
            atmosphere=AtmosphereParameters(
                extinction_coefficient_per_km=float(extinction)
            ),
            target=target,
        )

        rows.append(
            {
                "extinction_coefficient_per_km": float(extinction),
                "expected_signal_photons": expected_signal,
            }
        )

    return rows


def evaluate_reflectivity_sweep() -> list[dict[str, float]]:
    """Evaluate detected signal photons across target reflectivities."""
    rows: list[dict[str, float]] = []

    laser = LaserParameters(emitted_photons=1e12)
    receiver = ReceiverParameters()
    detector = DetectorParameters()
    atmosphere = AtmosphereParameters(
        extinction_coefficient_per_km=0.10
    )

    for reflectivity in np.linspace(0.01, 0.90, 50):
        expected_signal = expected_detected_signal_photons(
            laser=laser,
            receiver=receiver,
            detector=detector,
            atmosphere=atmosphere,
            target=TargetParameters(
                range_m=1000.0,
                reflectivity=float(reflectivity),
            ),
        )

        rows.append(
            {
                "reflectivity": float(reflectivity),
                "expected_signal_photons": expected_signal,
            }
        )

    return rows


def evaluate_monte_carlo(
    trials: int = 5000,
    seed: int = 42,
) -> dict[str, float]:
    """Compare Monte Carlo observations against the expected photon count."""
    rng = np.random.default_rng(seed)

    parameters = {
        "laser": LaserParameters(emitted_photons=1e12),
        "receiver": ReceiverParameters(),
        "detector": DetectorParameters(),
        "atmosphere": AtmosphereParameters(
            extinction_coefficient_per_km=0.10,
            background_photons_per_bin=0.02,
        ),
        "target": TargetParameters(
            range_m=1000.0,
            reflectivity=0.20,
        ),
        "histogram": HistogramParameters(
            num_bins=4096,
            bin_width_ns=5.0,
        ),
    }

    observed_totals: list[float] = []
    expected_total = None

    for _ in range(trials):
        result = simulate_photon_return(
            **parameters,
            rng=rng,
        )

        observed_totals.append(float(result.photon_counts.sum()))

        if expected_total is None:
            expected_total = float(result.expected_counts.sum())

    observed_array = np.asarray(observed_totals)

    if expected_total is None:
        raise RuntimeError("Monte Carlo simulation produced no observations.")

    relative_error = abs(observed_array.mean() - expected_total) / expected_total

    return {
        "trials": float(trials),
        "seed": float(seed),
        "expected_total_counts": expected_total,
        "observed_mean_counts": float(observed_array.mean()),
        "observed_variance_counts": float(observed_array.var(ddof=1)),
        "relative_mean_error": float(relative_error),
    }


def plot_range_sweep(rows: list[dict[str, float]]) -> None:
    """Create the range-response figure."""
    ranges = [row["range_m"] for row in rows]
    signals = [row["expected_signal_photons"] for row in rows]

    plt.figure(figsize=(7, 5))
    plt.plot(ranges, signals)
    plt.xlabel("Target range (m)")
    plt.ylabel("Expected detected signal photons")
    plt.title("Detected photon return versus target range")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIRECTORY / "signal_vs_range.png",
        dpi=300,
    )
    plt.close()


def plot_extinction_sweep(rows: list[dict[str, float]]) -> None:
    """Create the atmospheric-extinction figure."""
    extinctions = [
        row["extinction_coefficient_per_km"] for row in rows
    ]
    signals = [row["expected_signal_photons"] for row in rows]

    plt.figure(figsize=(7, 5))
    plt.plot(extinctions, signals)
    plt.xlabel("Extinction coefficient (km⁻¹)")
    plt.ylabel("Expected detected signal photons")
    plt.title("Detected photon return versus atmospheric extinction")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIRECTORY / "signal_vs_extinction.png",
        dpi=300,
    )
    plt.close()


def plot_reflectivity_sweep(rows: list[dict[str, float]]) -> None:
    """Create the target-reflectivity figure."""
    reflectivities = [row["reflectivity"] for row in rows]
    signals = [row["expected_signal_photons"] for row in rows]

    plt.figure(figsize=(7, 5))
    plt.plot(reflectivities, signals)
    plt.xlabel("Target reflectivity")
    plt.ylabel("Expected detected signal photons")
    plt.title("Detected photon return versus target reflectivity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIRECTORY / "signal_vs_reflectivity.png",
        dpi=300,
    )
    plt.close()


def main() -> None:
    """Run all physics-validation experiments."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    range_rows = evaluate_range_sweep()
    extinction_rows = evaluate_extinction_sweep()
    reflectivity_rows = evaluate_reflectivity_sweep()
    monte_carlo_results = evaluate_monte_carlo()

    save_rows(
        OUTPUT_DIRECTORY / "range_sweep.csv",
        range_rows,
    )

    save_rows(
        OUTPUT_DIRECTORY / "extinction_sweep.csv",
        extinction_rows,
    )

    save_rows(
        OUTPUT_DIRECTORY / "reflectivity_sweep.csv",
        reflectivity_rows,
    )

    plot_range_sweep(range_rows)
    plot_extinction_sweep(extinction_rows)
    plot_reflectivity_sweep(reflectivity_rows)

    metadata = {
        "git_commit": get_git_commit(),
        "experiment": "physics_validation_v1",
        "monte_carlo": monte_carlo_results,
    }

    with (
        OUTPUT_DIRECTORY / "validation_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("Physics validation completed.")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
