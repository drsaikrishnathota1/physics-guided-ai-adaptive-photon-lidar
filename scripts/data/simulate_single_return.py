"""Generate and save one example single-photon LiDAR histogram."""

from pathlib import Path

import numpy as np

from src.physics.photon_return_model import (
    AtmosphereParameters,
    DetectorParameters,
    HistogramParameters,
    LaserParameters,
    ReceiverParameters,
    TargetParameters,
    simulate_photon_return,
)


def main() -> None:
    output_directory = Path("results/raw")
    output_directory.mkdir(parents=True, exist_ok=True)

    result = simulate_photon_return(
        laser=LaserParameters(emitted_photons=1e12),
        receiver=ReceiverParameters(),
        detector=DetectorParameters(),
        atmosphere=AtmosphereParameters(
            extinction_coefficient_per_km=0.10,
            background_photons_per_bin=0.02,
        ),
        target=TargetParameters(
            range_m=1000.0,
            reflectivity=0.20,
        ),
        histogram=HistogramParameters(
            num_bins=2048,
            bin_width_ns=5.0,
        ),
        rng=np.random.default_rng(42),
    )

    output_path = output_directory / "example_photon_return.npz"

    np.savez_compressed(
        output_path,
        photon_counts=result.photon_counts,
        expected_counts=result.expected_counts,
        expected_signal_counts=result.expected_signal_counts,
        expected_background_counts=result.expected_background_counts,
        target_bin=result.target_bin,
        round_trip_time_s=result.round_trip_time_s,
        expected_detected_signal_photons=(
            result.expected_detected_signal_photons
        ),
    )

    print(f"Saved simulation to: {output_path}")
    print(f"Target bin: {result.target_bin}")
    print(
        "Expected detected signal photons:",
        result.expected_detected_signal_photons,
    )


if __name__ == "__main__":
    main()
