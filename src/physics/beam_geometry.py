"""Laser beam divergence and target-footprint coupling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BeamParameters:
    """Laser beam geometry."""

    full_angle_divergence_mrad: float = 0.5
    initial_beam_radius_m: float = 0.01
    transmitter_efficiency: float = 0.70

    def __post_init__(self) -> None:
        if self.full_angle_divergence_mrad < 0:
            raise ValueError(
                "full_angle_divergence_mrad cannot be negative."
            )

        if self.initial_beam_radius_m <= 0:
            raise ValueError("initial_beam_radius_m must be positive.")

        if not 0.0 <= self.transmitter_efficiency <= 1.0:
            raise ValueError(
                "transmitter_efficiency must be between 0 and 1."
            )


def beam_radius_m(
    range_m: float,
    beam: BeamParameters,
) -> float:
    """Return approximate beam radius at the target range.

    Divergence is treated as the full angular divergence.
    """
    if range_m < 0:
        raise ValueError("range_m cannot be negative.")

    half_angle_rad = beam.full_angle_divergence_mrad * 1.0e-3 / 2.0

    return float(
        beam.initial_beam_radius_m
        + range_m * np.tan(half_angle_rad)
    )


def beam_spot_area_m2(
    range_m: float,
    beam: BeamParameters,
) -> float:
    """Return circular beam footprint area at the target."""
    radius = beam_radius_m(range_m, beam)

    return float(np.pi * radius**2)


def target_intercept_fraction(
    range_m: float,
    projected_target_area_m2: float,
    beam: BeamParameters,
) -> float:
    """Return the fraction of the beam footprint intercepted by target."""
    if projected_target_area_m2 <= 0:
        raise ValueError("projected_target_area_m2 must be positive.")

    spot_area = beam_spot_area_m2(range_m, beam)

    return float(
        np.clip(projected_target_area_m2 / spot_area, 0.0, 1.0)
    )


def incidence_factor(incidence_angle_deg: float) -> float:
    """Return cosine incidence-angle coupling."""
    if not 0.0 <= incidence_angle_deg <= 90.0:
        raise ValueError(
            "incidence_angle_deg must be between 0 and 90."
        )

    return float(
        np.cos(np.deg2rad(incidence_angle_deg))
    )
