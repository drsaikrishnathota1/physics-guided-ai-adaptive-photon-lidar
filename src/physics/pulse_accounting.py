"""Pulse, dwell, gate, and photon-budget accounting for pulsed LiDAR."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PulseSchedule:
    """Defines one LiDAR measurement dwell."""

    photons_per_pulse: float = 1.0e8
    repetition_rate_hz: float = 100_000.0
    dwell_time_s: float = 0.01
    active_gate_width_ns: float = 20_000.0

    def __post_init__(self) -> None:
        if self.photons_per_pulse <= 0:
            raise ValueError("photons_per_pulse must be positive.")

        if self.repetition_rate_hz <= 0:
            raise ValueError("repetition_rate_hz must be positive.")

        if self.dwell_time_s <= 0:
            raise ValueError("dwell_time_s must be positive.")

        if self.active_gate_width_ns <= 0:
            raise ValueError("active_gate_width_ns must be positive.")

    @property
    def pulses_per_dwell(self) -> int:
        """Return the number of emitted pulses in one dwell."""
        return max(
            1,
            int(round(self.repetition_rate_hz * self.dwell_time_s)),
        )

    @property
    def total_transmitted_photons(self) -> float:
        """Return the total emitted photon budget in one dwell."""
        return self.photons_per_pulse * self.pulses_per_dwell

    @property
    def pulse_period_s(self) -> float:
        """Return the temporal spacing between pulses."""
        return 1.0 / self.repetition_rate_hz

    @property
    def active_gate_width_s(self) -> float:
        """Return active detector gate width in seconds."""
        return self.active_gate_width_ns * 1.0e-9

    @property
    def total_active_detector_time_s(self) -> float:
        """Return accumulated detector-active time over the dwell."""
        return self.pulses_per_dwell * self.active_gate_width_s
