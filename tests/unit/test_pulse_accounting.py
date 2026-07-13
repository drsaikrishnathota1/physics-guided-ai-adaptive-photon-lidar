from src.physics.pulse_accounting import PulseSchedule


def test_pulse_accounting() -> None:
    schedule = PulseSchedule(
        photons_per_pulse=100.0,
        repetition_rate_hz=1000.0,
        dwell_time_s=0.1,
        active_gate_width_ns=100.0,
    )

    assert schedule.pulses_per_dwell == 100
    assert schedule.total_transmitted_photons == 10_000.0
    assert schedule.total_active_detector_time_s > 0.0
