import math

import numpy as np
import pytest

from physics.superposition import (
    relative_phase,
    state_coefficients,
    superposition_dynamics,
    superposition_probability_density,
)


def test_superposition_coefficients_preserve_total_probability():
    coefficient_a, coefficient_b = state_coefficients(0.35)

    assert coefficient_a ** 2 + coefficient_b ** 2 == pytest.approx(1.0)
    assert coefficient_b ** 2 == pytest.approx(0.35)


def test_interference_changes_density_with_relative_phase():
    wave_a = np.array([1.0, 1.0])
    wave_b = np.array([1.0, -1.0])

    phase_zero = superposition_probability_density(
        wave_a, wave_b, weight_b=0.5, relative_phase_rad=0.0,
    )
    phase_pi = superposition_probability_density(
        wave_a, wave_b, weight_b=0.5, relative_phase_rad=math.pi,
    )

    assert phase_zero == pytest.approx([2.0, 0.0], abs=1e-12)
    assert phase_pi == pytest.approx([0.0, 2.0], abs=1e-12)


def test_single_state_probability_is_stationary():
    wave_a = np.array([0.25, -0.5, 1.0])
    wave_b = np.array([2.0, 3.0, 4.0])

    density_zero = superposition_probability_density(
        wave_a, wave_b, weight_b=0.0, relative_phase_rad=0.0,
    )
    density_later = superposition_probability_density(
        wave_a, wave_b, weight_b=0.0, relative_phase_rad=1.7,
    )

    assert density_zero == pytest.approx(np.abs(wave_a) ** 2)
    assert density_later == pytest.approx(density_zero)


def test_energy_difference_defines_beat_frequency_and_period():
    dynamics = superposition_dynamics(-13.6057, -3.4014)

    assert dynamics.delta_energy_ev == pytest.approx(10.2043)
    assert dynamics.beat_frequency_hz == pytest.approx(2.467e15, rel=5e-4)
    assert dynamics.beat_period_s == pytest.approx(
        1.0 / dynamics.beat_frequency_hz
    )
    assert relative_phase(
        dynamics.delta_energy_ev, dynamics.beat_period_s,
    ) == pytest.approx(0.0, abs=1e-12)


def test_degenerate_states_have_stationary_relative_phase():
    dynamics = superposition_dynamics(-3.4, -3.4)

    assert dynamics.is_stationary
    assert math.isinf(dynamics.beat_period_s)
    assert relative_phase(0.0, 100.0, 0.7) == pytest.approx(0.7)


@pytest.mark.parametrize("weight", [-0.1, 1.1, float("nan")])
def test_invalid_superposition_weight_is_rejected(weight):
    with pytest.raises(ValueError):
        state_coefficients(weight)
