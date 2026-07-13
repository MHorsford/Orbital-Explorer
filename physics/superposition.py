"""Superposição coerente e evolução temporal de dois estados orbitais."""

from dataclasses import dataclass
import math

import numpy as np

from physics.constants import E_CHARGE, H, HBAR


@dataclass(frozen=True)
class SuperpositionDynamics:
    """Escalas físicas associadas à diferença de energia entre dois estados."""

    energy_a_ev: float
    energy_b_ev: float
    delta_energy_ev: float
    beat_frequency_hz: float
    beat_period_s: float

    @property
    def is_stationary(self) -> bool:
        return self.beat_frequency_hz <= 1e-12


def state_coefficients(weight_b: float = 0.5):
    """Retorna coeficientes reais normalizados para pesos probabilísticos."""
    if not math.isfinite(weight_b) or not 0.0 <= weight_b <= 1.0:
        raise ValueError("O peso do estado B deve estar entre 0 e 1")
    return math.sqrt(1.0 - weight_b), math.sqrt(weight_b)


def superposition_dynamics(
        energy_a_ev: float, energy_b_ev: float,
) -> SuperpositionDynamics:
    """Calcula frequência e período do batimento quântico entre dois estados."""
    if not math.isfinite(energy_a_ev) or not math.isfinite(energy_b_ev):
        raise ValueError("As energias dos estados devem ser finitas")
    delta_energy_ev = energy_b_ev - energy_a_ev
    beat_frequency_hz = abs(delta_energy_ev) * E_CHARGE / H
    beat_period_s = (
        math.inf if beat_frequency_hz <= 1e-12 else 1.0 / beat_frequency_hz
    )
    return SuperpositionDynamics(
        energy_a_ev=energy_a_ev,
        energy_b_ev=energy_b_ev,
        delta_energy_ev=delta_energy_ev,
        beat_frequency_hz=beat_frequency_hz,
        beat_period_s=beat_period_s,
    )


def relative_phase(
        delta_energy_ev: float, physical_time_s: float,
        initial_phase_rad: float = 0.0,
) -> float:
    """Fase relativa de B em relação a A, reduzida ao intervalo [0, 2π)."""
    values = (delta_energy_ev, physical_time_s, initial_phase_rad)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Energia, tempo e fase devem ser finitos")
    phase = initial_phase_rad - (
        delta_energy_ev * E_CHARGE * physical_time_s / HBAR
    )
    return phase % (2.0 * math.pi)


def superposition_wavefunction(
        wave_a, wave_b, weight_b: float = 0.5,
        relative_phase_rad: float = 0.0,
):
    """Combina dois estados espaciais com uma fase relativa complexa."""
    wave_a = np.asarray(wave_a)
    wave_b = np.asarray(wave_b)
    if wave_a.shape != wave_b.shape:
        raise ValueError("Os dois estados devem ser avaliados no mesmo grid")
    if not math.isfinite(relative_phase_rad):
        raise ValueError("A fase relativa deve ser finita")
    coefficient_a, coefficient_b = state_coefficients(weight_b)
    return (
        coefficient_a * wave_a
        + coefficient_b * np.exp(1j * relative_phase_rad) * wave_b
    )


def superposition_probability_density(
        wave_a, wave_b, weight_b: float = 0.5,
        relative_phase_rad: float = 0.0,
):
    """Retorna |Ψ(t)|², incluindo o termo de interferência entre os estados."""
    wave = superposition_wavefunction(
        wave_a, wave_b, weight_b=weight_b,
        relative_phase_rad=relative_phase_rad,
    )
    return np.abs(wave) ** 2
