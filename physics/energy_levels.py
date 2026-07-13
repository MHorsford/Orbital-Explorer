"""Estimativas didáticas de níveis e transições eletrônicas."""

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Tuple

from physics.constants import (
    C, ELECTRON_SPIN_G, E_CHARGE, H, MU_B_EV_T, RY_EV,
)
from physics.screening import (
    get_orbital_sequence, max_electrons_in_subshell,
    orbital_state_effective_charge,
)


@dataclass(frozen=True)
class EnergyLevel:
    n: int
    l: int
    occupancy: int
    energy_ev: float
    z_eff: float
    aufbau_index: int

    @property
    def key(self) -> Tuple[int, int]:
        return self.n, self.l


@dataclass(frozen=True)
class TransitionResult:
    initial: Tuple[int, int]
    final: Tuple[int, int]
    initial_m: int
    final_m: int
    initial_spin: float
    final_spin: float
    delta_n: int
    delta_l: int
    delta_m: int
    delta_spin: float
    zero_field_delta_energy_ev: float
    zero_field_photon_energy_ev: float
    zero_field_frequency_hz: float
    zero_field_wavelength_nm: float
    magnetic_field_t: float
    initial_zeeman_shift_ev: float
    final_zeeman_shift_ev: float
    transition_zeeman_shift_ev: float
    zeeman_frequency_shift_hz: float
    delta_energy_ev: float
    photon_energy_ev: float
    frequency_hz: float
    wavelength_nm: float
    process: str
    dipole_l_allowed: bool
    dipole_m_allowed: bool
    spin_allowed: bool
    parity_changes: bool
    electric_dipole_allowed: bool
    spectral_region: str


def relevant_subshells(electron_count: int, extra_levels: int = 4) -> List[Tuple[int, int]]:
    """Subníveis necessários para acomodar N elétrons mais níveis seguintes."""
    sequence = get_orbital_sequence()
    capacity = 0
    last_index = 0
    for index, (_, l) in enumerate(sequence):
        capacity += max_electrons_in_subshell(l)
        last_index = index
        if capacity >= max(1, electron_count):
            break
    return sequence[:min(len(sequence), last_index + 1 + extra_levels)]


def diagram_subshells(
        electron_count: int, selected: Tuple[int, int] = None,
        extra_levels: int = 4,
        configuration: Dict[Tuple[int, int], int] = None,
) -> List[Tuple[int, int]]:
    """Inclui no diagrama o subnível selecionado e seu contexto energético."""
    sequence = get_orbital_sequence()
    subshells = relevant_subshells(electron_count, extra_levels)
    if selected is not None and selected not in subshells:
        if selected in sequence:
            subshells = sequence[:max(len(subshells), sequence.index(selected) + 1)]
        else:
            subshells.append(selected)

    for key, occupancy in (configuration or {}).items():
        if occupancy and key not in subshells:
            subshells.append(key)
    return sorted(
        subshells,
        key=lambda key: (
            sequence.index(key) if key in sequence else len(sequence) + sum(key)
        ),
    )


def approximate_orbital_energy(
        Z: int, n: int, l: int, electron_count: int = None,
        configuration: Dict[Tuple[int, int], int] = None,
) -> Tuple[float, float]:
    """Retorna (E, Z_eff) pela aproximação hidrogenoide E=-Ry·Z_eff²/n²."""
    z_eff = orbital_state_effective_charge(
        Z, n, l, electron_count=electron_count, configuration=configuration,
    )
    energy_ev = -RY_EV * (z_eff ** 2) / (n ** 2)
    return energy_ev, z_eff


def build_energy_levels(
        Z: int, electron_count: int,
        configuration: Dict[Tuple[int, int], int],
        subshells: Iterable[Tuple[int, int]] = None,
) -> List[EnergyLevel]:
    """Constrói níveis aproximados para a configuração eletrônica atual."""
    sequence = get_orbital_sequence()
    selected = list(subshells or relevant_subshells(electron_count))
    levels = []
    for n, l in selected:
        energy_ev, z_eff = approximate_orbital_energy(
            Z, n, l, electron_count=electron_count,
            configuration=configuration,
        )
        aufbau_index = (
            sequence.index((n, l))
            if (n, l) in sequence else len(sequence) + n + l
        )
        levels.append(EnergyLevel(
            n=n,
            l=l,
            occupancy=configuration.get((n, l), 0),
            energy_ev=energy_ev,
            z_eff=z_eff,
            aufbau_index=aufbau_index,
        ))
    return levels


def spectral_region(wavelength_nm: float) -> str:
    if wavelength_nm < 0.01:
        return "raios gama"
    if wavelength_nm < 10:
        return "raios X"
    if wavelength_nm < 380:
        return "ultravioleta"
    if wavelength_nm < 750:
        return "visível"
    if wavelength_nm < 1_000_000:
        return "infravermelho"
    return "micro-ondas ou rádio"


def zeeman_shift_ev(
        magnetic_field_t: float, magnetic_number: int,
        spin: float, spin_g_factor: float = ELECTRON_SPIN_G,
) -> float:
    """Deslocamento Zeeman linear para |n,l,mₗ,mₛ⟩ com B paralelo a z."""
    if not math.isfinite(magnetic_field_t):
        raise ValueError("O campo magnético deve ser um número finito")
    if spin not in (-0.5, 0.5):
        raise ValueError("mₛ deve ser +½ ou −½")
    return MU_B_EV_T * magnetic_field_t * (
        magnetic_number + spin_g_factor * spin
    )


def calculate_transition(
        initial: EnergyLevel, final: EnergyLevel,
        initial_m: int = 0, final_m: int = 0,
        initial_spin: float = 0.5, final_spin: float = 0.5,
        magnetic_field_t: float = 0.0,
) -> TransitionResult:
    """Calcula uma transição e testa regras E1 para estados de um elétron."""
    if initial.key == final.key:
        raise ValueError("Os níveis inicial e final devem ser diferentes")
    if abs(initial_m) > initial.l or abs(final_m) > final.l:
        raise ValueError("mₗ deve satisfazer −l ≤ mₗ ≤ l")
    if initial_spin not in (-0.5, 0.5) or final_spin not in (-0.5, 0.5):
        raise ValueError("mₛ deve ser +½ ou −½")

    zero_field_delta = final.energy_ev - initial.energy_ev
    zero_field_photon_energy = abs(zero_field_delta)
    initial_zeeman = zeeman_shift_ev(
        magnetic_field_t, initial_m, initial_spin,
    )
    final_zeeman = zeeman_shift_ev(
        magnetic_field_t, final_m, final_spin,
    )
    transition_zeeman = final_zeeman - initial_zeeman
    delta = zero_field_delta + transition_zeeman
    photon_energy = abs(delta)
    if photon_energy <= 1e-12:
        raise ValueError("A diferença de energia é pequena demais para a estimativa")
    if zero_field_photon_energy <= 1e-12:
        zero_field_frequency = 0.0
        zero_field_wavelength_nm = math.inf
    else:
        zero_field_energy_joule = zero_field_photon_energy * E_CHARGE
        zero_field_frequency = zero_field_energy_joule / H
        zero_field_wavelength_nm = (C / zero_field_frequency) * 1e9
    energy_joule = photon_energy * E_CHARGE
    frequency = energy_joule / H
    wavelength_nm = (C / frequency) * 1e9
    delta_n = final.n - initial.n
    delta_l = final.l - initial.l
    delta_m = final_m - initial_m
    delta_spin = final_spin - initial_spin
    dipole_l_allowed = abs(delta_l) == 1
    dipole_m_allowed = abs(delta_m) <= 1
    spin_allowed = abs(delta_spin) < 1e-12
    parity_changes = (initial.l - final.l) % 2 != 0

    return TransitionResult(
        initial=initial.key,
        final=final.key,
        initial_m=initial_m,
        final_m=final_m,
        initial_spin=initial_spin,
        final_spin=final_spin,
        delta_n=delta_n,
        delta_l=delta_l,
        delta_m=delta_m,
        delta_spin=delta_spin,
        zero_field_delta_energy_ev=zero_field_delta,
        zero_field_photon_energy_ev=zero_field_photon_energy,
        zero_field_frequency_hz=zero_field_frequency,
        zero_field_wavelength_nm=zero_field_wavelength_nm,
        magnetic_field_t=magnetic_field_t,
        initial_zeeman_shift_ev=initial_zeeman,
        final_zeeman_shift_ev=final_zeeman,
        transition_zeeman_shift_ev=transition_zeeman,
        zeeman_frequency_shift_hz=frequency - zero_field_frequency,
        delta_energy_ev=delta,
        photon_energy_ev=photon_energy,
        frequency_hz=frequency,
        wavelength_nm=wavelength_nm,
        process="absorção" if delta > 0 else "emissão",
        dipole_l_allowed=dipole_l_allowed,
        dipole_m_allowed=dipole_m_allowed,
        spin_allowed=spin_allowed,
        parity_changes=parity_changes,
        electric_dipole_allowed=(
            dipole_l_allowed and dipole_m_allowed and spin_allowed
        ),
        spectral_region=spectral_region(wavelength_nm),
    )
