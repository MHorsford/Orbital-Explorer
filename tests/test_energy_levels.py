import pytest

from physics.energy_levels import (
    approximate_orbital_energy,
    build_energy_levels,
    calculate_transition,
    diagram_subshells,
    zeeman_shift_ev,
)
from physics.constants import MU_B_EV_T
from physics.screening import build_ground_state_config


def test_hydrogen_1s_uses_rydberg_energy():
    energy, z_eff = approximate_orbital_energy(
        1, 1, 0, electron_count=1, configuration={(1, 0): 1}
    )
    assert z_eff == pytest.approx(1.0)
    assert energy == pytest.approx(-13.605693122994)


def test_empty_hydrogen_levels_relocate_the_same_electron():
    energy, z_eff = approximate_orbital_energy(
        1, 3, 2, electron_count=1, configuration={(1, 0): 1}
    )

    assert z_eff == pytest.approx(1.0)
    assert energy == pytest.approx(-13.605693122994 / 9)


def test_energy_levels_include_occupancy_and_future_levels():
    config = build_ground_state_config(11, electron_count=10)
    levels = build_energy_levels(11, 10, config)
    by_key = {level.key: level for level in levels}
    assert by_key[(2, 1)].occupancy == 6
    assert (3, 0) in by_key
    assert all(level.energy_ev < 0 for level in levels)


def test_energy_diagram_includes_selected_distant_subshell():
    subshells = diagram_subshells(1, selected=(4, 2))

    assert (4, 2) in subshells
    assert subshells.index((4, 2)) > subshells.index((3, 1))


def test_energy_diagram_accepts_exploratory_subshell_outside_aufbau_sequence():
    subshells = diagram_subshells(1, selected=(5, 4))
    levels = build_energy_levels(1, 1, {(1, 0): 1}, subshells=subshells)

    assert levels[-1].key == (5, 4)


def test_energy_diagram_keeps_occupied_excited_subshell_visible():
    configuration = {(3, 0): 1}
    subshells = diagram_subshells(
        1, selected=(1, 0), configuration=configuration,
    )

    assert (1, 0) in subshells
    assert (3, 0) in subshells


def test_transition_reports_absorption_emission_and_selection_rule():
    levels = build_energy_levels(1, 1, {(1, 0): 1})
    by_key = {level.key: level for level in levels}

    absorption = calculate_transition(by_key[(1, 0)], by_key[(2, 1)])
    emission = calculate_transition(by_key[(2, 1)], by_key[(1, 0)])
    forbidden_l = calculate_transition(by_key[(1, 0)], by_key[(2, 0)])

    assert absorption.process == "absorção"
    assert emission.process == "emissão"
    assert absorption.photon_energy_ev == pytest.approx(emission.photon_energy_ev)
    assert absorption.frequency_hz > 0
    assert absorption.wavelength_nm == pytest.approx(121.5, abs=0.5)
    assert absorption.dipole_l_allowed
    assert absorption.dipole_m_allowed
    assert absorption.spin_allowed
    assert absorption.parity_changes
    assert absorption.electric_dipole_allowed
    assert not forbidden_l.dipole_l_allowed
    assert not forbidden_l.electric_dipole_allowed


def test_e1_rule_checks_magnetic_number_and_spin():
    levels = build_energy_levels(
        1, 1, {(1, 0): 1},
        subshells=diagram_subshells(1, selected=(3, 2)),
    )
    by_key = {level.key: level for level in levels}

    allowed = calculate_transition(
        by_key[(1, 0)], by_key[(2, 1)],
        initial_m=0, final_m=1,
        initial_spin=0.5, final_spin=0.5,
    )
    forbidden_m = calculate_transition(
        by_key[(2, 1)], by_key[(3, 2)],
        initial_m=-1, final_m=1,
    )
    forbidden_spin = calculate_transition(
        by_key[(1, 0)], by_key[(2, 1)],
        initial_spin=0.5, final_spin=-0.5,
    )

    assert allowed.delta_n == 1
    assert allowed.delta_l == 1
    assert allowed.delta_m == 1
    assert allowed.delta_spin == 0
    assert allowed.electric_dipole_allowed
    assert not forbidden_m.dipole_m_allowed
    assert not forbidden_m.electric_dipole_allowed
    assert not forbidden_spin.spin_allowed
    assert not forbidden_spin.electric_dipole_allowed


@pytest.mark.parametrize(
    "initial_key,final_key,initial_m,final_m,initial_spin,final_spin",
    [
        ((1, 0), (2, 1), 1, 0, 0.5, 0.5),
        ((1, 0), (2, 1), 0, 2, 0.5, 0.5),
        ((1, 0), (2, 1), 0, 0, 0.0, 0.5),
    ],
)
def test_transition_rejects_invalid_single_electron_quantum_numbers(
        initial_key, final_key, initial_m, final_m,
        initial_spin, final_spin,
):
    levels = build_energy_levels(1, 1, {(1, 0): 1})
    by_key = {level.key: level for level in levels}

    with pytest.raises(ValueError):
        calculate_transition(
            by_key[initial_key], by_key[final_key],
            initial_m, final_m, initial_spin, final_spin,
        )


def test_hydrogen_1s_to_4d_matches_selected_orbital_case():
    subshells = diagram_subshells(1, selected=(4, 2))
    levels = build_energy_levels(1, 1, {(1, 0): 1}, subshells=subshells)
    by_key = {level.key: level for level in levels}

    transition = calculate_transition(by_key[(1, 0)], by_key[(4, 2)])

    assert transition.wavelength_nm == pytest.approx(97.20, abs=0.05)
    assert not transition.dipole_l_allowed


def test_same_level_transition_is_rejected():
    level = build_energy_levels(1, 1, {(1, 0): 1})[0]
    with pytest.raises(ValueError):
        calculate_transition(level, level)


def test_linear_zeeman_shift_uses_orbital_and_spin_projections():
    field = 1.0

    spin_up_1s = zeeman_shift_ev(field, magnetic_number=0, spin=0.5)
    spin_down_1s = zeeman_shift_ev(field, magnetic_number=0, spin=-0.5)
    orbital_2p = zeeman_shift_ev(field, magnetic_number=1, spin=0.5)

    assert spin_up_1s == pytest.approx(MU_B_EV_T * 2.00231930436 / 2)
    assert spin_down_1s == pytest.approx(-spin_up_1s)
    assert orbital_2p - spin_up_1s == pytest.approx(MU_B_EV_T)


def test_zeeman_transition_splits_sigma_but_not_pi_component():
    levels = build_energy_levels(
        1, 1, {(1, 0): 1}, subshells=[(1, 0), (2, 1)],
    )
    by_key = {level.key: level for level in levels}
    pi_component = calculate_transition(
        by_key[(1, 0)], by_key[(2, 1)],
        initial_m=0, final_m=0, magnetic_field_t=1.0,
    )
    sigma_component = calculate_transition(
        by_key[(1, 0)], by_key[(2, 1)],
        initial_m=0, final_m=1, magnetic_field_t=1.0,
    )

    assert pi_component.transition_zeeman_shift_ev == pytest.approx(0.0)
    assert pi_component.wavelength_nm == pytest.approx(
        pi_component.zero_field_wavelength_nm
    )
    assert sigma_component.transition_zeeman_shift_ev == pytest.approx(MU_B_EV_T)
    assert sigma_component.zeeman_frequency_shift_hz / 1e9 == pytest.approx(
        13.9962, abs=0.001,
    )
    assert sigma_component.wavelength_nm < sigma_component.zero_field_wavelength_nm


def test_reversing_magnetic_field_reverses_zeeman_shift():
    levels = build_energy_levels(
        1, 1, {(1, 0): 1}, subshells=[(1, 0), (2, 1)],
    )
    by_key = {level.key: level for level in levels}

    positive = calculate_transition(
        by_key[(1, 0)], by_key[(2, 1)],
        final_m=1, magnetic_field_t=2.0,
    )
    negative = calculate_transition(
        by_key[(1, 0)], by_key[(2, 1)],
        final_m=1, magnetic_field_t=-2.0,
    )

    assert positive.transition_zeeman_shift_ev == pytest.approx(
        -negative.transition_zeeman_shift_ev
    )


def test_zeeman_rejects_non_finite_field():
    with pytest.raises(ValueError):
        zeeman_shift_ev(float("nan"), 0, 0.5)
