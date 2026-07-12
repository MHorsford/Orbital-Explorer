import pytest

from atom.atom import Atom
from atom.manual_configuration import ManualElectronConfiguration
from orbitals.orbital import Orbital


def occupancy(atom, n, l):
    return [
        orbital.electrons
        for orbital in atom.orbitals
        if orbital.n == n and orbital.l == l
    ]


def test_orbital_stores_explicit_opposite_spins():
    orbital = Orbital(n=1, l=0)

    assert orbital.add_electron(spin=Orbital.SPIN_UP)
    assert orbital.add_electron(spin=Orbital.SPIN_DOWN)
    assert orbital.electron_spins == (0.5, -0.5)
    assert orbital.spin_symbols == "↑↓"


def test_pauli_rejects_equal_spin_and_third_electron():
    orbital = Orbital(n=2, l=1, m=0)

    assert orbital.add_electron(spin=Orbital.SPIN_UP)
    assert not orbital.add_electron(spin=Orbital.SPIN_UP)
    assert orbital.add_electron(spin=Orbital.SPIN_DOWN)
    assert not orbital.add_electron()
    assert orbital.electrons == 2


@pytest.mark.parametrize(
    "atomic_number, expected_2p",
    [(6, [1, 1, 0]), (7, [1, 1, 1]), (8, [2, 1, 1])],
)
def test_hund_distributes_electrons_before_pairing(atomic_number, expected_2p):
    assert occupancy(Atom(atomic_number), 2, 1) == expected_2p


def test_aufbau_places_potassium_electron_in_4s():
    potassium = Atom(19)

    assert potassium.get_electron_config() == "1s2 2s2 2p6 3s2 3p6 4s1"
    assert occupancy(potassium, 4, 0) == [1]


@pytest.mark.parametrize(
    "atomic_number, expected_config",
    [
        (24, "1s2 2s2 2p6 3s2 3p6 4s1 3d5"),
        (29, "1s2 2s2 2p6 3s2 3p6 4s1 3d10"),
        (46, "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 4d10"),
    ],
)
def test_known_ground_state_exceptions(atomic_number, expected_config):
    atom = Atom(atomic_number)

    assert atom.get_electron_config() == expected_config
    assert atom.has_configuration_exception
    assert atom.N_electrons == atomic_number


@pytest.mark.parametrize("atomic_number", [1, 6, 8, 19, 24, 29, 46, 79, 92])
def test_generated_atoms_pass_all_three_rules(atomic_number):
    checks = Atom(atomic_number).validate_filling_rules()

    assert checks == {"aufbau": True, "hund": True, "pauli": True}


def test_diagram_exposes_boxes_and_spin_quantum_numbers():
    diagram = Atom(6).get_orbital_diagram()

    assert "1s²" in diagram and "[↑↓]" in diagram
    assert "2p²" in diagram and "[↑ ] [↑ ] [  ]" in diagram
    assert "mₗ" in diagram


def test_manual_configuration_starts_empty_with_electron_budget():
    manual = ManualElectronConfiguration(6)

    assert manual.electron_count == 0
    assert manual.remaining_electrons == 6
    assert manual.configuration_string() == "vazia"
    assert (3, 0) in manual.visible_subshells()  # nível extra para excitações


def test_manual_pauli_violation_is_blocked():
    manual = ManualElectronConfiguration(2)

    assert manual.add_electron(1, 0, 0, Orbital.SPIN_UP).ok
    result = manual.add_electron(1, 0, 0, Orbital.SPIN_UP)

    assert not result.ok
    assert "Pauli" in result.message
    assert manual.electron_count == 1


def test_manual_aufbau_warns_when_higher_subshell_is_filled_first():
    manual = ManualElectronConfiguration(6)

    result = manual.add_electron(2, 1, -1, Orbital.SPIN_UP)

    assert result.ok
    assert not manual.validate_rules()["aufbau"]
    assert any("Aufbau" in warning for warning in result.warnings)


def test_manual_hund_warns_about_early_pairing():
    manual = ManualElectronConfiguration(8)
    manual.add_electron(1, 0, 0, Orbital.SPIN_UP)
    manual.add_electron(1, 0, 0, Orbital.SPIN_DOWN)
    manual.add_electron(2, 0, 0, Orbital.SPIN_UP)
    manual.add_electron(2, 0, 0, Orbital.SPIN_DOWN)
    manual.add_electron(2, 1, -1, Orbital.SPIN_UP)

    result = manual.add_electron(2, 1, -1, Orbital.SPIN_DOWN)

    assert result.ok
    assert not manual.validate_rules()["hund"]
    assert any("Hund" in warning for warning in result.warnings)


def test_manual_carbon_can_be_built_to_ground_state():
    manual = ManualElectronConfiguration(6)
    for n, l, m, spin in [
        (1, 0, 0, Orbital.SPIN_UP),
        (1, 0, 0, Orbital.SPIN_DOWN),
        (2, 0, 0, Orbital.SPIN_UP),
        (2, 0, 0, Orbital.SPIN_DOWN),
        (2, 1, -1, Orbital.SPIN_UP),
        (2, 1, 0, Orbital.SPIN_UP),
    ]:
        assert manual.add_electron(n, l, m, spin).ok

    checks = manual.validate_rules()
    assert manual.is_complete
    assert checks == {
        "aufbau": True, "hund": True, "pauli": True, "ground_state": True
    }
    assert manual.result_description() == "Configuração fundamental correta."


def test_manual_auto_fill_respects_chromium_exception_and_undo():
    manual = ManualElectronConfiguration(24)
    manual.fill_ground_state()

    assert manual.configuration_string().endswith("4s1 3d5")
    assert manual.validate_rules()["ground_state"]
    assert manual.undo()
    assert manual.electron_count == 0


def test_manual_table_scope_includes_selected_and_occupied_excited_subshells():
    manual = ManualElectronConfiguration(1)

    assert (3, 0) in manual.visible_subshells((3, 0))
    manual.add_electron(3, 0, 0, Orbital.SPIN_UP)
    assert (3, 0) in manual.visible_subshells()


def test_manual_explains_how_to_move_an_electron_when_budget_is_empty():
    manual = ManualElectronConfiguration(1)
    manual.add_electron(2, 0, 0, Orbital.SPIN_UP)

    result = manual.add_electron(3, 0, 0, Orbital.SPIN_UP)

    assert not result.ok
    assert "Nenhum elétron restante" in result.message
    assert "Remova um elétron de 2s" in result.message


def test_manual_labels_hydrogen_2s_as_permitted_excited_state():
    manual = ManualElectronConfiguration(1)

    result = manual.add_electron(2, 0, 0, Orbital.SPIN_UP)

    assert result.ok
    assert "Configuração permitida" in result.message
    assert any("estado excitado permitido" in warning for warning in result.warnings)


def test_manual_can_reveal_next_energy_levels_without_using_sliders():
    manual = ManualElectronConfiguration(1)

    visible = manual.visible_subshells(extra_levels=2)

    assert visible[:4] == [(1, 0), (2, 0), (2, 1), (3, 0)]


def test_manual_can_promote_hydrogen_from_1s_to_3s_in_one_action():
    manual = ManualElectronConfiguration(1)
    manual.add_electron(1, 0, 0, Orbital.SPIN_UP)

    result = manual.move_electron((1, 0, 0), (3, 0, 0))

    assert result.ok
    assert manual.electron_count == 1
    assert manual.get_orbital(1, 0, 0).electrons == 0
    assert manual.get_orbital(3, 0, 0).spin_symbols == "↑"
    assert any("estado excitado permitido" in warning for warning in result.warnings)
    assert manual.undo()
    assert manual.get_orbital(1, 0, 0).spin_symbols == "↑"


def test_manual_move_preserves_pauli_and_blocks_equal_spin_at_destination():
    manual = ManualElectronConfiguration(2)
    manual.add_electron(1, 0, 0, Orbital.SPIN_UP)
    manual.add_electron(2, 0, 0, Orbital.SPIN_UP)

    result = manual.move_electron((1, 0, 0), (2, 0, 0))

    assert not result.ok
    assert "Pauli" in result.message
    assert manual.electron_count == 2
