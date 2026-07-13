import pytest

from physics.energy_levels import build_energy_levels, calculate_transition
from physics.spectral_reference import (
    compare_with_nist_hydrogen,
    nist_comparison_scope_message,
)


def _hydrogen_levels(*keys):
    return {
        level.key: level
        for level in build_energy_levels(
            1, 1, {(1, 0): 1}, subshells=keys,
        )
    }


def test_lyman_alpha_comparison_uses_nist_evaluated_levels():
    levels = _hydrogen_levels((1, 0), (2, 1))
    transition = calculate_transition(
        levels[(1, 0)], levels[(2, 1)],
        initial_m=0, final_m=1,
    )

    comparison = compare_with_nist_hydrogen(transition, 1, 1)

    assert comparison.series == "Lyman"
    assert comparison.reference_vacuum_nm == pytest.approx(121.567, abs=0.002)
    assert comparison.relative_error_percent < 0.1


def test_balmer_alpha_comparison_is_in_vacuum():
    levels = _hydrogen_levels((2, 0), (3, 1))
    transition = calculate_transition(levels[(2, 0)], levels[(3, 1)])

    comparison = compare_with_nist_hydrogen(transition, 1, 1)

    assert comparison.series == "Balmer"
    assert comparison.reference_vacuum_nm == pytest.approx(656.47, abs=0.02)


def test_nist_comparison_is_symmetric_for_emission_and_absorption():
    levels = _hydrogen_levels((1, 0), (3, 1))
    absorption = calculate_transition(levels[(1, 0)], levels[(3, 1)])
    emission = calculate_transition(levels[(3, 1)], levels[(1, 0)])

    forward = compare_with_nist_hydrogen(absorption, 1, 1)
    reverse = compare_with_nist_hydrogen(emission, 1, 1)

    assert forward.reference_vacuum_nm == pytest.approx(reverse.reference_vacuum_nm)
    assert forward.relative_error_percent == pytest.approx(
        reverse.relative_error_percent
    )


def test_comparison_rejects_incompatible_species_and_forbidden_e1():
    levels = _hydrogen_levels((1, 0), (2, 0), (2, 1))
    allowed = calculate_transition(levels[(1, 0)], levels[(2, 1)])
    forbidden = calculate_transition(levels[(1, 0)], levels[(2, 0)])

    assert compare_with_nist_hydrogen(allowed, 2, 1) is None
    assert compare_with_nist_hydrogen(forbidden, 1, 1) is None
    assert "hidrogênio neutro" in nist_comparison_scope_message(allowed, 2, 1)
    assert "E1 permitidas" in nist_comparison_scope_message(forbidden, 1, 1)


def test_comparison_explains_reference_limit_above_n_five():
    levels = _hydrogen_levels((1, 0), (6, 1))
    transition = calculate_transition(levels[(1, 0)], levels[(6, 1)])

    assert compare_with_nist_hydrogen(transition, 1, 1) is None
    assert "até n=5" in nist_comparison_scope_message(transition, 1, 1)
