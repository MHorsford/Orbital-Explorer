import numpy as np
import pytest

from physics.constants import A0_ANGSTROM
from physics.radial_distribution import calculate_radial_distribution


def _integral(values, coordinates):
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    return float(trapezoid(values, coordinates))


def test_1s_probability_is_normalized_and_peaks_at_bohr_radius():
    distribution = calculate_radial_distribution(1, 0, z_eff=1.0)

    assert _integral(
        distribution.probability_density,
        distribution.radius_angstrom,
    ) == pytest.approx(1.0, abs=1e-6)
    assert distribution.most_probable_radius_angstrom == pytest.approx(
        A0_ANGSTROM, abs=0.01,
    )
    assert distribution.mean_radius_angstrom == pytest.approx(
        1.5 * A0_ANGSTROM, abs=0.01,
    )


def test_2s_has_one_radial_node_at_two_bohr_radii():
    distribution = calculate_radial_distribution(2, 0, z_eff=1.0)

    assert distribution.radial_node_count == 1
    assert distribution.radial_nodes_angstrom[0] == pytest.approx(
        2.0 * A0_ANGSTROM,
    )


@pytest.mark.parametrize(
    "n,l,radial_nodes,angular_nodes,total_nodes",
    [
        (2, 1, 0, 1, 1),
        (3, 0, 2, 0, 2),
        (3, 2, 0, 2, 2),
        (4, 2, 1, 2, 3),
    ],
)
def test_node_counts_follow_quantum_number_relations(
        n, l, radial_nodes, angular_nodes, total_nodes,
):
    distribution = calculate_radial_distribution(n, l)

    assert distribution.radial_node_count == radial_nodes
    assert distribution.angular_node_count == angular_nodes
    assert distribution.total_node_count == total_nodes


def test_characteristic_radii_contract_with_effective_charge():
    hydrogen = calculate_radial_distribution(1, 0, z_eff=1.0)
    contracted = calculate_radial_distribution(1, 0, z_eff=2.0)

    assert contracted.most_probable_radius_angstrom == pytest.approx(
        hydrogen.most_probable_radius_angstrom / 2.0,
    )
    assert contracted.mean_radius_angstrom == pytest.approx(
        hydrogen.mean_radius_angstrom / 2.0,
    )


@pytest.mark.parametrize(
    "n,l,z_eff,resolution",
    [
        (0, 0, 1.0, 2500),
        (2, 2, 1.0, 2500),
        (1, 0, 0.0, 2500),
        (1, 0, 1.0, 100),
    ],
)
def test_invalid_radial_parameters_are_rejected(n, l, z_eff, resolution):
    with pytest.raises(ValueError):
        calculate_radial_distribution(n, l, z_eff, resolution)
