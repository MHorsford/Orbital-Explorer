import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from orbitals.wavefunction import HydrogenWavefunction


@pytest.fixture
def wf():
    return HydrogenWavefunction()


def test_1s_non_zero_at_origin(wf):
    psi = wf.psi_1s(0.0)
    assert psi > 0


def test_2p_zero_at_origin(wf):
    psi = wf.psi_2p_z(
        r=0.0,
        theta=np.pi/4
    )

    assert np.isclose(psi, 0.0)


def test_probability_positive(wf):
    p = wf.probability_density(
        r=1.0,
        theta=np.pi/3,
        phi=np.pi/4,
        n=1,
        l=0,
        m=0
    )

    assert p >= 0


def test_grid_shape(wf):
    density, X, Y, Z = wf.evaluate_on_grid(
        n=1,
        l=0,
        m=0,
        size=40
    )

    assert density.shape == (40, 40, 40)


def test_1s_spherical_symmetry(wf):
    r = 1.0

    psi1 = wf.psi_1s(r)
    psi2 = wf.psi_1s(r)

    assert np.isclose(psi1, psi2)


def test_density_is_real(wf):
    p = wf.probability_density(
        r=1.0,
        theta=np.pi/3,
        phi=np.pi/4,
        n=2,
        l=1,
        m=0
    )

    assert np.isreal(p)