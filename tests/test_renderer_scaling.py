import math

import pytest

from orbitals.orbital import Orbital
from simulator.renderer import Renderer


def test_orbital_range_scales_inversely_with_effective_charge():
    renderer = Renderer()
    compact = Orbital(3, 2, 1, Z_eff=1.0)
    diffuse = Orbital(3, 2, 1, Z_eff=0.5)

    assert renderer._get_range_for_orbital(diffuse) == pytest.approx(
        2 * renderer._get_range_for_orbital(compact)
    )


@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)
def test_renderer_builds_time_dependent_superposition_density():
    renderer = Renderer()
    state_a = Orbital(1, 0, 0, Z_eff=1.0)
    state_b = Orbital(2, 1, 0, Z_eff=1.0)
    prepared = renderer.prepare_superposition(state_a, state_b, grid_size=32)

    phase_zero = renderer.render_superposition(
        prepared, weight_b=0.5, relative_phase_rad=0.0,
    )
    phase_pi = renderer.render_superposition(
        prepared, weight_b=0.5, relative_phase_rad=math.pi,
    )

    assert phase_zero.n_points > 0
    assert phase_pi.n_points > 0
    assert phase_zero.center[2] == pytest.approx(-phase_pi.center[2], rel=0.1)
