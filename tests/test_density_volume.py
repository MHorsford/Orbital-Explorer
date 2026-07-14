import numpy as np
import pyvista as pv

from orbitals.orbital import Orbital
from simulator.renderer import Renderer
from simulator.scene import Scene


def test_density_volume_preserves_probability_and_phase():
    renderer = Renderer()
    volume = renderer.render_density_volume(
        Orbital(n=2, l=1, m=0, electrons=1, Z_eff=1.0)
    )

    assert isinstance(volume, pv.ImageData)
    assert volume.n_points > 0
    assert 'signed_density' in volume.point_data
    assert 'probability' in volume.point_data
    values = volume['signed_density']
    assert np.all(np.isfinite(values))
    assert values.min() < 0.0 < values.max()
    assert np.max(np.abs(values)) <= 1.0 + 1e-12
    assert max(volume.dimensions) <= 84
    assert volume.n_points < 60 ** 3


def test_density_volume_reuses_cached_grid_for_same_state():
    renderer = Renderer()
    first = renderer.render_density_volume(
        Orbital(n=3, l=1, m=0, electrons=1, Z_eff=1.0)
    )
    second = renderer.render_density_volume(
        Orbital(n=3, l=1, m=0, electrons=0, Z_eff=1.0)
    )

    assert second is first


def test_superposition_can_return_normalized_probability_volume():
    renderer = Renderer()
    prepared = renderer.prepare_superposition(
        Orbital(1, 0, 0, electrons=1, Z_eff=1.0),
        Orbital(2, 1, 0, electrons=1, Z_eff=1.0),
        grid_size=32,
    )

    volume = renderer.render_superposition(
        prepared,
        weight_b=0.5,
        relative_phase_rad=np.pi / 3,
        as_volume=True,
    )

    values = volume['probability']
    assert isinstance(volume, pv.ImageData)
    assert np.all(np.isfinite(values))
    assert values.min() >= 0.0
    assert values.max() <= 1.0 + 1e-12


def test_superposition_volume_can_expose_and_remove_relative_phase():
    renderer = Renderer()
    prepared = renderer.prepare_superposition(
        Orbital(1, 0, 0, electrons=1, Z_eff=1.0),
        Orbital(2, 1, 0, electrons=1, Z_eff=1.0),
        grid_size=32,
    )

    volume = renderer.render_superposition(
        prepared,
        weight_b=0.5,
        relative_phase_rad=np.pi / 2,
        as_volume=True,
        phase_coloring=True,
    )
    phase = np.asarray(volume['phase_angle'])
    visible = np.asarray(volume['probability']) > 0.01

    assert np.all(np.isfinite(phase))
    assert np.ptp(phase[visible]) > 0.5

    renderer.render_superposition(
        prepared,
        weight_b=0.5,
        relative_phase_rad=0.0,
        as_volume=True,
        phase_coloring=False,
    )
    assert 'phase_angle' not in volume.point_data


def test_complex_phase_color_wheel_distinguishes_quadrants():
    colors = Scene._complex_phase_colors(
        np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    )

    assert colors.shape == (4, 3)
    assert np.all((0.0 <= colors) & (colors <= 1.0))
    assert np.unique(np.round(colors, 4), axis=0).shape[0] == 4
    assert colors[0, 1] > colors[0, 0]  # fase 0: ciano
    assert colors[0, 2] > colors[0, 0]
    assert colors[2, 0] > colors[2, 1]  # fase π: vermelho
    assert colors[2, 0] > colors[2, 2]


def test_volume_brightness_is_bounded():
    renderer = Renderer()
    renderer.set_mode('density_volume')
    renderer.set_volume_brightness(8.0)
    assert renderer.mode == 'density_volume'
    assert renderer.volume_brightness == 2.0
    renderer.set_volume_brightness(0.01)
    assert renderer.volume_brightness == 0.2


def test_scene_uses_rgba_volume_with_transparent_background():
    class FakePlotter:
        def __init__(self):
            self.volume_kwargs = None
            self.volume_calls = 0
            self.actors = {}

        def add_volume(self, volume, **kwargs):
            self.volume_calls += 1
            self.volume_kwargs = kwargs
            render_grid = volume.copy()
            render_grid['Data'] = kwargs['scalars']
            render_grid.set_active_scalars('Data')

            class FakeMapper:
                def __init__(self, dataset):
                    self.dataset = dataset
                    self.updates = 0

                def update(self):
                    self.updates += 1

            actor = type('Actor', (), {})()
            actor.prop = type(
                'Property', (), {'interpolation_type': None},
            )()
            actor.mapper = FakeMapper(render_grid)
            self.actors[kwargs['name']] = actor
            return actor

    grid = pv.ImageData(dimensions=(3, 3, 3))
    grid['signed_density'] = np.linspace(-1.0, 1.0, grid.n_points)
    grid['probability'] = np.abs(grid['signed_density'])
    scene = Scene.__new__(Scene)
    scene.plotter = FakePlotter()
    scene.orbital_meshes = {}

    scene.add_orbital_mesh(
        grid, 'phase_volume', (0.2, 0.88, 1.0),
        negative_color=(1.0, 0.35, 0.78),
        volume_brightness=1.0,
    )

    kwargs = scene.plotter.volume_kwargs
    rgba = np.asarray(kwargs['scalars']).copy()
    assert rgba.shape == (grid.n_points, 4)
    assert rgba.dtype == np.uint8
    # O modo aditivo do VTK pode revelar a caixa preta do grid RGBA mesmo
    # quando o canal alfa externo vale zero.
    assert kwargs['blending'] == 'composite'
    assert kwargs['render'] is False
    assert rgba[grid.n_points // 2, 3] == 0
    assert rgba[0, 3] == rgba[-1, 3]
    assert rgba[:, 3].max() >= 200
    assert not np.array_equal(rgba[0, :3], rgba[-1, :3])

    first_alpha = rgba[:, 3].copy()
    scene.add_orbital_mesh(
        grid, 'phase_volume', (0.2, 0.88, 1.0),
        negative_color=(1.0, 0.35, 0.78),
        volume_brightness=2.0,
    )

    assert scene.plotter.volume_calls == 1
    render_grid = scene.orbital_meshes['phase_volume']['render_grid']
    assert np.max(np.asarray(render_grid['Data'])[:, 3]) > np.max(first_alpha)
    assert scene.orbital_meshes['phase_volume']['actor'].mapper.updates == 1


def test_real_volume_render_is_visible_without_black_grid_box():
    background = np.array([7, 17, 31], dtype=np.int16)
    renderer = Renderer()
    grid = renderer.render_density_volume(
        Orbital(n=3, l=1, m=0, electrons=1, Z_eff=1.0)
    )
    plotter = pv.Plotter(off_screen=True, window_size=(320, 320))
    plotter.set_background(tuple((background / 255.0).tolist()))
    scene = Scene.__new__(Scene)
    scene.plotter = plotter
    scene.orbital_meshes = {}

    try:
        scene.add_orbital_mesh(
            grid, '3p_z', (0.20, 0.88, 1.00),
            negative_color=(1.00, 0.35, 0.78),
            volume_brightness=1.0,
        )
        plotter.reset_camera()
        image = plotter.screenshot(return_img=True)[..., :3].astype(np.int16)
    finally:
        plotter.close()

    distance_from_background = np.max(
        np.abs(image - background), axis=2,
    )
    black_pixels = np.max(image, axis=2) < 3
    assert (distance_from_background > 10).mean() > 0.05
    assert image.max() > 150
    assert black_pixels.mean() < 0.001
