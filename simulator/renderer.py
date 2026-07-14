"""
simulator/renderer.py

Converte orbitais em representações 3D ou cortes 2D. Suporta
isosuperfícies, volumes de densidade, pontos de grade e nuvens Monte Carlo,
com resolução e alcance adaptados aos números quânticos do orbital.
"""

import sys
import os
from dataclasses import dataclass
from collections import OrderedDict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pyvista as pv
from config import ISO_VALUE, GRID_SIZE, GRID_RANGE, NUM_POINTS_CLOUD
from utils.grid import make_grid, normalize_array, cartesian_to_spherical
from utils.sampling import sample_orbital_grid, add_noise_to_points
from utils.helpers import quantum_label
from orbitals.wavefunction import HydrogenWavefunction
from physics.superposition import (
    state_coefficients, superposition_wavefunction,
)
from utils.slice2d import plot_orbital_slice, plot_orbital_slice_panel


@dataclass
class PreparedSuperposition:
    """Partes espaciais de dois estados avaliadas uma única vez no mesmo grid."""

    state_a: tuple
    state_b: tuple
    wave_a: np.ndarray
    wave_b: np.ndarray
    grid: pv.ImageData
    range_max: float


class Renderer:
    """
    Responsável por converter dados de orbital em geometria 3D (PyVista).
    """
    def __init__(self, config=None):
        self.mode = 'isosurface'
        # None indica que o isovalor deve ser calculado a partir de n e l.
        # Um valor numérico representa uma escolha manual do usuário.
        self.iso_value = None
        self.grid_size = GRID_SIZE
        self.grid_range = GRID_RANGE
        self.point_cloud_size = NUM_POINTS_CLOUD
        self.volume_brightness = 1.0
        self._density_volume_cache = OrderedDict()
        self._density_volume_cache_limit = 6
        self.config = config
        self.relative_iso_factor = None

    def set_mode(self, mode: str):
        if mode not in ['isosurface', 'density_volume', 'grid_points', 'points']:
            print(f"⚠ Modo desconhecido: {mode}. Mantendo: {self.mode}")
            return
        self.mode = mode
        print(f"✓ Modo de renderização: {mode}")

    # PARÂMETROS ADAPTATIVOS

    def _grid_size_for_orbital(self, n, l):
        """
        Calcula a resolução do grid a partir de n, l e do número de nós
        radiais. O teto limita o consumo cúbico de memória e processamento
        para manter a interface interativa em orbitais de n elevado.
        """
        radial_nodes = max(0, n - l - 1)

        base = {
            1: 120,
            2: 140,
            3: 180,
            4: 240,
            5: 280,
        }.get(n, 320)

        # Orbitais com muitos nós precisam de ainda mais resolução
        base += radial_nodes * 30

        MAX_GRID_SIZE = 180  # teto de segurança para manter a UI interativa
        return min(base, MAX_GRID_SIZE)

    def _get_range_for_orbital(self, orbital):
        """
        Estima o alcance espacial a partir do tamanho característico do orbital.
        """
        n = orbital.n
        Zeff = max(float(orbital.Z_eff), 0.1)
        a0 = 0.529177  # Å (Bohr radius)

        # O raio hidrogenoide característico cresce com n² e diminui com Z_eff.
        range_max = (n ** 2) * a0 * 5.5 / Zeff
        return max(range_max, 2.0)  # Garante mínimo de 2 Å

    def _adaptive_iso_value(self, n, l):
        """
        Reduz o isovalor conforme aumenta o número de nós radiais, tornando
        visíveis regiões internas de menor amplitude.
        """
        radial_nodes = n - l - 1
        # Quanto mais nós, menor o valor para ver os lóbulos internos
        base = 0.06 if radial_nodes == 0 else 0.04
        return max(0.01, base / (radial_nodes + 0.5))

    # SUPERPOSIÇÃO TEMPORAL

    @staticmethod
    def _image_grid(values, X, Y, Z, scalar_name):
        """Cria um grid volumétrico preservando a orientação dos eixos."""
        grid = pv.ImageData()
        grid.dimensions = values.shape
        grid.origin = (float(X.min()), float(Y.min()), float(Z.min()))
        grid.spacing = (
            float((X.max() - X.min()) / (values.shape[0] - 1)),
            float((Y.max() - Y.min()) / (values.shape[1] - 1)),
            float((Z.max() - Z.min()) / (values.shape[2] - 1)),
        )
        grid[scalar_name] = values.flatten(order='F')
        grid.set_active_scalars(scalar_name)
        return grid

    @staticmethod
    def _visible_support_slices(probability, threshold=0.001, padding=2):
        """Recorta o vazio externo que não participa da imagem volumétrica."""
        support = np.asarray(probability) >= float(threshold)
        if not np.any(support):
            return tuple(slice(0, size) for size in probability.shape)

        indices = np.where(support)
        return tuple(
            slice(
                max(0, int(axis_indices.min()) - padding),
                min(probability.shape[axis], int(axis_indices.max()) + padding + 1),
            )
            for axis, axis_indices in enumerate(indices)
        )

    def prepare_superposition(self, orbital_a, orbital_b, grid_size: int = 64):
        """Avalia dois orbitais no mesmo grid para reutilização na animação."""
        grid_size = max(32, min(96, int(grid_size)))
        range_max = max(
            self._get_range_for_orbital(orbital_a),
            self._get_range_for_orbital(orbital_b),
        )
        wavefunction = HydrogenWavefunction(use_angstrom=True)
        X, Y, Z, R, theta, phi = wavefunction.generate_grid(
            size=grid_size, range_max=range_max,
        )
        wave_a = wavefunction.psi(
            R, theta, phi, orbital_a.n, orbital_a.l, orbital_a.m,
            orbital_a.Z_eff,
        )
        wave_b = wavefunction.psi(
            R, theta, phi, orbital_b.n, orbital_b.l, orbital_b.m,
            orbital_b.Z_eff,
        )

        # O grid teórico possui uma grande região praticamente nula. Manter
        # apenas o suporte conjunto melhora o enquadramento e reduz o volume
        # que a GPU percorre em todos os quadros da animação.
        envelope = np.abs(wave_a) ** 2 + np.abs(wave_b) ** 2
        envelope_max = float(np.max(envelope))
        if np.isfinite(envelope_max) and envelope_max > 1e-18:
            support_slices = self._visible_support_slices(
                envelope / envelope_max,
            )
            wave_a = wave_a[support_slices]
            wave_b = wave_b[support_slices]
            X = X[support_slices]
            Y = Y[support_slices]
            Z = Z[support_slices]

        grid = self._image_grid(
            np.zeros_like(np.real(wave_a), dtype=float),
            X, Y, Z, 'probability',
        )
        return PreparedSuperposition(
            state_a=(orbital_a.n, orbital_a.l, orbital_a.m),
            state_b=(orbital_b.n, orbital_b.l, orbital_b.m),
            wave_a=wave_a,
            wave_b=wave_b,
            grid=grid,
            range_max=range_max,
        )

    def render_superposition(
            self, prepared: PreparedSuperposition, weight_b: float,
            relative_phase_rad: float, iso_value: float = None,
            as_volume: bool = False, phase_coloring: bool = False,
    ):
        """Gera uma isosuperfície ou volume de |Ψ(t)|² para um quadro."""
        wave = superposition_wavefunction(
            prepared.wave_a, prepared.wave_b,
            weight_b=weight_b,
            relative_phase_rad=relative_phase_rad,
        )
        density = np.abs(wave) ** 2
        coefficient_a, coefficient_b = state_coefficients(weight_b)
        density_upper_bound = (
            coefficient_a * np.abs(prepared.wave_a)
            + coefficient_b * np.abs(prepared.wave_b)
        ) ** 2
        density_scale = float(np.max(density_upper_bound))
        if not np.isfinite(density_scale) or density_scale < 1e-18:
            return self._empty_mesh()

        density_norm = density / density_scale
        if not np.all(np.isfinite(density_norm)):
            return self._empty_mesh()
        prepared.grid['probability'] = density_norm.flatten(order='F')
        prepared.grid.set_active_scalars('probability')

        if as_volume:
            if phase_coloring:
                # A parte espacial de A é real e seu coeficiente é positivo;
                # assim, A define a referência de fase da combinação.
                prepared.grid['phase_angle'] = np.angle(wave).flatten(order='F')
            elif 'phase_angle' in prepared.grid.point_data:
                prepared.grid.point_data.remove('phase_angle')
            return prepared.grid

        threshold = iso_value
        if threshold is None:
            threshold = self.iso_value if self.iso_value is not None else 0.06
        threshold = max(0.005, min(0.45, float(threshold)))

        try:
            contour = prepared.grid.contour(
                isosurfaces=[threshold], scalars='probability',
                progress_bar=False,
            )
            if contour.n_points == 0:
                return self._empty_mesh()
            mesh = contour.smooth(
                n_iter=8, relaxation_factor=0.06, progress_bar=False,
            )
            mesh.compute_normals(inplace=True)
            if 'probability' in mesh.point_data:
                mesh.point_data.remove('probability')
            mesh.active_scalars_name = None
            return mesh
        except Exception as error:
            print(f"⚠ Erro controlado ao gerar superposição: {error}")
            return self._empty_mesh()

    # MODO: NUVEM DE DENSIDADE VOLUMÉTRICA

    def render_density_volume(self, orbital):
        """Cria um volume em que brilho representa |ψ|² e cor representa fase."""
        range_max = self._get_range_for_orbital(orbital)
        # Volume ray casting custa mais que uma superfície. Essa resolução
        # ainda preserva os nós até n=7 e reduz bastante o trabalho por quadro.
        grid_size = min(84, max(60, 56 + 4 * orbital.n))
        cache_key = (
            orbital.n, orbital.l, orbital.m,
            round(float(orbital.Z_eff), 6), grid_size, round(range_max, 6),
        )
        cached = self._density_volume_cache.pop(cache_key, None)
        if cached is not None:
            self._density_volume_cache[cache_key] = cached
            return cached

        wavefunction = HydrogenWavefunction(use_angstrom=True)
        X, Y, Z, R, theta, phi = wavefunction.generate_grid(
            size=grid_size, range_max=range_max,
        )
        wave = wavefunction.psi(
            R, theta, phi, orbital.n, orbital.l, orbital.m, orbital.Z_eff,
        )
        probability = np.abs(wave) ** 2
        probability_max = float(np.max(probability))
        if not np.isfinite(probability_max) or probability_max < 1e-18:
            return pv.ImageData()

        probability_norm = probability / probability_max
        # Orbitais reais resultam em ±|ψ|². Para um valor complexo, o cosseno
        # da fase fornece uma transição contínua de cor sem alterar a opacidade.
        phase_projection = np.cos(np.angle(wave))
        signed_density = phase_projection * probability_norm
        if not np.all(np.isfinite(signed_density)):
            return pv.ImageData()

        # A transparência já descarta densidades abaixo de 0,4%. Recortar em
        # 0,1%, com margem, não remove informação visível e evita enquadrar a
        # enorme caixa vazia do grid hidrogenoide.
        support_slices = self._visible_support_slices(probability_norm)
        probability_norm = probability_norm[support_slices]
        signed_density = signed_density[support_slices]
        X = X[support_slices]
        Y = Y[support_slices]
        Z = Z[support_slices]
        grid = self._image_grid(
            signed_density.astype(float), X, Y, Z, 'signed_density',
        )
        grid['probability'] = probability_norm.astype(float).flatten(order='F')
        grid.set_active_scalars('signed_density')
        self._density_volume_cache[cache_key] = grid
        while len(self._density_volume_cache) > self._density_volume_cache_limit:
            self._density_volume_cache.popitem(last=False)
        return grid

    # MODO: ISOSURFACE

    def render_isosurface(self, orbital):
        n = orbital.n
        l = orbital.l
        Zeff = orbital.Z_eff

        # Resolução e alcance dependem do orbital.
        current_grid_size = self._grid_size_for_orbital(n, l)
        range_max = self._get_range_for_orbital(orbital)

        wave, X, Y, Z = orbital.get_density(current_grid_size, range_max)

        wave_max = np.max(np.abs(wave))
        if not np.isfinite(wave_max) or wave_max < 1e-15:
            print(f"⚠ Orbital com amplitude inválida")
            return (None, None)

        grid = pv.ImageData()
        grid.dimensions = wave.shape

        x_min, x_max = X.min(), X.max()
        y_min, y_max = Y.min(), Y.max()
        z_min, z_max = Z.min(), Z.max()

        grid.origin = (x_min, y_min, z_min)
        grid.spacing = (
            (x_max - x_min) / (wave.shape[0] - 1) if wave.shape[0] > 1 else 1.0,
            (y_max - y_min) / (wave.shape[1] - 1) if wave.shape[1] > 1 else 1.0,
            (z_max - z_min) / (wave.shape[2] - 1) if wave.shape[2] > 1 else 1.0,
        )

        # Normaliza a amplitude preservando o sinal da função de onda.
        if wave_max > 1e-12:
            gamma = 0.85
            wave_norm = wave / wave_max # np.sign(wave) * (np.abs(wave) / wave_max) ** gamma
        else:
            wave_norm = wave

        # PyVista exige order='F' para mapear corretamente os eixos
        grid['wave'] = wave_norm.flatten(order='F')

        # Uma escolha manual tem prioridade sobre o valor adaptativo.
        adaptive_iso = self._adaptive_iso_value(n, l)
        iso_val = self.iso_value if self.iso_value is not None else adaptive_iso

        mesh_pos, mesh_neg = None, None

        try:
            contour_pos = grid.contour(isosurfaces=[iso_val], scalars='wave', progress_bar=False)
            if contour_pos.n_points > 0:
                # Suavização moderada preserva os nós e reduz facetas do grid.
                mesh_pos = contour_pos.smooth(n_iter=20, relaxation_factor=0.08, progress_bar=False)
                mesh_pos.compute_normals(inplace=True)
                # Remove scalar 'wave' para evitar barra de cores indesejada
                if 'wave' in mesh_pos.point_data:
                    mesh_pos.point_data.remove('wave')
                mesh_pos.active_scalars_name = None

            # Evita um segundo contour quando a função não possui região
            # negativa além do isovalor, como ocorre no orbital 1s.
            if wave_norm.min() < -iso_val:
                contour_neg = grid.contour(isosurfaces=[-iso_val], scalars='wave', progress_bar=False)
                if contour_neg.n_points > 0:
                    mesh_neg = contour_neg.smooth(n_iter=20, relaxation_factor=0.08, progress_bar=False)
                    mesh_neg.compute_normals(inplace=True)
                    if 'wave' in mesh_neg.point_data:
                        mesh_neg.point_data.remove('wave')
                    mesh_neg.active_scalars_name = None

            return (mesh_pos, mesh_neg)

        except Exception as e:
            print(f"⚠ Erro controlado ao gerar isosurface: {e}")
            return (None, None)

    # MODO: GRID POINTS

    def render_grid_points(self, orbital):
        n = orbital.n
        range_max = self._get_range_for_orbital(orbital)   # usa range automático

        density, X, Y, Z = orbital.get_density(self.grid_size // 2, range_max)

        dens_flat = density.flatten()
        max_dens = np.max(dens_flat)
        threshold = max_dens * 0.005 if max_dens > 1e-15 else 1e-15
        mask = dens_flat > threshold

        points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
        if not np.any(mask):
            return self._empty_mesh()

        mesh = pv.PolyData(points[mask])
        mesh['density'] = dens_flat[mask]
        return mesh

    # MODO: POINT CLOUD (Monte Carlo)

    def render_points(self, orbital):
        # Usa range automático
        range_max = self._get_range_for_orbital(orbital)
        # A mesma função de onda atende toda a avaliação vetorizada.
        wf = HydrogenWavefunction(use_angstrom=True)

        def psi_squared_func(x, y, z):
            r, theta, phi = cartesian_to_spherical(x, y, z)
            psi = wf.psi(r, theta, phi, orbital.n, orbital.l, orbital.m, orbital.Z_eff)
            return np.abs(psi) ** 2

        try:
            points = sample_orbital_grid(
                psi_squared_func,
                n_samples=self.point_cloud_size,
                size=30,
                range_max=range_max
            )
            if len(points) == 0:
                return self._empty_mesh()

            # Um jitter proporcional ao espaçamento reduz o padrão regular da
            # amostragem sem alterar a escala característica do orbital.
            spacing = (2 * range_max) / 29  # 30 pontos por eixo → 29 intervalos
            points = add_noise_to_points(points, std_dev=spacing * 0.4)

            mesh = pv.PolyData(points)
            # Calcula a densidade de todos os pontos de forma vetorizada.
            densities = psi_squared_func(points[:, 0], points[:, 1], points[:, 2])
            mesh['density'] = densities
            return mesh

        except Exception as e:
            print(f"⚠ Erro ao gerar point cloud: {e}")
            return self._empty_mesh()

    # MODO: CORTE 2D (ψ e |ψ|² num plano — não usa PyVista)

    def render_slice_2d(self, orbital, plane: str = 'xz', resolution: int = 400):
        """
        Gera uma figura matplotlib com o corte 2D do orbital (ψ com sinal
        + |ψ|² lado a lado) no plano escolhido ('xz', 'xy' ou 'yz').

        Usa o mesmo alcance adaptativo da representação 3D para manter a
        escala espacial consistente entre as visualizações.

        O cálculo não depende do PyVista nem modifica a cena 3D.
        """

        range_max = self._get_range_for_orbital(orbital)
        return plot_orbital_slice(orbital, plane=plane, range_max=range_max,
                                   resolution=resolution)


    def render_slice_2d_panels(self, orbital, plane: str = 'xz', resolution: int = 400):
        """Gera figuras separadas para amplitude ψ e probabilidade |ψ|²."""
        range_max = self._get_range_for_orbital(orbital)
        return {
            'amplitude': plot_orbital_slice_panel(
                orbital, plane=plane, range_max=range_max,
                resolution=resolution, panel='amplitude'
            ),
            'probability': plot_orbital_slice_panel(
                orbital, plane=plane, range_max=range_max,
                resolution=resolution, panel='probability'
            ),
        }

    # INTERFACE PÚBLICA

    def render_orbital(self, orbital):
        if self.mode == 'isosurface':
            return self.render_isosurface(orbital)
        elif self.mode == 'density_volume':
            return self.render_density_volume(orbital)
        elif self.mode == 'grid_points':
            return self.render_grid_points(orbital)
        elif self.mode == 'points':
            return self.render_points(orbital)
        else:
            print(f"⚠ Modo desconhecido: {self.mode}")
            return self._empty_mesh()

    def render_nucleus(self, nucleus, radius: float = 0.2):
        sphere = pv.Sphere(
            radius=radius,
            center=[0, 0, 0],
            theta_resolution=30,
            phi_resolution=30
        )
        return sphere

    # UTILITÁRIOS

    def _empty_mesh(self):
        return pv.PolyData()

    def set_iso_value(self, iso_value: float):
        self.iso_value = max(0.001, min(2.0, iso_value))

    def set_grid_resolution(self, size: int):
        self.grid_size = max(20, min(150, size))

    def set_point_cloud_size(self, n_points: int):
        self.point_cloud_size = max(1000, min(50000, n_points))

    def set_volume_brightness(self, brightness: float):
        self.volume_brightness = max(0.2, min(2.0, float(brightness)))
