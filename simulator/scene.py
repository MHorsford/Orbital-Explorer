"""
simulator/scene.py

Classe Scene — gerencia a cena 3D com PyVista.
Responsável pela câmera, iluminação, e adição/remoção de objetos 3D.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pyvista as pv
from config import (
    WINDOW_WIDTH, WINDOW_HEIGHT, BG_COLOR,
    SHOW_AXES, SHOW_NUCLEUS, CAMERA_INITIAL_POSITION, CAMERA_FOCAL_POINT,
    COLOR_NUCLEUS, ORBITAL_NEGATIVE_PHASE_COLOR,
)


class Scene:
    def __init__(self, title: str = "Atomic Orbital Simulator"):
        self.title = title

        self.plotter = self._create_default_plotter()
        self.actors = {}
        self.orbital_meshes = {}
        self._configure_plotter()

    def _create_default_plotter(self):
        return pv.Plotter(
            window_size=(WINDOW_WIDTH, WINDOW_HEIGHT),
            title=self.title,
            off_screen=False
        )

    def _configure_plotter(self):
        self.plotter.background_color = BG_COLOR

        self._setup_camera()
        # Iluminação padrão de três pontos do PyVista.
        self.plotter.enable_3_lights()

        if SHOW_AXES:
            self._add_axes()
        if SHOW_NUCLEUS:
            self._add_nucleus_sphere()

    def use_plotter(self, plotter):
        """Troca a cena para um plotter embutido na interface Qt."""
        old_plotter = self.plotter
        self.plotter = plotter
        self.actors = {}
        self.orbital_meshes = {}
        self._configure_plotter()

        if old_plotter is not plotter:
            try:
                old_plotter.close()
            except Exception:
                pass

    def _setup_camera(self):
        self.plotter.camera.position = CAMERA_INITIAL_POSITION
        self.plotter.camera.focal_point = CAMERA_FOCAL_POINT
        self.plotter.camera.up = (0, 0, 1)
        self.plotter.camera.zoom(0.8)

    def _add_axes(self):
        self.plotter.add_axes(
            xlabel='X', ylabel='Y', zlabel='Z',
            x_color='red', y_color='green', z_color='blue'
        )

    def _add_nucleus_sphere(self):
        nucleus = pv.Sphere(radius=0.2, center=[0,0,0], theta_resolution=20, phi_resolution=20)
        self.plotter.add_mesh(
            nucleus,
            color=COLOR_NUCLEUS,
            opacity=0.9,
            show_edges=False,
            label='Núcleo'
        )
        self.actors['nucleus'] = nucleus

    # CÂMERA

    def set_camera_for_range(self, range_max):
        """Ajusta a câmera para que o orbital caiba na tela."""
        distance = range_max * 2.5   # fator empírico
        self.plotter.camera.position = (distance, distance, distance)
        self.plotter.camera.focal_point = (0, 0, 0)
        self.plotter.camera.up = (0, 0, 1)
        self.plotter.render()

    # MALHAS DE ORBITAIS

    def update_orbital_mesh(self, orbital_id: str, mesh, color=None, opacity=None):
        """Atualiza uma malha; o PyVista substitui atores com o mesmo nome."""
        if orbital_id not in self.orbital_meshes:
            self.add_orbital_mesh(mesh, orbital_id, color, opacity)
            return
            
        old_data = self.orbital_meshes[orbital_id]
        color = color or old_data['color']
        opacity = opacity if opacity is not None else old_data['opacity']
        
        self.add_orbital_mesh(mesh, orbital_id, color, opacity)

    def add_orbital_mesh(
            self, mesh_data, orbital_id: str, color, opacity: float = 0.8,
            negative_color=None, **kwargs):
        extra_kwargs = {
            'smooth_shading': True,
            'show_edges': False,
            # Os orbitais usam cores explícitas; uma barra de escala seria redundante.
            'show_scalar_bar': False,
        }
        extra_kwargs.update(kwargs)

        # Nuvens de pontos usam vértices sem faces e exigem parâmetros de
        # exibição diferentes das superfícies poligonais.
        def _is_point_cloud(m):
            # PolyData de pontos possui células de vértice, mas nenhuma face.
            if m is None or not hasattr(m, 'n_points') or m.n_points == 0:
                return False
            n_verts = getattr(m, 'n_verts', 0)
            n_faces = getattr(m, 'n_faces', None)
            if n_faces is None:
                n_faces = getattr(m, 'n_faces_strict', 0)
            return n_verts > 0 and n_faces == 0

        if isinstance(mesh_data, tuple) and len(mesh_data) == 2:
            mesh_pos, mesh_neg = mesh_data
            phase_negative_color = negative_color or ORBITAL_NEGATIVE_PHASE_COLOR
            # Nomes estáveis permitem substituir cada fase sem duplicar atores.
            if mesh_pos is not None and mesh_pos.n_points > 0:
                self.plotter.add_mesh(mesh_pos, color=color, opacity=opacity, name=f"{orbital_id}_pos", **extra_kwargs)
            if mesh_neg is not None and mesh_neg.n_points > 0:
                self.plotter.add_mesh(
                    mesh_neg, color=phase_negative_color,
                    opacity=opacity, name=f"{orbital_id}_neg", **extra_kwargs
                )
                
            self.orbital_meshes[orbital_id] = {'mesh': mesh_data, 'actor': orbital_id, 'color': color, 'opacity': opacity}
        elif _is_point_cloud(mesh_data):
            point_kwargs = dict(extra_kwargs)
            # smooth_shading não se aplica a pontos sem faces; remove para
            # evitar warnings do PyVista e usa parâmetros próprios de pontos.
            point_kwargs.pop('smooth_shading', None)
            point_kwargs.update({
                'render_points_as_spheres': True,
                'point_size': 6.0,
                'opacity': opacity,
            })
            # A opacidade uniforme mantém compatibilidade entre versões do
            # PyVista. O array de densidade permanece disponível na malha.
            if getattr(mesh_data, 'active_scalars_name', None) is not None:
                mesh_data.active_scalars_name = None
            self.plotter.add_mesh(mesh_data, color=color, name=orbital_id, **point_kwargs)
            self.orbital_meshes[orbital_id] = {'mesh': mesh_data, 'actor': orbital_id, 'color': color, 'opacity': opacity}
        elif mesh_data is not None:
            self.plotter.add_mesh(mesh_data, color=color, opacity=opacity, name=orbital_id, **extra_kwargs)
            self.orbital_meshes[orbital_id] = {'mesh': mesh_data, 'actor': orbital_id, 'color': color, 'opacity': opacity}

    def remove_orbital_mesh(self, orbital_id: str):
        """Remove atores sem forçar renders intermediários piscando na tela."""
        for suffix in ["_pos", "_neg", ""]:
            name = f"{orbital_id}{suffix}" if suffix else orbital_id
            if name in self.plotter.actors:
                # render=False impede a tela de piscar no meio da troca
                self.plotter.remove_actor(name, render=False)
                
        if orbital_id in self.orbital_meshes:
            del self.orbital_meshes[orbital_id]


    def clear_orbital_meshes(self):
        for orbital_id in list(self.orbital_meshes.keys()):
            self.remove_orbital_mesh(orbital_id)

    def set_camera_position(self, position, focal_point=None):
        self.plotter.camera.position = position
        if focal_point is not None:
            self.plotter.camera.focal_point = focal_point
    
    def reset_camera(self):
        """Enquadra o orbital ativo ou restaura a câmera inicial."""
        if self.orbital_meshes:
            # O primeiro orbital ativo fornece a caixa de enquadramento.
            first_mesh_id = list(self.orbital_meshes.keys())[0]
            mesh_data = self.orbital_meshes[first_mesh_id]['mesh']
            
            # Isosuperfícies podem conter malhas separadas para as duas fases.
            target_mesh = mesh_data[0] if isinstance(mesh_data, tuple) else mesh_data
            
            if target_mesh is not None and hasattr(target_mesh, 'bounds'):
                # Considera os seis limites para cobrir qualquer orientação.
                bounds = target_mesh.bounds
                range_est = max(abs(b) for b in bounds)
                self.set_camera_for_range(range_est)
                return

        # Sem malhas ativas, usa a posição padrão.
        self.set_camera_position(CAMERA_INITIAL_POSITION, CAMERA_FOCAL_POINT)
        self.plotter.camera.up = (0, 0, 1)
        self.plotter.render()

    def get_camera_position(self):
        return self.plotter.camera.position

    def set_background_color(self, color):
        self.plotter.background_color = color

    def show(self):
        self.plotter.show()

    def close(self):
        self.plotter.close()

    def update(self):
        self.plotter.render()

    def screenshot(self, filename: str = "screenshot.png"):
        self.plotter.screenshot(filename)
        print(f"✓ Screenshot salva em {filename}")
