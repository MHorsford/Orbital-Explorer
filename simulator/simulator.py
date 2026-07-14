# simulator/simulator.py
"""
Classe Simulator — orquestrador geral.
Conecta o modelo atômico (Atom) com a renderização (Scene + Renderer).
"""

import sys
import os
import time
from pathlib import Path

# Path robusto
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from config import FPS_TARGET, ORBITAL_NEGATIVE_PHASE_COLOR
from simulator.scene import Scene
from simulator.renderer import Renderer
from utils.helpers import orbital_info_string, element_info_string


class Simulator:
    """
    Orquestrador principal do simulador.
    Gerencia o loop de simulação e atualização da cena.
    """
    
    def __init__(self, atom=None, title: str = "Atomic Orbital Simulator"):
        """
        Parâmetros:
            atom  : objeto Atom (criado aqui se None)
            title : título da janela
        """
        if atom is None:
            from atom.atom import Atom
            atom = Atom(Z=6)  # Padrão: Carbono
        
        self.atom = atom
        self.scene = Scene(title=title)
        self.renderer = Renderer()
        
        # Estado do simulador
        self.is_running = False
        self.frame_count = 0
        self.last_update_time = time.time()
        self.fps = 0
        
        # Orbitais visíveis 
        self.visible_orbitals = {}
        
        # Inicializar renderização
        self._initialize_rendering()
    
    def _initialize_rendering(self):
        """Renderiza o estado inicial do átomo."""
        self.update_atom_display()
    
    def update_atom_display(self):
        """
        Atualiza a renderização para o estado atual do átomo.
        Chamado quando o número atômico ou configuração muda.
        """
        # Limpar orbitais antigos da cena e do estado
        self.scene.clear_orbital_meshes()
        self.visible_orbitals.clear()
        
        # Renderizar cada orbital com elétrons
        for i, orbital in enumerate(self.atom.orbitals):
            if orbital.electrons > 0:
                # ID único para o orbital
                orbital_id = f"{orbital.n}{orbital.type_info.letter}_{orbital.m}_{i}"
                
                # Renderizar malha
                result = self.renderer.render_orbital(orbital)
                
                # Tratamento para diferentes modos de retorno
                if isinstance(result, tuple):  # isosurface retorna (pos, neg)
                    mesh_pos, mesh_neg = result
                    color = orbital.color
                    opacity = 0.75 if orbital.electrons == 2 else 0.55
                    
                    if mesh_pos:
                        self.scene.add_orbital_mesh(mesh_pos, f"{orbital_id}_pos", color, opacity)
                    if mesh_neg:
                        self.scene.add_orbital_mesh(
                            mesh_neg, f"{orbital_id}_neg",
                            ORBITAL_NEGATIVE_PHASE_COLOR, opacity * 0.8,
                        )
                        
                    self.visible_orbitals[orbital_id] = orbital
                else:  # outros modos retornam único mesh
                    mesh = result
                    if mesh and mesh.n_points > 0:
                        color = orbital.color
                        opacity = 0.75 if orbital.electrons == 2 else 0.55
                        self.scene.add_orbital_mesh(
                            mesh, orbital_id, color, opacity,
                            volume_brightness=self.renderer.volume_brightness,
                        )
                        self.visible_orbitals[orbital_id] = orbital

    def set_atom_z(self, Z: int):
        """Muda o número atômico do átomo."""
        if Z < 1 or Z == self.atom.Z:
            return
        
        from atom.atom import Atom
        self.atom = Atom(Z=Z)
        self.update_atom_display()
        print(f"✓ Átomo atualizado: {element_info_string(self.atom)}")
    
    def set_render_mode(self, mode: str):
        """Muda o modo de renderização global."""
        self.renderer.set_mode(mode)
        self.update_atom_display()
    
    def set_iso_value(self, iso_value: float):
        """Ajusta o valor de isosurface."""
        self.renderer.set_iso_value(iso_value)
        if self.renderer.mode == 'isosurface':
            # Atualização incremental
            for orbital_id, orbital in list(self.visible_orbitals.items()):
                new_mesh = self.renderer.render_orbital(orbital)
                self.scene.update_orbital_mesh(orbital_id, new_mesh)
            self.scene.update()
    
    def set_grid_resolution(self, size: int):
        """Ajusta a resolução do grid de cálculo."""
        self.renderer.set_grid_resolution(size)
        self.update_atom_display()

    def set_volume_brightness(self, brightness: float):
        """Ajusta o ganho visual da nuvem volumétrica."""
        self.renderer.set_volume_brightness(brightness)

    def toggle_orbital_visibility(self, orbital_id: str):
        """Alterna a visibilidade de um orbital específico."""
        if orbital_id in self.visible_orbitals:
            self.scene.remove_orbital_mesh(orbital_id)
            del self.visible_orbitals[orbital_id]
            print(f"✗ Orbital {orbital_id} ocultado")
        elif orbital_id == 'all':
            self.scene.clear_orbital_meshes()
            self.visible_orbitals.clear()
            print("✗ Todos os orbitais ocultados")
    
    def show_all_orbitals(self):
        """Mostra todos os orbitais preenchidos."""
        self.update_atom_display()
        print(f"✓ {len(self.visible_orbitals)} orbitais visíveis")
    
    def reset_camera(self):
        """Reseta a câmera para a posição ideal."""
        self.scene.reset_camera()
    
    def screenshot(self, filename: str = "screenshot.png"):
        """Captura uma screenshot do viewport atual."""
        self.scene.screenshot(filename)

    # ─── CORTE 2D (ψ e |ψ|² num plano) ───
    # O cálculo usa Matplotlib e não modifica a cena 3D.

    def get_orbital_slice_figure(self, orbital_id=None, plane: str = 'xz',
                                  resolution: int = 400):
        """
        Gera a figura Matplotlib do corte 2D sem exibir ou salvar. O retorno
        pode ser incorporado a um canvas da interface.

        Parâmetros:
            orbital_id : chave em self.visible_orbitals (ex: "2p_0_1").
                         Também aceita um objeto Orbital diretamente, ou
                         None usa o primeiro orbital visível na cena.
            plane      : 'xz', 'xy' ou 'yz'
            resolution : pontos por eixo do corte

        Retorna:
            Figure do matplotlib, ou None se não houver orbital disponível.
        """
        orbital = self._resolve_orbital_for_slice(orbital_id)
        if orbital is None:
            print("⚠ Nenhum orbital visível para gerar o corte 2D")
            return None
        return self.renderer.render_slice_2d(orbital, plane=plane, resolution=resolution)

    def show_orbital_slice(self, orbital_id=None, plane: str = 'xz',
                            resolution: int = 400):
        """
        Abre (numa janela matplotlib separada) o corte 2D de um orbital.
        """
        fig = self.get_orbital_slice_figure(orbital_id, plane=plane, resolution=resolution)
        if fig is None:
            return None

        import matplotlib.pyplot as plt
        plt.show()
        return fig

    def save_orbital_slice(self, filepath: str, orbital_id=None,
                            plane: str = 'xz', resolution: int = 400):
        """
        Salva em PNG o corte 2D de amplitude e probabilidade.
        """
        orbital = self._resolve_orbital_for_slice(orbital_id)
        if orbital is None:
            print("⚠ Nenhum orbital visível para gerar o corte 2D")
            return None

        from utils.slice2d import save_orbital_slice as _save_slice_png
        range_max = self.renderer._get_range_for_orbital(orbital)
        _save_slice_png(orbital, filepath, plane=plane, range_max=range_max,
                         resolution=resolution)
        print(f"✓ Corte 2D salvo em {filepath}")
        return filepath

    def _resolve_orbital_for_slice(self, orbital_id=None):
        """
        Resolve o orbital-alvo do corte 2D:
          - None            → primeiro orbital visível na cena 3D
          - str             → chave em self.visible_orbitals
          - objeto Orbital  → usado diretamente (permite cortar um orbital
                               mesmo que não esteja sendo exibido em 3D)
        """
        if orbital_id is None:
            if not self.visible_orbitals:
                return None
            return next(iter(self.visible_orbitals.values()))
        if isinstance(orbital_id, str):
            return self.visible_orbitals.get(orbital_id)
        return orbital_id

    def run(self):
        """Inicia o loop principal de renderização."""
        self.is_running = True
        print("🟢 Simulador iniciado")
        print(f"   Elemento: {element_info_string(self.atom)}")
        print(f"   Modo: {self.renderer.mode}")
        print("\nControles interativos:")
        print("  - Mouse: rotacionar câmera")
        print("  - Scroll: zoom")
        print("  - 'r': reset câmera")
        print("  - 'q': sair")
        
        self.scene.show()
        self.is_running = False
    
    def step(self, dt: float = 0.016):
        """Executa um passo de simulação."""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_update_time = current_time
        self.scene.update()
    
    def close(self):
        """Fecha o simulador."""
        self.scene.close()
        self.is_running = False
        print("🔴 Simulador encerrado")
    
    def get_info_string(self) -> str:
        """Retorna telemetria operacional."""
        info = f"{element_info_string(self.atom)}\n"
        info += f"Modo: {self.renderer.mode}\n"
        info += f"Orbitais visíveis: {len(self.visible_orbitals)}\n"
        info += f"FPS: {self.fps:.1f}"
        return info
