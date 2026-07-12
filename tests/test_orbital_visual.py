# tests/test_orbital_single.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pyvista as pv
from orbitals.orbital import Orbital

pv.set_plot_theme("dark")

def visualize_orbital(n, l, m=0, electrons=2, range_max=None, title=""):
    """Visualiza um único orbital"""
    print(f"🔬 Gerando orbital: {n}{['s','p','d','f'][l]}{m if l > 0 else ''}")
    
    orb = Orbital(n=n, l=l, m=m, electrons=electrons)
    
    if range_max is None:
        range_max = 6 if n <= 2 else 12
    
    orb.calculate_density(size=75 if n==1 else 90, range_max=range_max)
    
    # ================= PLOT =================
    plotter = pv.Plotter(window_size=[1400, 900])
    plotter.set_background('black')
    
    grid = pv.wrap(orb.density_grid)
    
    # Níveis diferentes conforme o tipo de orbital
    if l == 0:   # s (esférico)
        isos = [0.3, 0.6, 0.85]
    else:        # p, d, etc
        isos = [0.05, 0.15, 0.4, 0.7]
    
    plotter.add_mesh(grid.contour(isosurfaces=isos), 
                     color=orb.color, 
                     opacity=0.85, 
                     name=orb.name)
    
    # Núcleo
    plotter.add_mesh(pv.Sphere(radius=0.15, center=(0,0,0)), color='yellow')
    
    plotter.add_text(f"{title}\n{orb.name}  ({electrons} elétrons)", 
                     position='upper_edge', color='white', font_size=16)
    
    plotter.camera_position = 'yz'
    plotter.show()


# ===================== MENU DE TESTES =====================
if __name__ == "__main__":
    print("=== Teste Individual de Orbitais ===\n")
    
    visualize_orbital(n=1, l=0, electrons=2, title="1s - Orbital Esférico")
    

    
    # visualize_orbital(n=2, l=0, electrons=2, title="2s - Orbital Esférico com Nó")
    
    # visualize_orbital(n=2, l=1, m=0, electrons=2, title="2p_z - Haltere na direção Z")
    
    # visualize_orbital(n=2, l=1, m=1, electrons=2, title="2p_x - Haltere na direção X")