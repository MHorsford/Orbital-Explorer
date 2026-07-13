"""Ponto de entrada da interface do Orbital Explorer."""

import sys
import os


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from atom.atom import Atom
from simulator.simulator import Simulator
from ui.orbital_explorer import launch_explorer


def main():
    print("=" * 70)
    print("  🧪 ORBITAL EXPLORER — Simulador Interativo de Orbitais")
    print("=" * 70)
    print("\nIniciando simulador...")

    atom = Atom(Z=1)

    sim = Simulator(atom=atom, title="Orbital Explorer — 3D Interactive")
    sim.set_render_mode('isosurface')

    print(f"✓ Simulador criado: {atom.get_element_name()}")
    print("✓ Abrindo UI interativa...")

    app, explorer = launch_explorer(sim)

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
