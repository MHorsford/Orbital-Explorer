"""
particles/electron.py

Classe Electron — partícula carregada negativamente, orbita o núcleo.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from particles.particle import Particle
from physics.constants import M_ELECTRON, E_CHARGE


class Electron(Particle):
    """
    Elétron — partícula com carga -e e massa ~1/1836 da massa do próton.
    
    Ocupam orbitais ao redor do núcleo segundo as regras quânticas.
    No modelo clássico, não têm posição definida — representamos a partícula
    como entidade cujos orbitais são descritos por funções de onda.
    """

    def __init__(self, position: np.ndarray = None):
        """
        Parâmetros:
            position : posição no espaço (padrão: origem)
        """
        super().__init__(
            position=position,
            charge=-E_CHARGE,           # carga negativa
            mass=M_ELECTRON,            # massa do elétron (referência padrão)
            radius=0.05,                # raio visual ~0.05 Å (bem pequeno)
            color=(0.2, 0.6, 0.9)       # azul para negativo
        )

    def get_name(self) -> str:
        return "Elétron"