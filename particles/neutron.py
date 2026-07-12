"""
particles/neutron.py

Classe Neutron — partícula eletricamente neutra, núcleo do átomo.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from particles.particle import Particle
from physics.constants import M_NEUTRON


class Neutron(Particle):
    """
    Nêutron — partícula sem carga elétrica, massa ligeiramente maior que o próton.
    
    Usado no núcleo atômico para determinar o número de massa (A = Z + N).
    """

    def __init__(self, position: np.ndarray = None):
        """
        Parâmetros:
            position : posição no espaço (padrão: origem)
        """
        super().__init__(
            position=position,
            charge=0.0,                 # neutro
            mass=M_NEUTRON,             # massa do nêutron
            radius=0.1,                 # raio visual ~0.1 Å
            color=(0.4, 0.4, 0.4)       # cinza para neutro
        )

    def get_name(self) -> str:
        return "Nêutron"