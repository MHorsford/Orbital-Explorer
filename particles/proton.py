"""
particles/proton.py

Classe Proton — partícula carregada positivamente, núcleo do átomo.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from particles.particle import Particle
from physics.constants import M_PROTON, E_CHARGE


class Proton(Particle):
    """
    Próton — partícula com carga +e e massa ~1836× a massa do elétron.
    
    Usado no núcleo atômico para determinar o número atômico (Z).
    """

    def __init__(self, position: np.ndarray = None):
        """
        Parâmetros:
            position : posição no espaço (padrão: origem)
        """
        super().__init__(
            position=position,
            charge=+E_CHARGE,           # carga positiva
            mass=M_PROTON,              # massa do próton
            radius=0.1,                 # raio visual ~0.1 Å
            color=(0.9, 0.3, 0.2)       # vermelho-coral para positivo
        )

    def get_name(self) -> str:
        return "Próton"