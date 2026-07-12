"""
particles/particle.py

Classe base abstrata Particle — define a interface comum para prótons, nêutrons e elétrons.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from abc import ABC, abstractmethod
import numpy as np
from physics.constants import M_ELECTRON, M_PROTON, M_NEUTRON, E_CHARGE


class Particle(ABC):
    """
    Classe abstrata que representa uma partícula fundamental no átomo.
    
    Atributos base:
        position : posição no espaço (x, y, z) em metros
        charge   : carga em Coulombs
        mass     : massa em kg
        radius   : raio visual para renderização em Angstroms
        color    : cor para visualização (tuple RGB 0-1 ou hex)
    """

    def __init__(self, 
                 position: np.ndarray = None,
                 charge: float = 0.0,
                 mass: float = 0.0,
                 radius: float = 0.1,
                 color: tuple = (0.5, 0.5, 0.5)):
        """
        Parâmetros:
            position : array [x, y, z] em metros. Padrão: origem (0, 0, 0)
            charge   : carga em Coulombs
            mass     : massa em kg
            radius   : raio visual em Angstroms
            color    : (R, G, B) cada valor em [0, 1] ou string hex
        """
        if position is None:
            position = np.array([0.0, 0.0, 0.0])
        
        self.position = np.array(position, dtype=float)
        self.charge = charge
        self.mass = mass
        self.radius = radius
        self.color = color

    @abstractmethod
    def get_name(self) -> str:
        """Retorna o nome da partícula (ex: 'Próton', 'Elétron')"""
        pass

    def move(self, displacement: np.ndarray) -> None:
        """Move a partícula por um deslocamento"""
        self.position += np.array(displacement, dtype=float)

    def set_position(self, position: np.ndarray) -> None:
        """Define a posição absoluta da partícula"""
        self.position = np.array(position, dtype=float)

    def distance_to(self, other: 'Particle') -> float:
        """Calcula a distância até outra partícula"""
        return np.linalg.norm(self.position - other.position)

    def __str__(self) -> str:
        return f"{self.get_name()} @ ({self.position[0]:.3e}, {self.position[1]:.3e}, {self.position[2]:.3e}) | q={self.charge:.3e}C"

    def __repr__(self) -> str:
        return self.__str__()