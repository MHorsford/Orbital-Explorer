"""
physics/coulomb.py

Cálculos de força eletrostática e energia de Coulomb para o simulador.
Útil para visualizar interações entre partículas e estimar energias de ionização.

Referência: Lei de Coulomb e energia eletrostática em SI.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from physics.constants import K_COULOMB, E_CHARGE, EV_TO_JOULE, E_IONIZATION_H


# FORÇA DE COULOMB

def coulomb_force(q1: float, q2: float, r: float) -> float:
    """
    Calcula a magnitude da força eletrostática entre duas cargas.
    F = k · |q1| · |q2| / (r² + epsilon)
    """
    # O limite inferior evita a singularidade em r=0.
    eps = 1e-25
    if r <= 0:
        r = eps
    return K_COULOMB * abs(q1) * abs(q2) / (r ** 2 + eps)


def coulomb_force_vectors(pos1: np.ndarray, q1: float,
                          pos2: np.ndarray, q2: float) -> np.ndarray:
    """
    Calcula a força vetorial sobre a partícula 1 devido à partícula 2.
    """
    dr = pos1 - pos2
    r_mag = np.linalg.norm(dr)

    if r_mag < 1e-15:
        return np.zeros(3)

    r_hat = dr / r_mag
    F_mag = coulomb_force(q1, q2, r_mag)

    if q1 * q2 > 0:
        return F_mag * r_hat
    else:
        return -F_mag * r_hat


# ENERGIA DE COULOMB

def coulomb_potential_energy(q1: float, q2: float, r: float) -> float:
    """
    Calcula a energia potencial eletrostática entre duas cargas.
    """
    eps = 1e-25
    if r <= 0:
        r = eps
    return K_COULOMB * q1 * q2 / (r + eps)


def coulomb_potential_energy_ev(q1: float, q2: float, r: float) -> float:
    """
    O mesmo que coulomb_potential_energy, mas retorna em eV.
    """
    U_joule = coulomb_potential_energy(q1, q2, r)
    return U_joule / EV_TO_JOULE


# ENERGIA DE IONIZAÇÃO EFETIVA (Bohr)

def ionization_energy_bohr(Z_eff: float, n: int) -> float:
    """
    Estima a energia de ionização de um elétron no nível n com carga efetiva Z_eff.
    E_n = 13.6 eV · (Z_eff)² / n²
    """
    if n <= 0:
        return 0.0
    E_n = E_IONIZATION_H * (Z_eff ** 2) / (n ** 2)
    return abs(E_n)


def binding_energy(Z_eff: float, n: int) -> float:
    """
    Retorna a energia de ligação (negativa) do elétron no nível n.
    """
    return -ionization_energy_bohr(Z_eff, n)


# RAIO ORBITAL MÉDIO (Bohr)

def bohr_orbital_radius(Z_eff: float, n: int) -> float:
    """
    Estima o raio médio do orbital no nível n com carga nuclear efetiva Z_eff.
    <r> = a₀ · n² / Z_eff
    """
    from physics.constants import A0_ANGSTROM
    # Evita divisão por zero caso Z_eff seja nulo
    z_safe = max(0.1, Z_eff)
    return A0_ANGSTROM * (n ** 2) / z_safe


# REPULSÃO E ATRAÇÃO INTEGRADA

def nuclear_attraction_energy(Z: int, Z_eff: float, n: int) -> float:
    """
    Energia de atração elétron-núcleo real para um elétron no nível n.
    """
    r_avg = bohr_orbital_radius(Z_eff, n) * 1e-10  # Å → m
    
    # A atração nuclear usa a carga real Ze; Z_eff entra apenas no raio médio.
    U_joule = coulomb_potential_energy(Z * E_CHARGE, -E_CHARGE, r_avg)
    return U_joule / EV_TO_JOULE


def electron_electron_repulsion_estimate(Z: int, n1: int, n2: int) -> float:
    """
    Estimativa aproximada da repulsão de Coulomb entre dois elétrons.
    """
    # Estimativa de Z_eff para cada nível
    Z_eff_1 = max(1, Z - 0.5)
    Z_eff_2 = max(1, Z - (n1 + n2 - 1))

    r1 = bohr_orbital_radius(Z_eff_1, n1) * 1e-10  # → m
    r2 = bohr_orbital_radius(Z_eff_2, n2) * 1e-10  # → m
    
    # No mesmo nível, aproxima a separação pelo raio médio da camada.
    if n1 == n2:
        r_sep = r1  # Distância média estatística da casca
    else:
        r_sep = abs(r1 - r2)
        
    if r_sep < 1e-15:
        r_sep = 1e-15

    U_joule = coulomb_potential_energy(-E_CHARGE, -E_CHARGE, r_sep)
    return U_joule / EV_TO_JOULE
