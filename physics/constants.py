"""
physics/constants.py

Constantes físicas fundamentais e derivadas para o simulador de orbitais atômicos.
Todas as constantes estão em SI (Sistema Internacional) por padrão.

Referências:
  - CODATA 2018: https://nist.gov
  - IUPAC 2021: https://qmul.ac.uk
"""

import math

# CONSTANTES FUNDAMENTAIS (SI)

# Constante de Planck
H = 6.62607015e-34        # [J·s] Constante de Planck completa
HBAR = H / (2 * math.pi)  # [J·s] Constante de Planck reduzida (ℏ)

# Velocidade da luz
C = 299792458             # [m/s] Velocidade da luz no vácuo

# Carga elementar
E_CHARGE = 1.602176634e-19  # [C] Carga do elétron (em módulo)
EV_TO_JOULE = E_CHARGE      # [J/eV] 1 eV em Joules — mesmo que a carga elementar

# Massa de repouso
M_ELECTRON = 9.1093837015e-31    # [kg] Massa do elétron
M_PROTON = 1.67262192369e-27     # [kg] Massa do próton
M_NEUTRON = 1.67492749804e-27    # [kg] Massa do nêutron

# Permitividade do vácuo
EPSILON_0 = 8.8541878128e-12  # [F/m] Permitividade do vácuo

# Constante de Coulomb
K_COULOMB = 1 / (4 * math.pi * EPSILON_0)  # [N·m²/C²] ≈ 8.9875517923e9

# Constante gravitacional
G = 6.67430e-11  # [m³·kg⁻¹·s⁻²] Constante gravitacional


# CONSTANTES ATÔMICAS (Unidades Atômicas - a.u.)

# Raio de Bohr (unidade de comprimento)
A0_METERS = 5.29177210903e-11  # [m] Raio de Bohr em metros (SI)
A0_ANGSTROM = 0.529177210903   # [Å] Raio de Bohr em Angstroms

# Energia de Rydberg
RY_EV = 13.605693122994          # [eV] Energia de Rydberg
RY_JOULE = RY_EV * E_CHARGE      # [J] Energia de Rydberg em Joules

# 1 Hartree = 2 Rydberg.
RY_HARTREE = 0.5                 # [E_h] Energia de Rydberg em Hartrees

# Energia de ionização do hidrogênio
E_IONIZATION_H = RY_EV           # [eV] Primeira energia de ionização do H

# Comprimento de Compton do elétron
COMPTON_WAVELENGTH = H / (M_ELECTRON * C)  # [m]

# Raio clássico do elétron
ELECTRON_RADIUS = K_COULOMB * E_CHARGE**2 / (M_ELECTRON * C**2)  # [m]

# Magnetão de Bohr (momento magnético do elétron)
MU_B = E_CHARGE * HBAR / (2 * M_ELECTRON)  # [A·m² ou J/T]


# CONSTANTES ATÔMICAS DERIVADAS

# Frequência linear obtida por E = hν.
RY_FREQ = RY_JOULE / H           # [Hz] Frequência espectral de Rydberg

# Comprimento de onda de Rydberg (série H)
RY_WAVELENGTH = C / RY_FREQ      # [m] Comprimento de onda correto no vácuo

# Número de massa atômica (unidade de massa atômica)
AMU = 1.66053906660e-27  # [kg] Unidade de massa atômica

# Razão de massa próton/elétron
MASS_RATIO_P_E = M_PROTON / M_ELECTRON  # ≈ 1836.15


# CONSTANTES PARA SCREENING E CÁLCULOS MULTIELETRÔNICOS

# Referências para blindagens na mesma camada; screening.py trata níveis distintos.
SLATER_SCREENING = {
    's': 0.30,  # Orbitais 1s (0.35 para demais s)
    'p': 0.35,  
    'd': 0.35,  
    'f': 0.35   
}

PENETRATION_FACTOR = {
    's': 0.85,  # Penetam e sofrem menos blindagem de camadas internas (s/p)
    'p': 0.85,  
    'd': 0.35,  # Blindados fortemente por subníveis s/p internos
    'f': 0.35   
}


# CONSTANTES DE VISUALIZAÇÃO E SIMULAÇÃO

# Raios visuais para as partículas nucleares (em Angstroms)
PROTON_RADIUS_VISUAL = 0.1    # [Å]
NEUTRON_RADIUS_VISUAL = 0.1   # [Å]
ELECTRON_RADIUS_VISUAL = 0.05 # [Å]

# Isosurface de probabilidade padrão para visualização
ISO_VALUE_DEFAULT = 0.02  

# Resolução padrão do grid 3D
GRID_SIZE_DEFAULT = 80  
GRID_RANGE_DEFAULT = 8.0  # Extensão da caixa em unidades de Bohr (a.u.)


# FUNÇÕES DE CONVENIÊNCIA

def bohr_to_meters(r_bohr):
    """Converte raio em unidades de Bohr para metros"""
    return r_bohr * A0_METERS

def meters_to_bohr(r_meters):
    """Converte raio em metros para unidades de Bohr"""
    return r_meters / A0_METERS

def bohr_to_angstrom(r_bohr):
    """Converte raio em unidades de Bohr para Angstroms"""
    return r_bohr * A0_ANGSTROM

def angstrom_to_bohr(r_angstrom):
    """Converte raio em Angstroms para unidades de Bohr"""
    return r_angstrom / A0_ANGSTROM

def ev_to_joule(energy_ev):
    """Converte energia em eV para Joules"""
    return energy_ev * E_CHARGE

def joule_to_ev(energy_j):
    """Converte energia em Joules para eV"""
    return energy_j / E_CHARGE

def hartree_to_ev(energy_hartree):
    """Converte energia em Hartrees para eV"""
    return energy_hartree * (2 * RY_EV)

def ev_to_hartree(energy_ev):
    """Converte energia em eV para Hartrees"""
    return energy_ev / (2 * RY_EV)
