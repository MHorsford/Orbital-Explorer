"""
utils/helpers.py

Funções auxiliares e utilitários diversos.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np


# Em utils/helpers.py, substitua a função quantum_label por esta:

def quantum_label(n: int, l: int, m: int = None) -> str:
    """
    Converte números quânticos em rótulo legível.
    
    Exemplos:
        quantum_label(1, 0) → "1s"
        quantum_label(2, 1, 0) → "2p_z"
        quantum_label(3, 2, 1) → "3d_+1"
        quantum_label(5, 4, 0) → "5g_0"  (novo!)
    
    Parâmetros:
        n : número quântico principal
        l : número quântico azimutal (0=s, 1=p, 2=d, 3=f, 4=g, 5=h, 6=i)
        m : número quântico magnético (opcional)
    
    Retorna:
        String com o rótulo
    """
    from orbitals.orbital_types import get_orbital_type
    
    # Usar o tipo oficial do orbital_types
    orbital_type = get_orbital_type(l)
    l_letter = orbital_type.letter
    
    label = f"{n}{l_letter}"
    
    if m is not None:
        if l == 0:
            pass  # s não tem diferentes m
        elif l == 1:  # p
            # A combinação em cosseno corresponde a x; a combinação em seno, a y.
            m_names = {-1: 'y', 0: 'z', 1: 'x'}
            label += f"_{m_names.get(m, f'{m}')}"
        elif l == 2:  # d
            m_names = {-2: 'xy', -1: 'yz', 0: 'z²', 1: 'xz', 2: 'x²-y²'}
            label += f"_{m_names.get(m, f'{m}')}"
        elif l == 3:  # f
            # Nomes não são padronizados no ensino básico, usamos m numérico
            label += f"_{m:+d}"
        else:
            # Para g, h, i, usamos apenas o valor de m
            label += f"_{m:+d}"
    
    return label


def orbital_info_string(orbital) -> str:
    """
    Gera uma string descritiva de um orbital.
    
    Exemplo:
        "2p (m=0) — 1/2 elétrons | Z_eff=3.45"
    """
    from orbitals.orbital_types import get_orbital_type
    
    label = quantum_label(orbital.n, orbital.l, orbital.m)
    info = f"{label} — {orbital.electrons}/2 e⁻"
    
    if hasattr(orbital, '_Z_eff'):
        info += f" | Z_eff={orbital.Z_eff:.2f}"
    
    return info


def element_info_string(atom) -> str:
    """
    Gera uma string descritiva de um átomo.
    
    Exemplo:
        "Carbono (C) — Z=6, 6 elétrons, Config: 1s² 2s² 2p²"
    """
    symbol = atom.get_element_symbol()
    name = atom.get_element_name()
    config = atom.get_electron_config()
    
    return f"{name} ({symbol}) — Z={atom.Z}, {atom.N_electrons} e⁻, Config: {config}"


def color_to_rgb_normalized(color):
    """
    Converte cor de diferentes formatos para RGB normalizado (0-1).
    
    Formatos aceitos:
        - Tupla (r, g, b) com valores 0-1
        - String hex "#RRGGBB"
        - String hex "#RGB" (expandido)
    
    Retorna:
        Tupla (r, g, b) normalizada entre 0 e 1
    """
    if isinstance(color, (tuple, list)):
        return tuple(color[:3])
    
    if isinstance(color, str):
        color = color.strip()
        if color.startswith('#'):
            hex_str = color[1:]
            if len(hex_str) == 3:
                # Expandir #RGB para #RRGGBB
                hex_str = ''.join([c * 2 for c in hex_str])
            
            r = int(hex_str[0:2], 16) / 255.0
            g = int(hex_str[2:4], 16) / 255.0
            b = int(hex_str[4:6], 16) / 255.0
            return (r, g, b)
    
    # Default: cinza neutro
    return (0.5, 0.5, 0.5)


def clamp(value, min_val, max_val):
    """
    Limita um valor ao intervalo [min_val, max_val].
    """
    return max(min_val, min(max_val, value))


def lerp(a, b, t):
    """
    Interpolação linear: a + t*(b - a).
    """
    return a + t * (b - a)


def distance_3d(p1, p2):
    """
    Calcula a distância euclidiana entre dois pontos 3D.
    
    Parâmetros:
        p1, p2 : tuplas ou arrays (x, y, z)
    
    Retorna:
        Distância euclidiana
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.linalg.norm(p2 - p1)


def vec_normalize(vec):
    """
    Normaliza um vetor 3D para magnitude 1.
    
    Parâmetros:
        vec : array ou tupla (x, y, z)
    
    Retorna:
        Vetor normalizado
    """
    vec = np.array(vec, dtype=float)
    mag = np.linalg.norm(vec)
    if mag < 1e-10:
        return vec
    return vec / mag
