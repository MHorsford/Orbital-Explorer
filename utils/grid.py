"""
utils/grid.py

Funções para gerar grids 3D cartesianos e converter para coordenadas esféricas.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np


def make_grid(size: int = 80, range_max: float = 8.0):
    """
    Cria um grid 3D cartesiano (X, Y, Z) e o converte para esféricas (R, Theta, Phi).

    Parâmetros:
        size      : número de pontos por dimensão (ex: 80 → 80³ pontos)
        range_max : extensão da caixa em unidades de Bohr (ex: ±8 a.u.)

    Retorna:
        (X, Y, Z, R, THETA, PHI) - todos arrays numpy de shape (size, size, size)
    """
    # Grid cartesiano
    x = np.linspace(-range_max, range_max, size)
    y = np.linspace(-range_max, range_max, size)
    z = np.linspace(-range_max, range_max, size)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Converter para esféricas
    R = np.sqrt(X**2 + Y**2 + Z**2)
    
    # Ângulo polar (colatitude): de 0 a π
    # arctan2(√(x²+y²), z)
    Theta = np.arctan2(np.sqrt(X**2 + Y**2), Z)
    
    # Ângulo azimutal: de -π a π
    # arctan2(y, x)
    Phi = np.arctan2(Y, X)

    # Evitar divisão por zero
    R[R == 0] = 1e-10

    return X, Y, Z, R, Theta, Phi


def make_grid_cartesian_only(size: int = 80, range_max: float = 8.0):
    """
    Versão simplificada que retorna apenas o grid cartesiano.
    Útil quando não precisa de coordenadas esféricas.
    """
    x = np.linspace(-range_max, range_max, size)
    y = np.linspace(-range_max, range_max, size)
    z = np.linspace(-range_max, range_max, size)
    return np.meshgrid(x, y, z, indexing='ij')


def spherical_to_cartesian(r, theta, phi):
    """
    Converte coordenadas esféricas para cartesianas.
    
    Parâmetros:
        r     : raio (pode ser escalar ou array)
        theta : ângulo polar/colatitude em radianos (0 a π)
        phi   : ângulo azimutal em radianos (-π a π)
    
    Retorna:
        (x, y, z) tupla ou arrays
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def cartesian_to_spherical(x, y, z):
    """
    Converte coordenadas cartesianas para esféricas.
    
    Parâmetros:
        x, y, z : coordenadas cartesianas (podem ser escalares ou arrays)
    
    Retorna:
        (r, theta, phi) tupla ou arrays
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi = np.arctan2(y, x)
    
    # np.where trata a origem de forma uniforme para escalares e arrays.
    r = np.where(r == 0, 1e-10, r)
    
    return r, theta, phi


def points_on_sphere(n_points: int, radius: float = 1.0):
    """
    Gera n_points distribuídos uniformemente numa esfera usando Fibonacci sphere.
    
    Parâmetros:
        n_points : quantidade de pontos
        radius   : raio da esfera
    
    Retorna:
        Array (n_points, 3) com coordenadas cartesianas
    """
    indices = np.arange(0, n_points, dtype=float) + 0.5
    
    theta = np.arccos(1 - 2 * indices / n_points)  # colatitude
    phi = np.pi * (1 + 5**0.5) * indices            # azimute (golden angle)
    
    x, y, z = spherical_to_cartesian(radius, theta, phi)
    
    points = np.column_stack([x, y, z])
    return points


def normalize_array(arr: np.ndarray, vmin: float = 0.0, vmax: float = 1.0):
    """
    Normaliza um array para um intervalo [vmin, vmax].
    
    Parâmetros:
        arr  : array numpy
        vmin : valor mínimo do intervalo
        vmax : valor máximo do intervalo
    
    Retorna:
        Array normalizado
    """
    arr_min = arr.min()
    arr_max = arr.max()
    
    if arr_max == arr_min:
        return np.full_like(arr, (vmin + vmax) / 2, dtype=float)
    
    normalized = (arr - arr_min) / (arr_max - arr_min)
    return normalized * (vmax - vmin) + vmin
