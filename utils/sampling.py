"""
utils/sampling.py

Amostragem Monte Carlo para gerar nuvens de pontos probabilísticas
que representam a densidade eletrônica de um orbital.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from utils.grid import cartesian_to_spherical, spherical_to_cartesian


def sample_orbital_rejection(psi_squared_func, n_samples: int = 5000, 
                             range_max: float = 8.0, max_attempts: int = 100000):
    """
    Amostragem por rejeição: gera pontos aleatórios no cubo e aceita
    com probabilidade proporcional a |ψ|².

    Parâmetros:
        psi_squared_func : função que retorna |ψ|² dado (x, y, z)
        n_samples        : número de pontos desejados
        range_max        : extensão do cubo (±range_max)
        max_attempts     : máximo de tentativas antes de desistir

    Retorna:
        Array (n_samples, 3) com coordenadas dos pontos aceitos
    """
    points = []
    
    # Encontrar o máximo de |ψ|² numa amostra inicial (mais rápido que em todo espaço)
    sample_size = 1000
    x_sample = np.random.uniform(-range_max, range_max, sample_size)
    y_sample = np.random.uniform(-range_max, range_max, sample_size)
    z_sample = np.random.uniform(-range_max, range_max, sample_size)
    psi_max = np.max(psi_squared_func(x_sample, y_sample, z_sample))
    
    if psi_max < 1e-10:
        # Orbital vazio ou muito pequeno — retornar pontos aleatórios no origem
        return np.random.normal(0, 1e-2, (n_samples, 3))
    
    attempts = 0
    while len(points) < n_samples and attempts < max_attempts:
        # Gerar candidato aleatório no cubo
        x = np.random.uniform(-range_max, range_max)
        y = np.random.uniform(-range_max, range_max)
        z = np.random.uniform(-range_max, range_max)
        
        # Calcular |ψ|²
        psi_sq = psi_squared_func(np.array([x]), np.array([y]), np.array([z]))[0]
        
        # Aceitar com probabilidade proporcional a |ψ|²
        if np.random.random() < psi_sq / psi_max:
            points.append([x, y, z])
        
        attempts += 1
    
    if len(points) < n_samples:
        print(f"⚠ Aviso: só conseguiu {len(points)}/{n_samples} pontos após {attempts} tentativas")
    
    return np.array(points, dtype=np.float32)


def sample_orbital_grid(psi_squared_func, n_samples: int = 5000,
                        size: int = 40, range_max: float = 8.0):
    """
    Amostragem em grid: calcula |ψ|² numa malha, normaliza como distribuição
    de probabilidade e amostra pontos com base nela.

    Mais rápido que rejection para distribuições bem localizadas.

    Parâmetros:
        psi_squared_func : função que retorna |ψ|² em arrays
        n_samples        : número de pontos desejados
        size             : resolução do grid auxiliar (ex: 40 → 40³ pontos)
        range_max        : extensão da caixa

    Retorna:
        Array (n_samples, 3) com coordenadas dos pontos
    """
    # Criar grid
    x = np.linspace(-range_max, range_max, size)
    y = np.linspace(-range_max, range_max, size)
    z = np.linspace(-range_max, range_max, size)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Calcular |ψ|²
    psi_sq = psi_squared_func(X, Y, Z)
    psi_sq = np.abs(psi_sq)
    
    # Normalizar para distribuição de probabilidade
    psi_sq_flat = psi_sq.flatten()
    psi_sq_flat = psi_sq_flat / (psi_sq_flat.sum() + 1e-15)
    
    # Amostragem multinomial: escolher índices com base em psi_sq
    indices = np.random.choice(len(psi_sq_flat), size=n_samples, p=psi_sq_flat)
    
    # Converter índices para coordenadas
    idx_3d = np.unravel_index(indices, psi_sq.shape)
    
    points = np.column_stack([
        X[idx_3d],
        Y[idx_3d],
        Z[idx_3d]
    ])
    
    return points.astype(np.float32)


def sample_orbital_spherical_shell(psi_squared_func, n_samples: int = 5000,
                                    n_radial: int = 50, n_angular: int = 100,
                                    range_max: float = 8.0):
    """
    Amostragem em coordenadas esféricas: primeiro integra |ψ|² sobre ângulos
    (obtém a densidade radial), depois amostra r e depois (θ, φ) uniformemente.

    Melhor para orbitais altamente não-esféricos (p, d, f).

    Parâmetros:
        psi_squared_func : função que retorna |ψ|² em arrays
        n_samples        : número de pontos desejados
        n_radial         : pontos para integração radial
        n_angular        : pontos para integração angular (por shell)
        range_max        : extensão da caixa

    Retorna:
        Array (n_samples, 3) com coordenadas dos pontos
    """
    points = []
    
    # Grid radial
    r_vals = np.linspace(0.01, range_max, n_radial)
    
    # Calcular densidade radial (integração bruta sobre ângulos)
    theta_vals = np.linspace(0, np.pi, n_angular)
    phi_vals = np.linspace(-np.pi, np.pi, n_angular)
    
    density_radial = np.zeros(n_radial)
    
    for i, r in enumerate(r_vals):
        density_shell = 0
        for theta in theta_vals:
            for phi in phi_vals:
                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)
                
                psi_sq = psi_squared_func(np.array([x]), np.array([y]), np.array([z]))[0]
                density_shell += psi_sq
        
        density_radial[i] = density_shell * (r ** 2)  # fator de volume
    
    # Normalizar densidade radial
    density_radial = density_radial / (density_radial.sum() + 1e-15)
    
    # Amostragem de r
    r_samples = np.random.choice(r_vals, size=n_samples, p=density_radial)
    
    # Amostragem uniforme de ângulos
    theta_samples = np.arccos(np.random.uniform(-1, 1, n_samples))  # cos(θ) uniforme
    phi_samples = np.random.uniform(-np.pi, np.pi, n_samples)
    
    # Converter para cartesianas
    x = r_samples * np.sin(theta_samples) * np.cos(phi_samples)
    y = r_samples * np.sin(theta_samples) * np.sin(phi_samples)
    z = r_samples * np.cos(theta_samples)
    
    points = np.column_stack([x, y, z])
    return points.astype(np.float32)


def add_noise_to_points(points: np.ndarray, std_dev: float = 0.01):
    """
    Adiciona ruído gaussiano aos pontos para suavizar a nuvem.
    
    Parâmetros:
        points  : array (N, 3)
        std_dev : desvio padrão do ruído
    
    Retorna:
        Array (N, 3) com ruído adicionado
    """
    noise = np.random.normal(0, std_dev, points.shape)
    return points + noise