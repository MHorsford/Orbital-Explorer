"""
orbitals/wavefunction.py
Funções de onda hidrogenoides (ψ) para orbitais atômicos.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import math
from scipy.special import assoc_laguerre, sph_harm_y
from physics.constants import A0_ANGSTROM, A0_METERS  


class HydrogenWavefunction:
    """
    Classe responsável por calcular as funções de onda do átomo de Hidrogênio
    e átomos hidrogenoides (usando Z efetivo).
    """
    
    def __init__(self, use_angstrom=True):
        if use_angstrom:
            self.a0 = A0_ANGSTROM  # [Å] 0.529177 Angstroms
            self.unit = "Angstrom"
        else:
            self.a0 = A0_METERS    # [m] 5.292e-11 metros
            self.unit = "meters"
    
    # RADIAL
    
    def radial_wavefunction(self, r, n, l, Z=1):
        """
        Calcula a parte radial R_{n,l}(r)
        """
        # rho (variável adimensional)
        rho = (2 * Z * r) / (n * self.a0)
        
        # Normalização
        norm = np.sqrt( (2*Z/(n*self.a0))**3 * math.factorial(n - l - 1) / 
                   (2 * n * (math.factorial(n + l))))
        
        # Termos
        exp_term = np.exp(-rho / 2)
        rho_term = rho ** l
        laguerre_poly = assoc_laguerre(rho, n - l - 1, 2 * l + 1)
        
        R_nl = norm * rho_term * exp_term * laguerre_poly
        return R_nl

    def angular_part(self, theta, phi, l, m):
        """
        Retorna harmônicos esféricos reais para qualquer l e m.

        Combinações utilizadas:
        - m = 0   → Y_l^0 (já é real)
        - m > 0   → (Y_l^{-m} + (-1)^m Y_l^m) / sqrt(2)
        - m < 0   → i * (Y_l^{|m|} - (-1)^{|m|} Y_l^{-|m|}) / sqrt(2)

        Para m != 0, a simetria de conjugação evita uma segunda avaliação:
            Y_l^{-m} = (-1)^m * conj(Y_l^m)
        """

        if m == 0:
            # Y_l^0 é puramente real (fase de Condon-Shortley já incluída)
            return np.real(sph_harm_y(l, 0, theta, phi))
        
        elif m > 0:
            # Combinação par (cosseno) → gera orbitais como p_x, d_xy, etc.
            Y_m = sph_harm_y(l, m, theta, phi)
            Y_minus_m = ((-1) ** m) * np.conj(Y_m)
            return np.real((Y_minus_m + (-1)**m * Y_m) / np.sqrt(2))
        
        else:  # m < 0
            # Combinação ímpar (seno) → gera orbitais como p_y, d_xz, etc.
            k = -m  # |m|
            Y_k = sph_harm_y(l, k, theta, phi)
            Y_minus_k = ((-1) ** k) * np.conj(Y_k)
            # O fator 1j garante que o resultado seja puramente real
            return np.real(1j * (Y_k - (-1)**k * Y_minus_k) / np.sqrt(2))
    
    # FUNÇÃO DE ONDA COMPLETA
    
    def psi(self, r, theta, phi, n, l, m, Z=1):
        """Função de onda completa usando a parte angular REAL."""
        R_nl = self.radial_wavefunction(r, n, l, Z)
        Y_lm_real = self.angular_part(theta, phi, l, m)
        return R_nl * Y_lm_real
    
    def probability_density(self, r, theta, phi, n, l, m, Z=1):
        """ |ψ|² """
        wave = self.psi(r, theta, phi, n, l, m, Z)
        return np.abs(wave)**2

   
    
    def psi_1s(self, r, theta=None, phi=None, Z=1):
        """Orbital 1s - não depende de theta/phi"""
        rho = (2 * Z * r) / self.a0
        norm = np.sqrt((Z**3) / (np.pi * self.a0**3))
        return norm * np.exp(-rho / 2)
    
    def psi_2s(self, r, theta=None, phi=None, Z=1):
        """Orbital 2s"""
        rho = (2 * Z * r) / (2 * self.a0)
        norm = np.sqrt((Z**3) / (8 * np.pi * self.a0**3))
        return norm * (2 - rho) * np.exp(-rho / 2)
    
    def psi_2p_z(self, r, theta, phi=None, Z=1):
        """Orbital 2p_z"""
        rho = (2 * Z * r) / (2 * self.a0)
        norm = np.sqrt((Z**3) / (32 * np.pi * self.a0**3))
        return norm * rho * np.exp(-rho / 2) * np.cos(theta)
    
    def psi_2p_x(self, r, theta, phi, Z=1):
        """Orbital 2p_x"""
        rho = (2 * Z * r) / (2 * self.a0)
        norm = np.sqrt((Z**3) / (32 * np.pi * self.a0**3))
        return norm * rho * np.exp(-rho / 2) * np.sin(theta) * np.cos(phi)
    
    def psi_2p_y(self, r, theta, phi, Z=1):
        """Orbital 2p_y"""
        rho = (2 * Z * r) / (2 * self.a0)
        norm = np.sqrt((Z**3) / (32 * np.pi * self.a0**3))
        return norm * rho * np.exp(-rho / 2) * np.sin(theta) * np.sin(phi)

    # 3D
    
    def generate_grid(self, size=80, range_max=15.0):
        """
        Cria grid 3D cartesiano e converte para coordenadas esféricas de forma estável.
        """
        x = np.linspace(-range_max, range_max, size)
        y = np.linspace(-range_max, range_max, size)
        z = np.linspace(-range_max, range_max, size)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # O deslocamento infinitesimal estabiliza a conversão na origem.
        eps = 1e-14
        X_safe = X + eps
        Y_safe = Y + eps
        Z_safe = Z + eps
        
        # Coordenadas esféricas estáveis
        R = np.sqrt(X_safe**2 + Y_safe**2 + Z_safe**2)
        Theta = np.arctan2(np.sqrt(X_safe**2 + Y_safe**2), Z_safe)   # Ângulo polar
        Phi = np.arctan2(Y_safe, X_safe)                           # Ângulo azimutal
        
        return X, Y, Z, R, Theta, Phi
    

    def evaluate_on_grid(self, n, l, m, Z=1, size=80, range_max=15.0):
        """
        Avalia a amplitude da função de onda (ψ) em todo o grid 3D.
        Usa harmônicos esféricos reais para todos os orbitais.
        """
        X, Y, Z_coord, R, Theta, Phi = self.generate_grid(size, range_max)
        
        # Radial
        radial = self.radial_wavefunction(R, n, l, Z)
        
        # Angular real (funciona para s, p, d, f, g...)
        angular = self.angular_part(Theta, Phi, l, m)
        
        # Função de onda
        wave = radial * angular
        
        return wave, X, Y, Z_coord
