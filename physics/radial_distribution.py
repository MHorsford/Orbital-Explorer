"""Distribuição radial de orbitais hidrogenoides."""

from dataclasses import dataclass

import numpy as np
from scipy.special import roots_genlaguerre

from orbitals.wavefunction import HydrogenWavefunction
from physics.constants import A0_ANGSTROM


@dataclass(frozen=True)
class RadialDistribution:
    radius_angstrom: np.ndarray
    radial_amplitude: np.ndarray
    scaled_radial_amplitude: np.ndarray
    probability_density: np.ndarray
    radial_nodes_angstrom: np.ndarray
    radial_node_count: int
    angular_node_count: int
    total_node_count: int
    most_probable_radius_angstrom: float
    mean_radius_angstrom: float
    range_max_angstrom: float


def _integrate(values, coordinates) -> float:
    """Integra mantendo compatibilidade com versões anteriores do NumPy."""
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    return float(trapezoid(values, coordinates))


def radial_node_positions(n: int, l: int, z_eff: float) -> np.ndarray:
    """Retorna as posições analíticas dos nós radiais, sem incluir r=0."""
    node_count = n - l - 1
    if node_count <= 0:
        return np.array([], dtype=float)
    roots, _ = roots_genlaguerre(node_count, 2 * l + 1)
    return roots * n * A0_ANGSTROM / (2 * z_eff)


def calculate_radial_distribution(
        n: int, l: int, z_eff: float = 1.0, resolution: int = 2500,
) -> RadialDistribution:
    """
    Calcula R_nl(r) e P(r)=r²|R_nl(r)|² em ångströms.

    P(r) é normalizada numericamente para que sua integral no intervalo
    exibido seja igual a um. A amplitude radial escalada preserva o sinal e
    serve apenas para comparar nós e lóbulos no gráfico.
    """
    if n < 1:
        raise ValueError("n deve ser maior ou igual a 1")
    if l < 0 or l >= n:
        raise ValueError("l deve satisfazer 0 ≤ l < n")
    if z_eff <= 0:
        raise ValueError("Z_eff deve ser positivo")
    if resolution < 400:
        raise ValueError("A resolução radial deve ser de pelo menos 400 pontos")

    range_max = 8.0 * (n ** 2) * A0_ANGSTROM / z_eff
    radius = np.linspace(0.0, range_max, resolution)
    wavefunction = HydrogenWavefunction(use_angstrom=True)
    radial = wavefunction.radial_wavefunction(radius, n, l, Z=z_eff)

    max_amplitude = float(np.max(np.abs(radial)))
    scaled_radial = (
        radial / max_amplitude
        if max_amplitude > 0 else np.zeros_like(radial)
    )

    probability = radius ** 2 * np.abs(radial) ** 2
    integral = _integrate(probability, radius)
    if not np.isfinite(integral) or integral <= 0:
        raise ValueError("Não foi possível normalizar a distribuição radial")
    probability = probability / integral

    most_probable = float(radius[int(np.argmax(probability))])
    mean_radius = _integrate(radius * probability, radius)
    radial_nodes = radial_node_positions(n, l, z_eff)
    radial_node_count = n - l - 1

    return RadialDistribution(
        radius_angstrom=radius,
        radial_amplitude=radial,
        scaled_radial_amplitude=scaled_radial,
        probability_density=probability,
        radial_nodes_angstrom=radial_nodes,
        radial_node_count=radial_node_count,
        angular_node_count=l,
        total_node_count=n - 1,
        most_probable_radius_angstrom=most_probable,
        mean_radius_angstrom=mean_radius,
        range_max_angstrom=range_max,
    )
