"""
utils/slice2d.py

Cálculo e representação de cortes planares da amplitude ψ e da densidade
de probabilidade |ψ|². Usa NumPy e Matplotlib, sem depender do PyVista.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

from utils.grid import cartesian_to_spherical
from utils.helpers import quantum_label


# Rótulos dos eixos horizontal e vertical de cada plano.
PLANE_LABELS = {
    'xz': ('x', 'z'),
    'xy': ('x', 'y'),
    'yz': ('y', 'z'),
}


def _build_plane_grid(plane: str, range_max: float, resolution: int):
    """
    Constrói o grid cartesiano de um plano que passa pela origem e retorna
    também seus dois eixos unidimensionais.

    Convenção: 'xz' → plano y=0, 'xy' → plano z=0, 'yz' → plano x=0.
    Orbitais com dependência azimutal podem exigir um plano específico para
    exibir seus lóbulos; por isso os três planos cartesianos são suportados.
    """
    if plane not in PLANE_LABELS:
        raise ValueError(f"Plano desconhecido: {plane!r}. Use 'xz', 'xy' ou 'yz'.")

    u = np.linspace(-range_max, range_max, resolution)
    v = np.linspace(-range_max, range_max, resolution)
    U, V = np.meshgrid(u, v, indexing='xy')  # V cresce verticalmente na imagem.
    zeros = np.zeros_like(U)

    if plane == 'xz':
        X, Y, Z = U, zeros, V
    elif plane == 'xy':
        X, Y, Z = U, V, zeros
    else:  # 'yz'
        X, Y, Z = zeros, U, V

    return X, Y, Z, u, v


def compute_orbital_slice(orbital, plane: str = 'xz', range_max: float = None,
                           resolution: int = 400):
    """
    Calcula psi e |psi|^2 num plano que passa pelo núcleo.

    Parâmetros:
        orbital    : objeto com atributos n, l, m, Z_eff (ex: Orbital)
        plane      : 'xz', 'xy' ou 'yz'
        range_max  : extensão do corte (mesma unidade da wavefunction — Å
                     neste projeto); se None, estima a partir de n
        resolution : pontos por eixo (resolution² pontos no total)

    Retorna:
        dict com 'u', 'v' (eixos 1D, shape (resolution,)),
        'psi' e 'density' (arrays 2D, shape (resolution, resolution)),
        'plane', 'range_max', 'xlabel', 'ylabel'
    """
    from orbitals.wavefunction import HydrogenWavefunction

    if range_max is None:
        # O raio característico hidrogenoide escala como n²·a0/Z_eff.
        # O fator visual inclui a estrutura radial sem cortar as bordas.
        a0 = 0.529177  # Å
        range_max = (orbital.n ** 2) * a0 * 5.5 / max(orbital.Z_eff, 1e-6)
        range_max = max(range_max, 1.5)  # piso: nunca dar zoom além disso

    X, Y, Z, u, v = _build_plane_grid(plane, range_max, resolution)
    r, theta, phi = cartesian_to_spherical(X, Y, Z)

    wf = HydrogenWavefunction(use_angstrom=True)
    psi = wf.psi(r, theta, phi, orbital.n, orbital.l, orbital.m, orbital.Z_eff)
    psi = np.real(np.asarray(psi))  # o projeto usa a forma real dos orbitais
    density = np.abs(psi) ** 2

    # Detecção de "plano-nó": alguns orbitais têm um nó ANGULAR que coincide
    # exatamente com um dos três planos de corte (ex: p_z é identicamente
    # zero em todo o plano XY, pois lá theta = 90° em cada ponto). Nesses
    # casos psi não é pequeno por a física ser fraca ali — é zero de
    # verdade — mas o float64 não representa cos(90°) como um zero bit a
    # bit, sobra um resíduo de arredondamento (~1e-16 a 1e-18) multiplicando
    # a parte radial. Sem tratar isso, o gráfico normaliza esse ruído como
    # se fosse sinal (mapa de cor na escala 1e-15, contorno de nó "fantasma"
    # que na verdade é só o nó radial ampliado pelo ruído).
    #
    # Comparamos a amplitude de psi nesse corte contra a amplitude da parte
    # RADIAL sozinha (mesmo r, calculada separadamente) — que tem escala
    # física normal independente do plano escolhido. Se a razão for da
    # ordem da precisão de máquina, é ruído, não sinal: o plano é um nó.
    radial_ref = wf.radial_wavefunction(r, orbital.n, orbital.l, orbital.Z_eff)
    radial_scale = np.max(np.abs(radial_ref))
    psi_scale = np.max(np.abs(psi))
    is_node_plane = radial_scale > 0 and (psi_scale / radial_scale) < 1e-6

    xlabel_letter, ylabel_letter = PLANE_LABELS[plane]

    return {
        'u': u, 'v': v,
        'psi': psi, 'density': density,
        'plane': plane, 'range_max': range_max,
        'xlabel': f"{xlabel_letter} (Å)", 'ylabel': f"{ylabel_letter} (Å)",
        'is_node_plane': is_node_plane,
    }


def plot_orbital_slice(orbital, plane: str = 'xz', range_max: float = None,
                        resolution: int = 400, show_nodes: bool = True):
    """
    Gera a figura matplotlib com dois painéis lado a lado:
      - psi (sinal/fase), colormap divergente, com contorno preto marcando
        onde psi = 0 (os nós — exatamente o que costuma aparecer nos
        diagramas de livro-texto)
      - |psi|^2 (densidade de probabilidade), colormap sequencial com
        correção gama (PowerNorm) para preservar lóbulos externos mais fracos

    Retorna a Figure sem exibi-la, permitindo uso em canvas Qt, arquivo PNG
    ou janela interativa.
    """
    data = compute_orbital_slice(orbital, plane=plane, range_max=range_max,
                                  resolution=resolution)
    u, v = data['u'], data['v']
    psi, density = data['psi'], data['density']
    extent = [u.min(), u.max(), v.min(), v.max()]

    label = quantum_label(orbital.n, orbital.l, orbital.m)

    fig, (ax_psi, ax_dens) = plt.subplots(1, 2, figsize=(11, 5.2))
    fig.suptitle(f"{label} — corte no plano {plane.upper()}  (Z_eff = {orbital.Z_eff:.2f})")

    # Os dois painéis usam o mesmo alcance calculado a partir de n e Z_eff.
    for ax, title in ((ax_psi, "ψ (amplitude e fase)"),
                      (ax_dens, "|ψ|² (densidade de probabilidade)")):
        ax.set_facecolor('0.95')
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_title(title)
        ax.set_xlabel(data['xlabel'])
        ax.set_ylabel(data['ylabel'])
        ax.set_aspect('equal')

    # --- Caso especial: plano-nó (ψ ≡ 0 em toda a extensão) ---
    # Em vez de plotar o resíduo de arredondamento como se fosse dado (o que
    # produz um mapa de cor sem sentido e um "nó fantasma"), avisamos
    # explicitamente e sugerimos os outros planos, onde o orbital de fato
    # tem lóbulos visíveis.
    if data.get('is_node_plane', False):
        outros = [p.upper() for p in PLANE_LABELS if p != plane]
        msg = (
            f"ψ ≈ 0 em todo o plano {plane.upper()}\n\n"
            f"Esse plano é um nó angular\n"
            f"do orbital {label} — não é erro\n"
            f"de cálculo, a função de onda\n"
            f"realmente se anula aqui.\n\n"
            f"Tente o plano {' ou '.join(outros)}."
        )
        for ax in (ax_psi, ax_dens):
            ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha='center', va='center',
                    fontsize=8, color='0.3')
        fig.subplots_adjust(left=0.18, right=0.94, bottom=0.17, top=0.82)
        return fig

    # --- Painel 1: psi, com sinal ---
    # Percentil em vez de np.max: evita que um pico isolado bem no centro
    # (comum em orbitais s) esmague o contraste do resto do mapa de cor.
    psi_abs_max = np.percentile(np.abs(psi), 99.0)
    if psi_abs_max < 1e-15:
        psi_abs_max = np.max(np.abs(psi)) + 1e-15

    im1 = ax_psi.imshow(
        psi, extent=extent, origin='lower', cmap='RdBu_r',
        vmin=-psi_abs_max, vmax=psi_abs_max, interpolation='bilinear'
    )

    # Mascara resíduos numéricos antes de traçar ψ=0 para que apenas nós
    # fisicamente significativos produzam contornos.
    if show_nodes and psi.min() < 0 < psi.max():
        psi_true_max = np.max(np.abs(psi))
        noise_floor = psi_true_max * 1e-6
        psi_for_contour = np.ma.masked_where(np.abs(psi) < noise_floor, psi)
        if psi_for_contour.count() > 0 and psi_for_contour.min() < 0 < psi_for_contour.max():
            ax_psi.contour(u, v, psi_for_contour, levels=[0.0],
                            colors='black', linewidths=0.8)

    ax_psi.axhline(0, color='0.6', linewidth=0.5, zorder=0)
    ax_psi.axvline(0, color='0.6', linewidth=0.5, zorder=0)
    fig.colorbar(im1, ax=ax_psi, fraction=0.046, pad=0.04)

    # --- Painel 2: |psi|^2 ---
    dens_max = np.percentile(density, 99.5)
    if dens_max < 1e-15:
        dens_max = np.max(density) + 1e-15

    im2 = ax_dens.imshow(
        density, extent=extent, origin='lower', cmap='inferno',
        norm=PowerNorm(gamma=0.5, vmin=0, vmax=dens_max), interpolation='bilinear'
    )
    fig.colorbar(im2, ax=ax_dens, fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig

def plot_orbital_slice_panel(orbital, plane: str = 'xz', range_max: float = None,
                             resolution: int = 400, panel: str = 'amplitude',
                             show_nodes: bool = True):
    """
    Gera uma figura matplotlib com apenas um dos painéis do corte 2D.

    panel:
      - 'amplitude'   -> psi com sinal/fase
      - 'probability' -> |psi|^2
    """
    if panel not in {'amplitude', 'probability'}:
        raise ValueError("panel deve ser 'amplitude' ou 'probability'")

    data = compute_orbital_slice(orbital, plane=plane, range_max=range_max,
                                  resolution=resolution)
    u, v = data['u'], data['v']
    psi, density = data['psi'], data['density']
    extent = [u.min(), u.max(), v.min(), v.max()]

    label = quantum_label(orbital.n, orbital.l, orbital.m)
    panel_title = "Amplitude ψ" if panel == 'amplitude' else "Probabilidade |ψ|²"

    fig, ax = plt.subplots(1, 1, figsize=(4.4, 3.4))

    ax.set_facecolor('0.95')
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_title(panel_title, fontsize=7.5, pad=1)
    ax.set_xlabel(data['xlabel'].split()[0], fontsize=6.8, labelpad=1)
    ax.set_ylabel(data['ylabel'].split()[0], fontsize=6.8, labelpad=1)
    ax.tick_params(axis='both', labelsize=6.4, pad=1)
    ax.set_aspect('equal')

    if data.get('is_node_plane', False):
        outros = [p.upper() for p in PLANE_LABELS if p != plane]
        msg = (
            f"ψ ≈ 0 em todo o plano {plane.upper()}\n\n"
            f"Esse plano é um nó angular\n"
            f"do orbital {label} — não é erro\n"
            f"de cálculo, a função de onda\n"
            f"realmente se anula aqui.\n\n"
            f"Tente o plano {' ou '.join(outros)}."
        )
        ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha='center', va='center',
                fontsize=7, color='0.3')
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.90)
        return fig

    if panel == 'amplitude':
        psi_abs_max = np.percentile(np.abs(psi), 99.0)
        if psi_abs_max < 1e-15:
            psi_abs_max = np.max(np.abs(psi)) + 1e-15

        im = ax.imshow(
            psi, extent=extent, origin='lower', cmap='RdBu_r',
            vmin=-psi_abs_max, vmax=psi_abs_max, interpolation='bilinear'
        )

        if show_nodes and psi.min() < 0 < psi.max():
            psi_true_max = np.max(np.abs(psi))
            noise_floor = psi_true_max * 1e-6
            psi_for_contour = np.ma.masked_where(np.abs(psi) < noise_floor, psi)
            if psi_for_contour.count() > 0 and psi_for_contour.min() < 0 < psi_for_contour.max():
                ax.contour(u, v, psi_for_contour, levels=[0.0],
                           colors='black', linewidths=0.8)

        ax.axhline(0, color='0.6', linewidth=0.5, zorder=0)
        ax.axvline(0, color='0.6', linewidth=0.5, zorder=0)
    else:
        dens_max = np.percentile(density, 99.5)
        if dens_max < 1e-15:
            dens_max = np.max(density) + 1e-15

        im = ax.imshow(
            density, extent=extent, origin='lower', cmap='inferno',
            norm=PowerNorm(gamma=0.5, vmin=0, vmax=dens_max), interpolation='bilinear'
        )

    fig.tight_layout()
    return fig

def save_orbital_slice(orbital, filepath: str, plane: str = 'xz',
                        range_max: float = None, resolution: int = 400):
    """Salva o corte 2D diretamente em PNG."""
    fig = plot_orbital_slice(orbital, plane=plane, range_max=range_max,
                              resolution=resolution)
    fig.savefig(filepath, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    return filepath

