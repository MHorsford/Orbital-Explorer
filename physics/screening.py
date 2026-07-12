"""
physics/screening.py

Cálculo da carga nuclear efetiva (Z_eff) usando as Regras de Slater.

Para átomos com múltiplos elétrons, cada elétron "vê" uma carga nuclear
reduzida devido ao screening (blindagem) dos outros elétrons.

Esta implementação usa as regras de Slater originais (1930) para calcular
a constante de blindagem σ, e então Z_eff = Z - σ.

Referência: Slater, J. C. (1930). Phys. Rev. 36(1), 57-64.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Dict, Tuple, List


# SEQUÊNCIA DE AUFBAU (ordem de preenchimento)

def get_orbital_sequence() -> List[Tuple[int, int]]:
    """
    Retorna a sequência de orbitais no preenchimento (regra de Aufbau).

    Ordem: 1s → 2s → 2p → 3s → 3p → 4s → 3d → 4p → 5s → 4d → ...
    """
    return [
        (1, 0),              # 1s (2 elétrons)
        (2, 0),              # 2s (2)
        (2, 1),              # 2p (6)
        (3, 0),              # 3s (2)
        (3, 1),              # 3p (6)
        (4, 0),              # 4s (2)
        (3, 2),              # 3d (10)
        (4, 1),              # 4p (6)
        (5, 0),              # 5s (2)
        (4, 2),              # 4d (10)
        (5, 1),              # 5p (6)
        (6, 0),              # 6s (2)
        (4, 3),              # 4f (14)
        (5, 2),              # 5d (10)
        (6, 1),              # 6p (6)
        (7, 0),              # 7s (2)
        (5, 3),              # 5f (14)
        (6, 2),              # 6d (10)
        (7, 1),              # 7p (6)
    ]


def max_electrons_in_subshell(l: int) -> int:
    """Retorna o número máximo de elétrons num subnível l."""
    return 2 * (2 * l + 1)


# Configurações fundamentais que fogem da aplicação mecânica da regra de
# Madelung/Aufbau. Os valores abaixo substituem apenas os subníveis envolvidos
# na promoção eletrônica; os demais continuam sendo preenchidos normalmente.
# Isso mantém a sequência didática de Aufbau sem apresentar configurações
# sabidamente incorretas para estes elementos.
# Referência das configurações fundamentais: NIST Atomic Reference Data,
# https://www.nist.gov/pml/atomic-reference-data-electronic-structure-calculations/
# atomic-reference-data-electronic-8
GROUND_STATE_EXCEPTIONS: Dict[int, Dict[Tuple[int, int], int]] = {
    24: {(4, 0): 1, (3, 2): 5},                    # Cr
    29: {(4, 0): 1, (3, 2): 10},                   # Cu
    41: {(5, 0): 1, (4, 2): 4},                    # Nb
    42: {(5, 0): 1, (4, 2): 5},                    # Mo
    44: {(5, 0): 1, (4, 2): 7},                    # Ru
    45: {(5, 0): 1, (4, 2): 8},                    # Rh
    46: {(5, 0): 0, (4, 2): 10},                   # Pd
    47: {(5, 0): 1, (4, 2): 10},                   # Ag
    57: {(4, 3): 0, (5, 2): 1},                    # La
    58: {(4, 3): 1, (5, 2): 1},                    # Ce
    64: {(4, 3): 7, (5, 2): 1},                    # Gd
    78: {(6, 0): 1, (5, 2): 9},                    # Pt
    79: {(6, 0): 1, (5, 2): 10},                   # Au
    89: {(5, 3): 0, (6, 2): 1},                    # Ac
    90: {(5, 3): 0, (6, 2): 2},                    # Th
    91: {(5, 3): 2, (6, 2): 1},                    # Pa
    92: {(5, 3): 3, (6, 2): 1},                    # U
    93: {(5, 3): 4, (6, 2): 1},                    # Np
    96: {(5, 3): 7, (6, 2): 1},                    # Cm
    103: {(6, 2): 0, (7, 1): 1},                   # Lr
}


def has_ground_state_exception(Z: int) -> bool:
    """Indica se o elemento possui uma exceção energética ao Aufbau simples."""
    return Z in GROUND_STATE_EXCEPTIONS


# CONSTRUÇÃO DA CONFIGURAÇÃO ELETRÔNICA (para um dado Z)

def build_ground_state_config(Z: int) -> Dict[Tuple[int, int], int]:
    """
    Constrói a configuração fundamental do átomo com número atômico Z,
    partindo de Aufbau e aplicando as promoções eletrônicas conhecidas.

    Retorna um dicionário: {(n, l): número_de_elétrons}
    Exemplo: Z=6 (Carbono) → {(1,0):2, (2,0):2, (2,1):2}
    """
    config = {}
    electrons_left = Z

    for n, l in get_orbital_sequence():
        if electrons_left <= 0:
            break
        max_e = max_electrons_in_subshell(l)
        fill = min(electrons_left, max_e)
        if fill > 0:
            config[(n, l)] = fill
            electrons_left -= fill

    # Se ainda sobrou elétrons (Z > 118), coloca no próximo orbital (improvável)
    if electrons_left > 0:
        # Fallback: coloca no último subnível disponível
        last_key = list(config.keys())[-1]
        config[last_key] += electrons_left

    for subshell, electron_count in GROUND_STATE_EXCEPTIONS.get(Z, {}).items():
        if electron_count:
            config[subshell] = electron_count
        else:
            config.pop(subshell, None)

    return config


# Alias público para a construção da configuração fundamental.
_build_aufbau_config = build_ground_state_config


# CÁLCULO DE Z_EFF (REGRAS DE SLATER)


def slater_effective_charge(Z: int, n: int, l: int) -> float:
    """
    Calcula a carga nuclear efetiva Z_eff para um elétron no orbital (n, l)
    de um átomo com número atômico Z seguindo rigorosamente os grupos de Slater.
    """
    if Z <= 0 or n <= 0:
        return 1.0

    # 1. Obter a configuração eletrônica completa
    full_config = build_ground_state_config(Z)

    # 2. Remover 1 elétron do subnível alvo
    target_key = (n, l)
    target_count = full_config.get(target_key, 0)

    if target_count <= 0:
        return float(Z)

    config = full_config.copy()
    config[target_key] = target_count - 1

    # Mapeamento de precedência dos grupos de Slater:
    # Um grupo A está "à esquerda" (internamente) de B se:
    # (n_A < n_B) OU (n_A == n_B E l_A é s/p enquanto l_B é d/f) OU (n_A == n_B E l_A == 2 E l_B == 3)
    def is_strictly_internal(ni, li, nt, lt):
        if ni < nt:
            # Para alvos s/p, se ni == nt - 1, a regra impõe 0.85, tratado separadamente.
            # Mas se li >= 2 (orbitais d ou f), eles blindam 1.00 para elétrons s/p externos.
            if (lt == 0 or lt == 1) and ni == nt - 1 and (li == 0 or li == 1):
                return False # Será pego pela regra do 0.85
            return True
        if ni == nt:
            # ns,np blindam totalmente (1.00) um orbital nd ou nf na mesma camada
            if (li == 0 or li == 1) and lt >= 2:
                return True
            # nd blinda totalmente (1.00) um orbital nf na mesma camada
            if li == 2 and lt == 3:
                return True
        return False

    sigma = 0.0

    for (ni, li), count in config.items():
        if count <= 0:
            continue

        # 1. Mesmíssimo grupo de Slater (ns, np compartilham o grupo; nd e nf são isolados)
        if ni == n and ((l <= 1 and li <= 1) or (l == li)):
            if n == 1:
                sigma += count * 0.30  # Exceção histórica para o H e He (1s)
            else:
                sigma += count * 0.35
            continue

        # 2. Elétrons em grupos situados "à direita" (mais externos) possuem efeito nulo (0.0)
        if ni > n or (ni == n and li > l and l >= 2) or (ni == n and li >= 2 and l <= 1):
            continue

        # 3. Regras de Blindagem baseadas na natureza do orbital alvo
        if l == 0 or l == 1:  # Alvo é um elétron s ou p
            if ni == n - 1 and (li == 0 or li == 1):
                sigma += count * 0.85  # Apenas elétrons s/p da camada imediatamente inferior
            elif is_strictly_internal(ni, li, n, l):
                sigma += count * 1.00  # d/f de n-1 ou qualquer elétron de n-2 para trás
            else:
                sigma += count * 1.00  # Fallback seguro para camadas muito profundas
        else:  # Alvo é um elétron d ou f
            if is_strictly_internal(ni, li, n, l):
                sigma += count * 1.00

    # 4. Calcular Z_eff final
    Z_eff = Z - sigma
    return max(0.1, Z_eff)


# FUNÇÕES AUXILIARES DA API PÚBLICA

def slater_group(n: int, l: int) -> int:
    """Retorna o grupo de Slater: n-1 para n>=2 e 0 para n=1."""
    return max(0, n - 1)


