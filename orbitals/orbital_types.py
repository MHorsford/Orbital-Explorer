"""
orbitals/orbital_types.py

Configurações e metadados dos tipos de orbitais (s, p, d, f, g, h, i).
Este arquivo serve como referência central para características 
comuns de cada tipo de orbital.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class OrbitalType:
    """Representa as características de um tipo de orbital (s, p, d, f, ...)"""
    letter: str
    max_electrons: int
    degeneracy: int          # quantidade de orbitais 
    default_color: Tuple[float, float, float]
    description: str


# Configuração central dos tipos de orbitais
ORBITAL_TYPES: Dict[int, OrbitalType] = {
    0: OrbitalType(  # s
        letter="s",
        max_electrons=2,
        degeneracy=1,
        default_color=(0.2, 0.8, 1.0),      # Azul claro
        description="Esférico, simétrico em todas as direções"
    ),
    1: OrbitalType(  # p
        letter="p",
        max_electrons=6,
        degeneracy=3,
        default_color=(1.0, 0.55, 0.0),     # Laranja
        description="Formato de haltere (dumbbell) em 3 orientações"
    ),
    2: OrbitalType(  # d
        letter="d",
        max_electrons=10,
        degeneracy=5,
        default_color=(0.8, 0.2, 1.0),      # Roxo
        description="Formas complexas com 4 ou 5 lobos"
    ),
    3: OrbitalType(  # f
        letter="f",
        max_electrons=14,
        degeneracy=7,
        default_color=(0.2, 1.0, 0.4),      # Verde
        description="Orbitais muito complexos com múltiplos lobos"
    ),
    # Tipos de momento angular superior.
    4: OrbitalType(  # g
        letter="g",
        max_electrons=18,
        degeneracy=9,
        default_color=(1.0, 0.8, 0.0),      # Amarelo dourado
        description="Orbitais com 8 lobos ou formas anelares complexas"
    ),
    5: OrbitalType(  # h
        letter="h",
        max_electrons=22,
        degeneracy=11,
        default_color=(1.0, 0.5, 0.0),      # Laranja escuro / terracota
        description="Orbitais de altíssima energia com geometrias exóticas (h)"
    ),
    6: OrbitalType(  # i
        letter="i",
        max_electrons=26,
        degeneracy=13,
        default_color=(0.0, 1.0, 1.0),      # Ciano
        description="Orbitais superiores (i) — raramente ocupados na prática"
    ),
}


def get_orbital_type(l: int) -> OrbitalType:
    """Retorna as informações do tipo de orbital baseado no número quântico l"""
    # Se l for maior que o máximo definido, retorna o último (i) como fallback
    if l in ORBITAL_TYPES:
        return ORBITAL_TYPES[l]
    else:
        # Fallback seguro: retorna o tipo 'i' (l=6) com valores genéricos
        print(f"⚠ Aviso: l={l} não definido, usando fallback genérico.")
        return OrbitalType(
            letter="?",
            max_electrons=2*(2*l+1),
            degeneracy=2*l+1,
            default_color=(0.5, 0.5, 0.5),
            description=f"Orbital desconhecido (l={l})"
        )


def get_orbital_name(n: int, l: int) -> str:
    """Retorna o nome padrão do orbital (ex: 1s, 2p, 3d, 5g)"""
    letter = get_orbital_type(l).letter
    return f"{n}{letter}"


def get_max_electrons(l: int) -> int:
    """Retorna a capacidade máxima de elétrons para o subnível"""
    return get_orbital_type(l).max_electrons

# Alias público para a capacidade do subnível.
max_electrons_in_subshell = get_max_electrons


def get_default_color(l: int) -> Tuple[float, float, float]:
    """Retorna a cor padrão do tipo de orbital"""
    return get_orbital_type(l).default_color


def list_all_orbital_types() -> None:
    """Imprime os tipos de orbitais disponíveis para diagnóstico."""
    print("Tipos de Orbitais Disponíveis:")
    for l, ot in ORBITAL_TYPES.items():
        print(f"  l={l} ({ot.letter}) → {ot.max_electrons} elétrons, {ot.degeneracy} orientações")
