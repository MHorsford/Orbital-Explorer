"""
atom/atom.py

Classe Atom — representa um átomo completo.
Combina núcleo (Nucleus) com orbitais eletrônicos (lista de Orbital).
Aplica as regras de preenchimento: Aufbau, Hund e Princípio de Pauli.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Dict, Tuple
import numpy as np

from nucleus.nucleus import Nucleus
from orbitals.orbital import Orbital
from orbitals.orbital_types import get_orbital_type
from physics.screening import (
    build_ground_state_config,
    has_ground_state_exception,
    slater_effective_charge,
    get_orbital_sequence,
)


class Atom:
    """
    Representa um átomo completo com núcleo e nuvem eletrônica.

    Atributos:
        nucleus       : objeto Nucleus (prótons + nêutrons)
        orbitals      : lista de objetos Orbital preenchidos
        Z             : número atômico (prótons)
        N             : número de elétrons (igual a Z para átomo neutro)
        position      : posição do átomo no espaço
    """

    def __init__(self, Z: int = 1):
        """
        Cria um átomo com número atômico Z.

        Parâmetros:
            Z : número atômico (número de prótons)
        """
        self.nucleus = Nucleus(Z=Z, N=0)  # Por enquanto, isótopo natural (padrão)
        self.orbitals: List[Orbital] = []
        self.position = np.array([0.0, 0.0, 0.0])

        # Constrói e preenche os orbitais
        self._build_orbitals(Z)

    def _build_orbitals(self, Z: int) -> None:
        """
        Cria e preenche orbitais subnível por subnível.
        
        Algoritmo:
            Para cada subnível (n, l) na ordem de Aufbau:
                1. Cria todos os orbitais do subnível (m_l = -l até +l)
                2. Preenche até capacidade (máx 2 elétrons por orbital)
                3. Aplica Hund (1 por orbital antes do 2º)
                4. Aplica Pauli (máx 2 por orbital)
        """
        self.orbitals = []
        ground_state_config = build_ground_state_config(Z)

        for n, l in get_orbital_sequence():
            electrons_in_subshell = ground_state_config.get((n, l), 0)
            if electrons_in_subshell <= 0:
                continue
            
            # Calcula Z_eff uma vez para todo o subnível
            Z_eff = slater_effective_charge(Z, n, l)
            
            # Cria todos os orbitais deste subnível
            subshell_orbitals = []
            for m_l in range(-l, l + 1):
                orbital = Orbital(n=n, l=l, m=m_l, electrons=0, Z_eff=Z_eff)
                subshell_orbitals.append(orbital)
                self.orbitals.append(orbital)
            
            # Preenche o subnível: Hund (↑ em cada orbital) e depois Pauli (↓).
            # PASSO A: Hund — 1 elétron em cada orbital (spin up)
            for orbital in subshell_orbitals:
                if electrons_in_subshell <= 0:
                    break
                orbital.add_electron(spin=Orbital.SPIN_UP)
                electrons_in_subshell -= 1
            
            # PASSO B: Pauli — segundo elétron em cada orbital (spin down)
            for orbital in subshell_orbitals:
                if electrons_in_subshell <= 0:
                    break
                if orbital.electrons == 1:
                    orbital.add_electron(spin=Orbital.SPIN_DOWN)
                    electrons_in_subshell -= 1

    @property
    def Z(self) -> int:
        """Número atômico (número de prótons)"""
        return self.nucleus.Z

    @property
    def N_electrons(self) -> int:
        """Número total de elétrons"""
        return sum(orb.electrons for orb in self.orbitals)

    @property
    def is_neutral(self) -> bool:
        """Retorna True se o átomo é neutro (N_elétrons == Z)"""
        return self.N_electrons == self.Z

    def get_element_symbol(self) -> str:
        """Retorna o símbolo do elemento (ex: 'H', 'C')"""
        return self.nucleus.get_element_symbol()

    def get_element_name(self) -> str:
        """Retorna o nome do elemento (ex: 'Hydrogen')"""
        return self.nucleus.get_element_name()

    def get_electron_config(self) -> str:
        """
        Retorna a configuração eletrônica atual mantendo RIGOROSAMENTE a ordem de Aufbau.
        Exemplo: "1s² 2s² 2p⁶ 3s² 3p⁶ 4s¹"
        """
        config_dict: Dict[Tuple[int, int], int] = {}

        # Agrupa elétrons por (n, l)
        for orbital in self.orbitals:
            if orbital.electrons > 0:
                key = (orbital.n, orbital.l)
                if key not in config_dict:
                    config_dict[key] = 0
                config_dict[key] += orbital.electrons

        # Preserva a ordem energética de Aufbau na representação textual.
        config_str = ""
        for n, l in get_orbital_sequence():
            if (n, l) in config_dict:
                electron_count = config_dict[(n, l)]
                l_letter = get_orbital_type(l).letter
                config_str += f"{n}{l_letter}{electron_count} "

        return config_str.strip()

    def get_valence_electrons(self) -> int:
        """
        Retorna o número correto de elétrons de valência, suportando elementos 
        dos blocos s, p e os metais de transição do bloco d.
        """
        if not self.orbitals or not any(orb.electrons > 0 for orb in self.orbitals):
            return 0

        # Encontra o maior n preenchido
        max_n = max(orb.n for orb in self.orbitals if orb.electrons > 0)
        
        # Filtra os subníveis ativos
        active_subshells = set((orb.n, orb.l) for orb in self.orbitals if orb.electrons > 0)
        
        valence_electrons = 0
        for n, l in active_subshells:
            # Regra do bloco s e p: elétrons na camada mais externa (n == max_n)
            if n == max_n:
                valence_electrons += sum(orb.electrons for orb in self.orbitals if orb.n == n and orb.l == l)
            # Regra do bloco d (metais de transição): subnível d incompleto da camada (max_n - 1)
            elif n == max_n - 1 and l == 2:
                total_subshell_e = sum(orb.electrons for orb in self.orbitals if orb.n == n and orb.l == l)
                if total_subshell_e < 10:  # Só conta como valência se não estiver totalmente cheio
                    valence_electrons += total_subshell_e
                    
        return valence_electrons

    def get_orbital_filling_order(self) -> str:
        """
        Retorna uma representação visual limpa condensada por subnível.
        Exemplo: 1s²↓↑ 2s²↓↑ 2p²↓↓
        """
        subshell_map = {}
        # Agrupa os spins diretamente por subnível cartesiano completo
        for orbital in self.orbitals:
            if orbital.electrons == 0:
                continue
            key = (orbital.n, orbital.l)
            if key not in subshell_map:
                subshell_map[key] = []
            
            subshell_map[key].append(orbital.spin_symbols)

        # Cada subnível aparece uma única vez na ordem de Aufbau.
        config_list = []
        for n, l in get_orbital_sequence():
            if (n, l) in subshell_map:
                label = f"{n}{get_orbital_type(l).letter}"
                spins = "".join(subshell_map[(n, l)])
                total_e = sum(len(spins) for spins in subshell_map[(n, l)])
                config_list.append(f"{label}^{total_e}{spins}")
                
        return " ".join(config_list)

    def get_subshell_occupancy(self) -> Dict[Tuple[int, int], int]:
        """Agrupa a quantidade atual de elétrons por subnível (n, l)."""
        occupancy: Dict[Tuple[int, int], int] = {}
        for orbital in self.orbitals:
            key = (orbital.n, orbital.l)
            occupancy[key] = occupancy.get(key, 0) + orbital.electrons
        return {key: value for key, value in occupancy.items() if value > 0}

    def validate_filling_rules(self) -> Dict[str, bool]:
        """Verifica Aufbau, Hund e Pauli na configuração eletrônica atual."""
        expected = build_ground_state_config(self.Z)
        expected = {key: value for key, value in expected.items() if value > 0}
        aufbau_ok = self.get_subshell_occupancy() == expected

        pauli_ok = all(
            orbital.electrons <= orbital.MAX_ELECTRONS_PER_ORBITAL
            and len(set(orbital.electron_spins)) == orbital.electrons
            and all(spin in (Orbital.SPIN_UP, Orbital.SPIN_DOWN)
                    for spin in orbital.electron_spins)
            for orbital in self.orbitals
        )

        hund_ok = True
        grouped: Dict[Tuple[int, int], List[Orbital]] = {}
        for orbital in self.orbitals:
            grouped.setdefault((orbital.n, orbital.l), []).append(orbital)
        for orbitals in grouped.values():
            occupancies = [orbital.electrons for orbital in orbitals]
            # Não pode haver emparelhamento enquanto existir orbital vazio.
            if 2 in occupancies and 0 in occupancies:
                hund_ok = False
                break
            # Elétrons desemparelhados do mesmo subnível devem ser paralelos.
            single_spins = [orbital.electron_spins[0]
                            for orbital in orbitals if orbital.electrons == 1]
            if len(set(single_spins)) > 1:
                hund_ok = False
                break

        return {"aufbau": aufbau_ok, "hund": hund_ok, "pauli": pauli_ok}

    def get_orbital_diagram(self) -> str:
        """Retorna um diagrama de caixas e setas para uso didático na UI."""
        superscript = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        grouped: Dict[Tuple[int, int], List[Orbital]] = {}
        for orbital in self.orbitals:
            grouped.setdefault((orbital.n, orbital.l), []).append(orbital)

        lines = []
        for n, l in get_orbital_sequence():
            orbitals = grouped.get((n, l))
            if not orbitals:
                continue
            total = sum(orbital.electrons for orbital in orbitals)
            label = f"{n}{get_orbital_type(l).letter}{str(total).translate(superscript)}"
            boxes = " ".join(f"[{orbital.spin_symbols:<2}]" for orbital in orbitals)
            magnetic = " ".join(f"{orbital.m:+d}".center(4) for orbital in orbitals)
            lines.append(f"{label:<5} {boxes}")
            if len(orbitals) > 1:
                lines.append(f"{'mₗ':<5} {magnetic}")
        return "\n".join(lines)

    @property
    def has_configuration_exception(self) -> bool:
        """Indica uma promoção eletrônica na configuração fundamental."""
        return has_ground_state_exception(self.Z)

    def get_orbital_by_quantum_numbers(self, n: int, l: int, m: int) -> Orbital:
        """
        Procura um orbital específico pelos números quânticos (n, l, m).

        Retorna None se não encontrar.
        """
        for orbital in self.orbitals:
            if orbital.n == n and orbital.l == l and orbital.m == m:
                return orbital
        return None

    def list_orbitals(self) -> str:
        """Retorna uma lista legível de todos os orbitais e seus preenchimentos"""
        lines = []
        for i, orb in enumerate(self.orbitals):
            lines.append(
                f"  [{i:2d}] {orb.n}{get_orbital_type(orb.l).letter:1s}(m={orb.m:+d}) — "
                f"{orb.electrons}/2 e⁻ | Z_eff={orb.Z_eff:.2f}"
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        symbol = self.get_element_symbol()
        config = self.get_electron_config()
        return (
            f"{symbol} (Z={self.Z}) | "
            f"Elétrons: {self.N_electrons} | "
            f"Config: {config}"
        )

    def __repr__(self) -> str:
        return self.__str__()

