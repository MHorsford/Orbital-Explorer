"""
nucleus/nucleus.py

Classe Nucleus — gerencia o núcleo atômico (prótons + nêutrons).
Responsável por calcular Z, A e identificar o elemento na tabela periódica.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from pathlib import Path

from particles.proton import Proton
from particles.neutron import Neutron


class Nucleus:
    """
    Representa o núcleo de um átomo.
    
    Atributos:
        protons   : lista de objetos Proton
        neutrons  : lista de objetos Neutron
        Z         : número atômico (número de prótons)
        A         : número de massa (Z + número de nêutrons)
        position  : posição do núcleo no espaço (padrão: origem)
    """

    def __init__(self, Z: int = 1, N: int = 0):
        """
        Parâmetros:
            Z : número atômico (número de prótons)
            N : número de nêutrons
        """
        self.protons = [Proton() for _ in range(Z)]
        self.neutrons = [Neutron() for _ in range(N)]
        self.position = np.array([0.0, 0.0, 0.0])
        
        # Cache de dados do elemento
        self._element_data = None
        self._load_periodic_table()

    @property
    def Z(self) -> int:
        """Número atômico (número de prótons)"""
        return len(self.protons)

    @property
    def N(self) -> int:
        """Número de nêutrons"""
        return len(self.neutrons)

    @property
    def A(self) -> int:
        """Número de massa (Z + N)"""
        return self.Z + self.N

    def _load_periodic_table(self) -> None:
        """Carrega a tabela periódica do CSV"""
        # Tenta encontrar o arquivo em vários caminhos
        possible_paths = [
            Path("data/periodic_table.csv"),
            Path("../data/periodic_table.csv"),
            Path("../../data/periodic_table.csv"),
            Path(__file__).parent.parent / "data" / "periodic_table.csv",
        ]
        
        csv_path = None
        for path in possible_paths:
            if path.exists():
                csv_path = path
                break
        
        if csv_path:
            try:
                self.periodic_table = pd.read_csv(csv_path)
                self.periodic_table.set_index('atomic_number', inplace=True)
            except Exception as e:
                print(f"Aviso: Não conseguiu carregar tabela periódica ({e})")
                self.periodic_table = None
        else:
            print(f"Aviso: periodic_table.csv não encontrado em {possible_paths}")
            self.periodic_table = None

    def get_element_symbol(self) -> str:
        """Retorna o símbolo do elemento (ex: 'H', 'C', 'O')"""
        if self.periodic_table is None or self.Z not in self.periodic_table.index:
            return f"Z{self.Z}"  # fallback
        return self.periodic_table.loc[self.Z, 'symbol']

    def get_element_name(self) -> str:
        """Retorna o nome do elemento (ex: 'Hidrogênio', 'Carbono')"""
        if self.periodic_table is None or self.Z not in self.periodic_table.index:
            return f"Elemento {self.Z}"  # fallback
        return self.periodic_table.loc[self.Z, 'name']

    def get_electron_config(self) -> str:
        """Retorna a configuração eletrônica tabelada (ex: '1s2 2s2 2p2')"""
        if self.periodic_table is None or self.Z not in self.periodic_table.index:
            return "desconhecida"
        return self.periodic_table.loc[self.Z, 'electron_config']

    def add_proton(self) -> None:
        """Adiciona um próton ao núcleo (aumenta Z em 1)"""
        self.protons.append(Proton())

    def remove_proton(self) -> bool:
        """Remove um próton do núcleo (diminui Z em 1)
        
        Retorna True se conseguiu, False se o núcleo já tem Z=0.
        """
        if len(self.protons) > 0:
            self.protons.pop()
            return True
        return False

    def add_neutron(self) -> None:
        """Adiciona um nêutron ao núcleo (aumenta N em 1)"""
        self.neutrons.append(Neutron())

    def remove_neutron(self) -> bool:
        """Remove um nêutron do núcleo (diminui N em 1)
        
        Retorna True se conseguiu, False se não há nêutrons.
        """
        if len(self.neutrons) > 0:
            self.neutrons.pop()
            return True
        return False

    def get_isotope_symbol(self) -> str:
        """Retorna o símbolo do isótopo (ex: 'C-12', 'U-235')"""
        symbol = self.get_element_symbol()
        return f"{symbol}-{self.A}"

    def __str__(self) -> str:
        symbol = self.get_element_symbol()
        name = self.get_element_name()
        return f"{name} ({symbol}) — Z={self.Z} N={self.N} A={self.A}"

    def __repr__(self) -> str:
        return self.__str__()


# TESTE

if __name__ == "__main__":
    print("=" * 70)
    print("TESTE: NÚCLEO ATÔMICO")
    print("=" * 70)

    # Teste 1: Hidrogênio
    print("\n[Teste 1: Hidrogênio (Z=1)]")
    H = Nucleus(Z=1, N=0)
    print(f"  {H}")
    print(f"  Símbolo: {H.get_element_symbol()}")
    print(f"  Isótopo: {H.get_isotope_symbol()}")
    print(f"  Config eletrônica: {H.get_electron_config()}")

    # Teste 2: Carbono-12
    print("\n[Teste 2: Carbono-12 (Z=6, N=6)]")
    C = Nucleus(Z=6, N=6)
    print(f"  {C}")
    print(f"  Símbolo: {C.get_element_symbol()}")
    print(f"  Isótopo: {C.get_isotope_symbol()}")
    print(f"  Config eletrônica: {C.get_electron_config()}")

    # Teste 3: Oxigênio
    print("\n[Teste 3: Oxigênio (Z=8, N=8)]")
    O = Nucleus(Z=8, N=8)
    print(f"  {O}")
    print(f"  Config eletrônica: {O.get_electron_config()}")

    # Teste 4: Adicionar/remover próton
    print("\n[Teste 4: Dinâmica de prótons]")
    test_nuc = Nucleus(Z=1, N=0)
    print(f"  Inicial: {test_nuc.get_element_symbol()} (Z={test_nuc.Z})")
    test_nuc.add_proton()
    print(f"  Depois de add_proton(): {test_nuc.get_element_symbol()} (Z={test_nuc.Z})")
    test_nuc.add_proton()
    print(f"  Depois de outro add_proton(): {test_nuc.get_element_symbol()} (Z={test_nuc.Z})")

    print("\n" + "=" * 70)
