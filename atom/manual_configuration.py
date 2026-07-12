"""Estado editável usado pelo modo didático de preenchimento eletrônico."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from orbitals.orbital import Orbital
from orbitals.orbital_types import get_orbital_type
from physics.screening import build_ground_state_config, get_orbital_sequence


@dataclass(frozen=True)
class ManualActionResult:
    ok: bool
    message: str
    warnings: Tuple[str, ...] = ()


class ManualElectronConfiguration:
    """Configuração separada do átomo real, inicialmente sem elétrons."""

    def __init__(self, atomic_number: int):
        if atomic_number < 1:
            raise ValueError("O número atômico deve ser positivo")
        self.atomic_number = atomic_number
        self.orbitals: List[Orbital] = []
        self._history: List[Dict[Tuple[int, int, int], Tuple[float, ...]]] = []

        for n, l in get_orbital_sequence():
            for m in range(-l, l + 1):
                self.orbitals.append(Orbital(n=n, l=l, m=m, electrons=0))

    @property
    def electron_count(self) -> int:
        return sum(orbital.electrons for orbital in self.orbitals)

    @property
    def remaining_electrons(self) -> int:
        return self.atomic_number - self.electron_count

    @property
    def is_complete(self) -> bool:
        return self.electron_count == self.atomic_number

    def get_orbital(self, n: int, l: int, m: int):
        for orbital in self.orbitals:
            if (orbital.n, orbital.l, orbital.m) == (n, l, m):
                return orbital
        return None

    def visible_subshells(self, selected=None, extra_levels: int = 0) -> List[Tuple[int, int]]:
        """Subníveis básicos, ocupados e o selecionado pelo usuário."""
        sequence = get_orbital_sequence()
        basic = []
        capacity = 0
        reached_target = False
        for n, l in sequence:
            basic.append((n, l))
            capacity += 2 * (2 * l + 1)
            if reached_target or capacity >= self.atomic_number:
                if reached_target:
                    break
                reached_target = True

        visible = set(basic)
        visible.update(self.subshell_occupancy())
        if selected in sequence:
            visible.add(selected)
        if extra_levels > 0 and visible:
            last_index = max(sequence.index(subshell) for subshell in visible)
            visible.update(sequence[last_index + 1:last_index + 1 + extra_levels])
        return [subshell for subshell in sequence if subshell in visible]

    def _snapshot(self) -> Dict[Tuple[int, int, int], Tuple[float, ...]]:
        return {
            (orbital.n, orbital.l, orbital.m): orbital.electron_spins
            for orbital in self.orbitals if orbital.electrons
        }

    def _restore(self, snapshot) -> None:
        for orbital in self.orbitals:
            orbital.electrons = 0
            for spin in snapshot.get((orbital.n, orbital.l, orbital.m), ()):
                orbital.add_electron(spin=spin)

    def _save_history(self) -> None:
        self._history.append(self._snapshot())

    def add_electron(self, n: int, l: int, m: int, spin: float) -> ManualActionResult:
        orbital = self.get_orbital(n, l, m)
        if orbital is None:
            return ManualActionResult(False, "Combinação de números quânticos inválida.")
        if self.remaining_electrons <= 0:
            occupied = [
                f"{item.n}{get_orbital_type(item.l).letter} (mₗ={item.m:+d})"
                for item in self.orbitals if item.electrons
            ]
            location = ", ".join(occupied) if occupied else "outro orbital"
            return ManualActionResult(
                False,
                "Nenhum elétron restante. Remova um elétron de "
                f"{location} antes de ocupar este orbital.",
            )
        if spin not in (Orbital.SPIN_UP, Orbital.SPIN_DOWN):
            return ManualActionResult(False, "O spin deve ser +½ ou -½.")
        if spin in orbital.electron_spins:
            return ManualActionResult(
                False,
                "Pauli bloqueou a operação: já existe um elétron com esse spin na caixa.",
            )
        if orbital.is_full():
            return ManualActionResult(
                False,
                "Pauli bloqueou a operação: um orbital aceita no máximo dois elétrons.",
            )

        self._save_history()
        orbital.add_electron(spin=spin)
        message = "Elétron adicionado."
        if self.is_complete:
            message += f" {self.result_description()}"
        return ManualActionResult(True, message, self._current_warnings())

    def remove_electron(self, n: int, l: int, m: int) -> ManualActionResult:
        orbital = self.get_orbital(n, l, m)
        if orbital is None or orbital.is_empty():
            return ManualActionResult(False, "O orbital selecionado já está vazio.")
        self._save_history()
        orbital.remove_electron()
        return ManualActionResult(True, "Elétron removido.", self._current_warnings())

    def move_electron(self, source_numbers, target_numbers) -> ManualActionResult:
        """Move um elétron entre orbitais sem alterar o total do elemento."""
        if source_numbers == target_numbers:
            return ManualActionResult(False, "Origem e destino são o mesmo orbital.")
        source = self.get_orbital(*source_numbers)
        target = self.get_orbital(*target_numbers)
        if source is None or target is None:
            return ManualActionResult(False, "Origem ou destino possui números quânticos inválidos.")
        if source.is_empty():
            return ManualActionResult(False, "O orbital de origem não possui elétrons.")
        if target.is_full():
            return ManualActionResult(
                False, "Pauli bloqueou a operação: o orbital de destino já está completo."
            )

        spin = next(
            (value for value in reversed(source.electron_spins)
             if value not in target.electron_spins),
            None,
        )
        if spin is None:
            return ManualActionResult(
                False,
                "Pauli bloqueou a operação: o destino já possui o mesmo spin disponível.",
            )

        self._save_history()
        source.remove_electron(spin=spin)
        target.add_electron(spin=spin)
        source_label = f"{source.n}{get_orbital_type(source.l).letter}"
        target_label = f"{target.n}{get_orbital_type(target.l).letter}"
        message = f"Elétron movido de {source_label} para {target_label}."
        if self.is_complete:
            message += f" {self.result_description()}"
        return ManualActionResult(True, message, self._current_warnings())

    def undo(self) -> bool:
        if not self._history:
            return False
        self._restore(self._history.pop())
        return True

    def reset(self) -> None:
        if self.electron_count:
            self._save_history()
        for orbital in self.orbitals:
            orbital.electrons = 0

    def fill_ground_state(self) -> None:
        self._save_history()
        for orbital in self.orbitals:
            orbital.electrons = 0

        target = build_ground_state_config(self.atomic_number)
        for n, l in get_orbital_sequence():
            remaining = target.get((n, l), 0)
            subshell = [
                orbital for orbital in self.orbitals
                if orbital.n == n and orbital.l == l
            ]
            for spin in (Orbital.SPIN_UP, Orbital.SPIN_DOWN):
                for orbital in subshell:
                    if remaining <= 0:
                        break
                    orbital.add_electron(spin=spin)
                    remaining -= 1

    def subshell_occupancy(self) -> Dict[Tuple[int, int], int]:
        result: Dict[Tuple[int, int], int] = {}
        for orbital in self.orbitals:
            if orbital.electrons:
                key = (orbital.n, orbital.l)
                result[key] = result.get(key, 0) + orbital.electrons
        return result

    @staticmethod
    def _simple_aufbau_config(electron_count: int) -> Dict[Tuple[int, int], int]:
        result = {}
        remaining = electron_count
        for n, l in get_orbital_sequence():
            if remaining <= 0:
                break
            capacity = 2 * (2 * l + 1)
            result[(n, l)] = min(remaining, capacity)
            remaining -= result[(n, l)]
        return result

    def validate_rules(self) -> Dict[str, bool]:
        current = self.subshell_occupancy()
        ground = build_ground_state_config(self.atomic_number)
        ground = {key: value for key, value in ground.items() if value}
        ground_match = self.is_complete and current == ground
        aufbau_ok = ground_match or current == self._simple_aufbau_config(self.electron_count)

        pauli_ok = all(
            orbital.electrons <= 2
            and len(set(orbital.electron_spins)) == orbital.electrons
            for orbital in self.orbitals
        )

        hund_ok = True
        for n, l in get_orbital_sequence():
            subshell = [
                orbital for orbital in self.orbitals
                if orbital.n == n and orbital.l == l
            ]
            occupancies = [orbital.electrons for orbital in subshell]
            if 2 in occupancies and 0 in occupancies:
                hund_ok = False
                break
            singles = [
                orbital.electron_spins[0]
                for orbital in subshell if orbital.electrons == 1
            ]
            if len(set(singles)) > 1:
                hund_ok = False
                break

        return {
            "aufbau": aufbau_ok,
            "hund": hund_ok,
            "pauli": pauli_ok,
            "ground_state": ground_match,
        }

    def _current_warnings(self) -> Tuple[str, ...]:
        checks = self.validate_rules()
        warnings = []
        if not checks["aufbau"]:
            warnings.append(
                "Aufbau: estado excitado permitido — há um subnível de menor "
                "energia disponível."
            )
        if not checks["hund"]:
            warnings.append(
                "Hund: configuração de maior energia — houve emparelhamento "
                "antes de ocupar todas as caixas equivalentes."
            )
        return tuple(warnings)

    def configuration_string(self) -> str:
        occupancy = self.subshell_occupancy()
        parts = []
        for n, l in get_orbital_sequence():
            count = occupancy.get((n, l), 0)
            if count:
                parts.append(f"{n}{get_orbital_type(l).letter}{count}")
        return " ".join(parts) if parts else "vazia"

    def result_description(self) -> str:
        checks = self.validate_rules()
        if not self.is_complete:
            return "Configuração em construção."
        if checks["ground_state"]:
            return "Configuração fundamental correta."
        if checks["pauli"]:
            return "Configuração permitida, mas não fundamental (possível estado excitado)."
        return "Configuração fisicamente proibida pelo princípio de Pauli."
