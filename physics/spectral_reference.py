"""Comparações espectroscópicas compatíveis com o modelo do simulador."""

from dataclasses import dataclass

from physics.energy_levels import TransitionResult


NIST_HYDROGEN_LEVELS_CM1 = {
    1: 0.0000,
    2: 82259.1109,
    3: 97492.3087,
    4: 102823.9002,
    5: 105291.6587,
}

NIST_HYDROGEN_LEVELS_URL = (
    "https://physics.nist.gov/PhysRefData/Handbook/Tables/"
    "hydrogentable5.htm"
)

SERIES_NAMES = {
    1: "Lyman",
    2: "Balmer",
    3: "Paschen",
    4: "Brackett",
}


@dataclass(frozen=True)
class SpectralComparison:
    series: str
    lower_n: int
    upper_n: int
    calculated_vacuum_nm: float
    reference_vacuum_nm: float
    absolute_difference_nm: float
    relative_error_percent: float
    source_url: str


def compare_with_nist_hydrogen(
        transition: TransitionResult, atomic_number: int,
        electron_count: int,
):
    """Compara uma transição E1 de H I com níveis médios avaliados pelo NIST."""
    if atomic_number != 1 or electron_count != 1:
        return None
    if not transition.electric_dipole_allowed:
        return None

    initial_n = transition.initial[0]
    final_n = transition.final[0]
    lower_n, upper_n = sorted((initial_n, final_n))
    if lower_n == upper_n:
        return None
    if lower_n not in NIST_HYDROGEN_LEVELS_CM1:
        return None
    if upper_n not in NIST_HYDROGEN_LEVELS_CM1:
        return None

    wavenumber = abs(
        NIST_HYDROGEN_LEVELS_CM1[upper_n]
        - NIST_HYDROGEN_LEVELS_CM1[lower_n]
    )
    if wavenumber <= 0:
        return None

    reference_nm = 1.0e7 / wavenumber
    calculated_nm = transition.zero_field_wavelength_nm
    difference_nm = abs(calculated_nm - reference_nm)
    return SpectralComparison(
        series=SERIES_NAMES.get(lower_n, f"n={lower_n}"),
        lower_n=lower_n,
        upper_n=upper_n,
        calculated_vacuum_nm=calculated_nm,
        reference_vacuum_nm=reference_nm,
        absolute_difference_nm=difference_nm,
        relative_error_percent=100.0 * difference_nm / reference_nm,
        source_url=NIST_HYDROGEN_LEVELS_URL,
    )


def nist_comparison_scope_message(
        transition: TransitionResult, atomic_number: int,
        electron_count: int,
) -> str:
    """Explica por que uma comparação não é oferecida para a seleção atual."""
    if atomic_number != 1 or electron_count != 1:
        return "Disponível nesta etapa apenas para o hidrogênio neutro (H I)."
    if not transition.electric_dipole_allowed:
        return "A referência agrupada é exibida somente para transições E1 permitidas."
    maximum_n = max(transition.initial[0], transition.final[0])
    if maximum_n > max(NIST_HYDROGEN_LEVELS_CM1):
        return "Os níveis médios incorporados do NIST cobrem H I até n=5."
    return "Não há uma correspondência NIST inequívoca para esta seleção."
