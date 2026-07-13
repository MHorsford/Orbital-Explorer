"""Diagrama didático de níveis e transições eletrônicas."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QLabel, QPushButton,
    QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from orbitals.orbital_types import get_orbital_type
from physics.energy_levels import (
    build_energy_levels, calculate_transition, diagram_subshells,
)
from physics.spectral_reference import (
    compare_with_nist_hydrogen, nist_comparison_scope_message,
)


SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
SPIN_SYMBOL = {0.5: "↑", -0.5: "↓"}


def subshell_label(n, l, occupancy=None):
    label = f"{n}{get_orbital_type(l).letter}"
    if occupancy is not None:
        label += str(occupancy).translate(SUPERSCRIPT)
    return label


def electron_state_label(state, energy_ev=None):
    """Formata (n, l, mₗ, mₛ) de modo compacto para os seletores."""
    n, l, m, spin = state
    label = (
        f"{subshell_label(n, l)}  mₗ={m:+d}  "
        f"{SPIN_SYMBOL[spin]} (mₛ={spin:+.1f})"
    )
    if energy_ev is not None:
        label += f"  E≈{energy_ev:.2f} eV"
    return label


def inferred_occupied_states(configuration):
    """Reconstrói estados por Hund e Pauli quando a UI não fornece orbitais."""
    states = []
    for (n, l), occupancy in configuration.items():
        magnetic_numbers = list(range(-l, l + 1))
        order = (
            [(m, 0.5) for m in magnetic_numbers]
            + [(m, -0.5) for m in magnetic_numbers]
        )
        states.extend((n, l, m, spin) for m, spin in order[:occupancy])
    return states


class EnergyDiagramWidget(QWidget):
    """Apresenta a ordem qualitativa dos subníveis e energias aproximadas."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.levels = []
        self.level_by_key = {}
        self.selected_key = None
        self.selected_m = 0
        self.occupied_states = []
        self.atomic_number = 0
        self.electron_count = 0
        self.species = "espécie atual"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.help_label = QLabel(
            "E1 no modelo de um elétron: Δl=±1, Δmₗ=0 ou ±1, spin "
            "conservado e Δn livre. Os valores E são estimativas "
            "hidrogenoides; Bz aplica o efeito Zeeman linear."
        )
        self.help_label.setObjectName("helpBanner")
        self.help_label.setWordWrap(True)
        self.help_label.setToolTip(
            "O simulador não calcula acoplamento spin–órbita nem J. O campo "
            "altera as energias, mas não deforma a visualização 3D do orbital."
        )
        layout.addWidget(self.help_label)

        self.figure = Figure(figsize=(7.5, 5.2), facecolor="#07111f")
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, stretch=1)

        controls = QGridLayout()
        controls.setHorizontalSpacing(8)
        controls.addWidget(QLabel("Elétron inicial ocupado:"), 0, 0)
        controls.addWidget(QLabel("Estado final:"), 0, 1)
        self.combo_initial = QComboBox()
        self.combo_final = QComboBox()
        for combo in (self.combo_initial, self.combo_final):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(14)
        self.combo_initial.setToolTip(
            "Escolha um elétron existente, incluindo mₗ e mₛ."
        )
        self.combo_final.setToolTip(
            "Escolha o estado de destino completo. O orbital definido nos "
            "controles quânticos aparece como sugestão."
        )
        controls.addWidget(self.combo_initial, 1, 0)
        controls.addWidget(self.combo_final, 1, 1)
        self.btn_transition = QPushButton("Calcular transição")
        self.btn_transition.setProperty("variant", "primary")
        self.btn_transition.setToolTip(
            "Calcula ΔE, frequência e comprimento de onda do fóton pela "
            "aproximação exibida."
        )
        self.btn_transition.clicked.connect(self.calculate_selected_transition)
        controls.addWidget(self.btn_transition, 1, 2)
        self.check_nist = QCheckBox("NIST (H I)")
        self.check_nist.setChecked(True)
        self.check_nist.setToolTip(
            "Compara o resultado sem campo com níveis médios de H I até n=5; "
            "não requer internet."
        )
        controls.addWidget(self.check_nist, 2, 0)
        field_label = QLabel("Campo Bz:")
        field_label.setToolTip("Campo magnético uniforme paralelo ao eixo z.")
        controls.addWidget(field_label, 2, 1, alignment=Qt.AlignRight)
        self.spin_magnetic_field = QDoubleSpinBox()
        self.spin_magnetic_field.setRange(-10.0, 10.0)
        self.spin_magnetic_field.setDecimals(2)
        self.spin_magnetic_field.setSingleStep(0.25)
        self.spin_magnetic_field.setSuffix(" T")
        self.spin_magnetic_field.setKeyboardTracking(False)
        self.spin_magnetic_field.setToolTip(
            "Intensidade e sentido de Bz em teslas. Valores negativos invertem "
            "o sentido do campo. Modelo Zeeman linear sem estrutura fina."
        )
        self.spin_magnetic_field.valueChanged.connect(
            self._on_magnetic_field_changed
        )
        controls.addWidget(self.spin_magnetic_field, 2, 2)
        layout.addLayout(controls)

        self.transition_result = QLabel(
            "Selecione dois subníveis para estimar absorção ou emissão."
        )
        self.transition_result.setObjectName("transitionResult")
        self.transition_result.setWordWrap(True)
        self.transition_result.setTextFormat(Qt.RichText)
        self.transition_result.setOpenExternalLinks(True)
        layout.addWidget(self.transition_result)

    def _on_magnetic_field_changed(self):
        """Recalcula a transição ao alterar Bz, quando há estados válidos."""
        initial_state = self.combo_initial.currentData()
        final_state = self.combo_final.currentData()
        if (
                self.btn_transition.isEnabled()
                and initial_state and final_state
                and tuple(initial_state[:2]) != tuple(final_state[:2])
        ):
            self.calculate_selected_transition()
        else:
            self._draw(self.species)

    def update_state(
            self, Z, electron_count, configuration, selected, species,
            occupied_states=None,
    ):
        selected_key = tuple(selected[:2])
        selected_m = selected[2] if len(selected) > 2 else 0
        selection_changed = (
            selected_key != self.selected_key or selected_m != self.selected_m
        )
        self.selected_key = selected_key
        self.selected_m = selected_m
        self.atomic_number = Z
        self.electron_count = electron_count
        self.species = species
        self.occupied_states = list(
            occupied_states
            if occupied_states is not None
            else inferred_occupied_states(configuration)
        )
        self.levels = build_energy_levels(
            Z, electron_count, configuration,
            subshells=diagram_subshells(
                electron_count, selected_key, configuration=configuration,
            ),
        )
        self.level_by_key = {level.key: level for level in self.levels}
        self._populate_transition_controls()
        if selection_changed:
            destination = self.combo_final.currentData()
            if destination and tuple(destination[:3]) == (*selected_key, selected_m):
                self.transition_result.setText(
                    f"Destino sincronizado com <b>{electron_state_label(destination)}</b>. "
                    "Clique em “Calcular transição”."
                )
            else:
                self.transition_result.setText(
                    "Escolha estados diferentes para estimar absorção ou emissão."
                )
        self._draw(species)

    @staticmethod
    def _index_for_data(combo, value):
        """Localiza dados compostos sem depender da conversão QVariant do Qt."""
        return next(
            (index for index in range(combo.count())
             if combo.itemData(index) == value),
            -1,
        )

    def _populate_transition_controls(self):
        previous_initial = self.combo_initial.currentData()
        previous_final = self.combo_final.currentData()
        self.combo_initial.blockSignals(True)
        self.combo_final.blockSignals(True)
        self.combo_initial.clear()
        self.combo_final.clear()

        available_states = [
            tuple(state) for state in self.occupied_states
            if tuple(state[:2]) in self.level_by_key
        ]
        for state in available_states:
            level = self.level_by_key[tuple(state[:2])]
            self.combo_initial.addItem(
                electron_state_label(state, level.energy_ev), state,
            )
        for level in self.levels:
            for m in range(-level.l, level.l + 1):
                for spin in (0.5, -0.5):
                    state = (level.n, level.l, m, spin)
                    self.combo_final.addItem(
                        electron_state_label(state, level.energy_ev), state,
                    )

        initial_index = self._index_for_data(
            self.combo_initial, previous_initial
        )
        if initial_index >= 0:
            self.combo_initial.setCurrentIndex(initial_index)
        initial_state = self.combo_initial.currentData()
        initial_spin = initial_state[3] if initial_state else 0.5
        selected_state = (
            *self.selected_key, self.selected_m, initial_spin
        )
        preferred_final = selected_state if selected_state != initial_state else previous_final
        final_index = self._index_for_data(self.combo_final, preferred_final)
        if final_index < 0 and available_states:
            final_index = next(
                (i for i in range(self.combo_final.count())
                 if self.combo_final.itemData(i) != initial_state
                 and self.combo_final.itemData(i)[3] == initial_spin),
                0,
            )
        if final_index >= 0:
            self.combo_final.setCurrentIndex(final_index)

        enabled = bool(available_states) and self.combo_final.count() > 1
        self.combo_initial.setEnabled(enabled)
        self.combo_final.setEnabled(enabled)
        self.btn_transition.setEnabled(enabled)
        self.combo_initial.blockSignals(False)
        self.combo_final.blockSignals(False)

    def _draw(self, species, transition=None):
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.set_facecolor("#07111f")
        axis.set_xlim(0, 1)
        axis.set_ylim(-0.7, max(1, len(self.levels) - 0.3))
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

        label_font_size = 9 if len(self.levels) <= 12 else 7
        value_font_size = 8 if len(self.levels) <= 12 else 6.5
        for position, level in enumerate(self.levels):
            is_selected = level.key == self.selected_key
            color = "#ffd166" if is_selected else ("#49d7a5" if level.occupancy else "#60788d")
            linewidth = 4.0 if is_selected else 2.5
            axis.hlines(position, 0.24, 0.72, color=color, linewidth=linewidth)
            axis.text(
                0.21, position, subshell_label(*level.key, level.occupancy),
                ha="right", va="center", color=color, fontsize=label_font_size,
                fontweight="bold" if is_selected else "normal",
            )
            axis.text(
                0.75, position,
                f"E≈{level.energy_ev:.2f} eV   Z_eff={level.z_eff:.2f}",
                ha="left", va="center", color="#c7d8e8", fontsize=value_font_size,
            )

        if transition is not None:
            positions = {level.key: index for index, level in enumerate(self.levels)}
            y0 = positions[transition.initial]
            y1 = positions[transition.final]
            arrow_color = "#62d8ff" if transition.process == "absorção" else "#ff8fb8"
            axis.annotate(
                "", xy=(0.48, y1), xytext=(0.48, y0),
                arrowprops=dict(arrowstyle="->", color=arrow_color, lw=2.5),
            )

        axis.set_title(
            (
                f"Níveis de energia — {species}"
                + (
                    f"   |   Bz={self.spin_magnetic_field.value():+.2f} T"
                    if abs(self.spin_magnetic_field.value()) > 1e-12 else ""
                )
            ),
            color="#edf7ff", fontsize=12, pad=10,
        )
        axis.text(
            0.01, 0.01, "posição vertical: ordem de Aufbau (escala qualitativa)",
            transform=axis.transAxes, color="#7790a5", fontsize=7,
        )
        self.figure.tight_layout(pad=1.2)
        self.canvas.draw_idle()

    def calculate_selected_transition(self):
        initial_state = self.combo_initial.currentData()
        final_state = self.combo_final.currentData()
        if not initial_state or not final_state:
            return
        initial_key = tuple(initial_state[:2])
        final_key = tuple(final_state[:2])
        if initial_key not in self.level_by_key or final_key not in self.level_by_key:
            return
        try:
            result = calculate_transition(
                self.level_by_key[initial_key], self.level_by_key[final_key],
                initial_m=initial_state[2], final_m=final_state[2],
                initial_spin=initial_state[3], final_spin=final_state[3],
                magnetic_field_t=self.spin_magnetic_field.value(),
            )
        except ValueError as error:
            self.transition_result.setText(f"<b>Transição inválida:</b> {error}")
            return

        mark = lambda valid: "✓" if valid else "✗"
        classification = (
            "PERMITIDA POR DIPOLO ELÉTRICO (E1) ✓"
            if result.electric_dipole_allowed
            else "NÃO PERMITIDA POR DIPOLO ELÉTRICO (E1) ✗"
        )
        comparison_html = ""
        if self.check_nist.isChecked():
            comparison = compare_with_nist_hydrogen(
                result, self.atomic_number, self.electron_count,
            )
            if comparison is not None:
                comparison_html = (
                    "<br><b>NIST — "
                    f"H I / {comparison.series}:</b> "
                    f"λ₀(modelo)={comparison.calculated_vacuum_nm:.3f} nm "
                    f"| λ(NIST)={comparison.reference_vacuum_nm:.3f} nm | "
                    f"Diferença = {comparison.absolute_difference_nm:.3f} nm "
                    f"| erro={comparison.relative_error_percent:.3f}% | "
                    f"<a href='{comparison.source_url}'>fonte</a><br>"
                    "Referência sem campo por níveis médios; estrutura fina não resolvida."
                )
            else:
                comparison_html = (
                    "<br><b>Comparação NIST:</b> "
                    + nist_comparison_scope_message(
                        result, self.atomic_number, self.electron_count,
                    )
                )
        magnetic_html = ""
        if abs(result.magnetic_field_t) > 1e-12:
            component = (
                "π (Δmₗ=0)"
                if result.delta_m == 0
                else f"σ (Δmₗ={result.delta_m:+d})"
            )
            wavelength_shift_pm = (
                result.wavelength_nm - result.zero_field_wavelength_nm
            ) * 1e3
            magnetic_html = (
                f"<br><b>ZEEMAN LINEAR — Bz={result.magnetic_field_t:+.2f} T; "
                f"componente {component}</b><br>"
                f"ΔE_Z(i)={result.initial_zeeman_shift_ev:+.3e} eV | "
                f"ΔE_Z(f)={result.final_zeeman_shift_ev:+.3e} eV | "
                f"efeito na transição={result.transition_zeeman_shift_ev:+.3e} eV | "
                f"Δf={result.zeeman_frequency_shift_hz / 1e9:+.3f} GHz<br>"
                f"λ(B=0)={result.zero_field_wavelength_nm:.6f} nm | "
                f"λ(B)={result.wavelength_nm:.6f} nm | "
                f"Δλ={wavelength_shift_pm:+.4f} pm. "
                "O campo altera a energia, não a forma 3D neste modelo."
            )
        self.transition_result.setText(
            f"<b>{result.process.upper()}</b> &nbsp; "
            f"ΔE = {result.delta_energy_ev:+.4f} eV &nbsp; | &nbsp; "
            f"f ≈ {result.frequency_hz:.3e} Hz &nbsp; | &nbsp; "
            f"λ ≈ {result.wavelength_nm:.2f} nm ({result.spectral_region})<br>"
            f"<b>{classification}</b><br>"
            f"Δn={result.delta_n:+d}: livre em E1 &nbsp; | &nbsp; "
            f"Δl={result.delta_l:+d}: {mark(result.dipole_l_allowed)} &nbsp; | &nbsp; "
            f"Δmₗ={result.delta_m:+d}: {mark(result.dipole_m_allowed)} &nbsp; | &nbsp; "
            f"Δmₛ={result.delta_spin:+.1f}: {mark(result.spin_allowed)} &nbsp; | &nbsp; "
            f"paridade muda: {mark(result.parity_changes)}<br>"
            "“Não permitida por E1” não significa impossível: mecanismos mais "
            "fracos, como M1 ou E2, não são calculados aqui."
            + magnetic_html
            + comparison_html
        )
        self._draw(self.species, transition=result)
