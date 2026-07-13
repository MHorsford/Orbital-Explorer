"""Gráficos didáticos da função e da probabilidade radial."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from physics.radial_distribution import calculate_radial_distribution


class RadialDistributionWidget(QWidget):
    """Exibe R_nl(r), P(r), nós e raios característicos."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.help_label = QLabel(
            "O gráfico superior preserva o sinal de Rₙₗ(r). O inferior mostra "
            "P(r)=r²|Rₙₗ(r)|², cuja área é normalizada para 1. A orientação mₗ "
            "não altera a parte radial."
        )
        self.help_label.setObjectName("helpBanner")
        self.help_label.setWordWrap(True)
        self.help_label.setToolTip(
            "Nós radiais são zeros de Rₙₗ(r) para r>0; nós angulares dependem de l."
        )
        layout.addWidget(self.help_label)

        self.figure = Figure(figsize=(7.5, 5.4), facecolor="#07111f")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setToolTip(
            "Linhas magenta: nós radiais. Linha amarela: raio mais provável."
        )
        layout.addWidget(self.canvas, stretch=1)

        self.summary = QLabel()
        self.summary.setObjectName("radialSummary")
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.RichText)
        layout.addWidget(self.summary)

    def update_state(
            self, n, l, z_eff, orbital_label, electron_count, interaction_mode,
    ):
        distribution = calculate_radial_distribution(n, l, z_eff)
        self._draw(distribution, orbital_label, z_eff)

        if electron_count:
            electron_word = "elétron" if electron_count == 1 else "elétrons"
            physical_note = (
                f"Orbital ocupado por {electron_count} {electron_word} "
                "na configuração exibida."
            )
        elif interaction_mode == "Explorar orbitais":
            physical_note = "Perfil matemático de um orbital atualmente vazio."
        else:
            physical_note = (
                "Orbital vazio: o perfil matemático existe, mas a densidade eletrônica "
                "física é zero."
            )

        nodes = distribution.radial_nodes_angstrom
        node_positions = (
            ", ".join(f"{value:.3f} Å" for value in nodes)
            if len(nodes) else "nenhum"
        )
        self.summary.setText(
            f"<b>{orbital_label}</b> &nbsp; Z_eff={z_eff:.2f}<br>"
            f"Nós radiais: <b>{distribution.radial_node_count}</b> "
            f"(n−l−1) &nbsp; | &nbsp; Nós angulares: "
            f"<b>{distribution.angular_node_count}</b> (l) &nbsp; | &nbsp; "
            f"Total: <b>{distribution.total_node_count}</b> (n−1)<br>"
            f"Posições dos nós radiais: {node_positions}<br>"
            f"Raio mais provável: <b>{distribution.most_probable_radius_angstrom:.3f} Å</b> "
            f"&nbsp; | &nbsp; ⟨r⟩: "
            f"<b>{distribution.mean_radius_angstrom:.3f} Å</b><br>"
            f"<i>{physical_note}</i>"
        )

    def _draw(self, distribution, orbital_label, z_eff):
        self.figure.clear()
        radial_axis, probability_axis = self.figure.subplots(2, 1, sharex=True)
        axes = (radial_axis, probability_axis)
        for axis in axes:
            axis.set_facecolor("#07111f")
            axis.tick_params(colors="#9fb5ca", labelsize=8)
            axis.grid(True, color="#29445a", alpha=0.32, linewidth=0.7)
            for spine in axis.spines.values():
                spine.set_color("#31556f")

        radius = distribution.radius_angstrom
        radial_axis.plot(
            radius, distribution.scaled_radial_amplitude,
            color="#55d7f2", linewidth=2.0,
        )
        radial_axis.axhline(0.0, color="#7890a6", linewidth=0.8)
        radial_axis.set_ylabel("R / max|R|", color="#dceeff", fontsize=9)
        radial_axis.set_title(
            f"{orbital_label} — função radial com sinal   (Z_eff={z_eff:.2f})",
            color="#edf7ff", fontsize=11, pad=8,
        )

        probability_axis.plot(
            radius, distribution.probability_density,
            color="#49d7a5", linewidth=2.0,
        )
        probability_axis.fill_between(
            radius, distribution.probability_density,
            color="#49d7a5", alpha=0.18,
        )
        probability_axis.axvline(
            distribution.most_probable_radius_angstrom,
            color="#ffd166", linestyle="--", linewidth=1.5,
            label="raio mais provável",
        )
        probability_axis.set_ylabel("P(r) [Å⁻¹]", color="#dceeff", fontsize=9)
        probability_axis.set_xlabel("Distância ao núcleo r [Å]", color="#dceeff", fontsize=9)

        for index, node in enumerate(distribution.radial_nodes_angstrom):
            label = "nó radial" if index == 0 else None
            radial_axis.axvline(
                node, color="#ff6fc7", linestyle=":", linewidth=1.4,
                label=label,
            )
            probability_axis.axvline(
                node, color="#ff6fc7", linestyle=":", linewidth=1.4,
                label=label,
            )

        radial_axis.set_xlim(0.0, distribution.range_max_angstrom)
        if len(distribution.radial_nodes_angstrom):
            radial_axis.legend(
                loc="upper right", fontsize=8, frameon=False,
                labelcolor="#dceeff",
            )
        probability_axis.legend(
            loc="upper right", fontsize=8, frameon=False,
            labelcolor="#dceeff",
        )
        self.figure.tight_layout(pad=1.25)
        self.canvas.draw_idle()
