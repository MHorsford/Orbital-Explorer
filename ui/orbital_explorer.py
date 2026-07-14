"""
ui/orbital_explorer.py

Interface PyQt5 para explorar orbitais atômicos de forma interativa.
Permite visualizar qualquer orbital (n, l, m) do hidrogênio e outros átomos.
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from pathlib import Path

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel,
    QPushButton, QComboBox, QGroupBox, QGridLayout, QTextEdit,
    QApplication, QCheckBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QSplitter, QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtGui import QIcon

# Backend Qt usado pelos gráficos de corte 2D incorporados à janela.
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from simulator.simulator import Simulator
from simulator.scene import Scene
from atom.manual_configuration import ManualElectronConfiguration
from orbitals.orbital import Orbital
from ui.app_theme import apply_app_theme
from utils.helpers import quantum_label
from config import (
    MAX_N, MAX_Z, ISO_VALUE, ORBITAL_EMPTY_POSITIVE_COLOR,
    ORBITAL_EMPTY_NEGATIVE_COLOR, ORBITAL_NEGATIVE_PHASE_COLOR,
)
import numpy as np


class OrbitalExplorer(QMainWindow):
    """
    Janela principal da UI para exploração de orbitais.
    """
    
    def __init__(self, simulator):
        """
        Parâmetros:
            simulator : objeto Simulator já inicializado
        """
        super().__init__()
        self.simulator = simulator
        self.interaction_mode = "Explorar orbitais"
        self.manual_config = None
        self.manual_feedback = ""
        self.manual_extra_levels = 0
        self.superposition_phase = 0.0
        self.superposition_cache = None
        self.superposition_cache_key = None
        self.superposition_primary_state = None
        self.superposition_dynamics = None
        self.superposition_rendering = False
        self.superposition_last_tick = None
        self.superposition_status_tick = 0
        self.iso_control_value = int(ISO_VALUE * 100)
        self.volume_brightness_percent = 100
        self.render_parameter_mode = 'isosurface'
        self.setWindowTitle("🧪 Orbital Explorer — Simulador de Orbitais Atômicos")
        self.setGeometry(80, 60, 1560, 900)
        self.setMinimumSize(1200, 720)

        
        # 1. Descobre o caminho da pasta raiz do projeto (onde está a pasta 'data')
        # Como este arquivo está dentro da pasta 'ui', precisamos subir um nível (..)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # 2. Aponta para a imagem dentro da pasta data
        caminho_icone = os.path.join(base_dir, "assets", "images", "quantum_icon_256.ico")


        # 3. Aplica o ícone na janela atual
        self.setWindowIcon(QIcon(caminho_icone))
        
        # Widget central
        self.central_widget = QWidget()
        self.central_widget.setObjectName("appRoot")
        self.setCentralWidget(self.central_widget)
        
        # Layout principal
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(14, 14, 14, 14)
        self.main_layout.setSpacing(12)

        # Divisores ajustáveis evitam que os painéis laterais esmaguem a
        # visualização 3D em telas menores e permitem redistribuir o espaço.
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setObjectName("mainSplitter")
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(8)
        self.main_layout.addWidget(self.main_splitter)
        
        # Criar painéis
        self.create_control_panel()
        self.create_viewer_panel()
        self.create_info_panel()
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setSizes([340, 790, 430])
        apply_app_theme(self)
        
        # Timer para atualizações suaves
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.on_slider_changed)
        self.update_timer.setInterval(100)
        self.timer_pending = False

        self.superposition_timer = QTimer(self)
        # Aproximadamente 6 quadros/s: suficiente para a evolução didática sem
        # monopolizar a thread da interface durante volume ray casting.
        self.superposition_timer.setInterval(160)
        self.superposition_timer.timeout.connect(self.advance_superposition)

        self.slice_canvases = {'amplitude': None, 'probability': None}
        self.update_filling_diagram()
        self.update_manual_panel()

        # O Renderer usa um isovalor adaptativo até o primeiro ajuste manual
        # do slider. A flag distingue a configuração inicial da ação do usuário.
        self.iso_manually_set = False
        self.on_render_clicked()

    def load_all_elements(self):
        """Carrega todos os elementos do CSV."""
        csv_path = Path(__file__).parent.parent / "data" / "periodic_table.csv"
        if not csv_path.exists():
            csv_path = Path("data/periodic_table.csv")
        
        try:
            df = pd.read_csv(csv_path)
            elements = []
            for _, row in df.iterrows():
                name = f"{row['name']} ({row['symbol']})"
                Z = int(row['atomic_number'])
                elements.append((name, Z))
            return elements
        except Exception as e:
            print(f"⚠ Erro: {e}. Usando fallback.")
            return [("Hydrogen (H)", 1), ("Helium (He)", 2), ("Carbon (C)", 6)]
    
    # CRIAÇÃO DA UI
    
    def create_control_panel(self):
            control_panel = QGroupBox("Controles de Orbital")
            control_panel.setObjectName("sideCard")
            control_panel.setMinimumWidth(300)
            control_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            layout = QGridLayout()
            layout.setContentsMargins(14, 18, 14, 14)
            layout.setHorizontalSpacing(10)
            layout.setVerticalSpacing(12)
            layout.setColumnMinimumWidth(0, 112)
            layout.setColumnStretch(1, 1)
            layout.setColumnMinimumWidth(2, 64)

            # Elemento
            layout.addWidget(QLabel("Elemento (Z):"), 0, 0)
            self.combo_element = QComboBox()
            for name, Z in self.load_all_elements():
                if Z <= MAX_Z:
                    self.combo_element.addItem(name, Z)
            self.combo_element.currentIndexChanged.connect(self.on_element_changed)
            layout.addWidget(self.combo_element, 0, 1, 1, 2)

            # Espécie iônica com significado e quantidade de elétrons explícitos.
            layout.addWidget(QLabel("Espécie / carga:"), 1, 0)
            self.combo_charge = QComboBox()
            initial_charge = getattr(self.simulator.atom, "charge", 0)
            self._populate_charge_options(
                self.combo_element.currentData(), initial_charge
            )
            self.combo_charge.currentIndexChanged.connect(self.on_charge_changed)
            layout.addWidget(self.combo_charge, 1, 1, 1, 2)

            # n            
            layout.addWidget(QLabel("Nível (n):"), 2, 0)
            self.slider_n = QSlider(Qt.Horizontal)
            self.slider_n.setRange(1, MAX_N)
            self.slider_n.setValue(1)
            self.slider_n.sliderMoved.connect(self.schedule_update)
            self.slider_n.valueChanged.connect(self.on_n_changed)
            layout.addWidget(self.slider_n, 2, 1)
            self.label_n = QLabel("n = 1")
            self.label_n.setFont(QFont("Cascadia Mono", 10, QFont.Bold))
            layout.addWidget(self.label_n, 2, 2)

            # l
            layout.addWidget(QLabel("Tipo orbital (l):"), 3, 0)
            self.slider_l = QSlider(Qt.Horizontal)
            self.slider_l.setRange(0, MAX_N - 1)
            self.slider_l.setValue(0)
            self.slider_l.sliderMoved.connect(self.schedule_update)
            self.slider_l.valueChanged.connect(self.on_l_changed)
            layout.addWidget(self.slider_l, 3, 1)
            self.label_l = QLabel("l = 0 (s)")
            self.label_l.setFont(QFont("Cascadia Mono", 10, QFont.Bold))
            layout.addWidget(self.label_l, 3, 2)

            # m
            layout.addWidget(QLabel("Orientação (m):"), 4, 0)
            self.slider_m = QSlider(Qt.Horizontal)
            self.slider_m.setRange(-3, 3)
            self.slider_m.setValue(0)
            self.slider_m.sliderMoved.connect(self.schedule_update)
            self.slider_m.valueChanged.connect(self.on_m_changed)
            layout.addWidget(self.slider_m, 4, 1)
            self.label_m = QLabel("m = 0")
            self.label_m.setFont(QFont("Cascadia Mono", 10, QFont.Bold))
            layout.addWidget(self.label_m, 4, 2)

            # Modo renderização
            layout.addWidget(QLabel("Renderização:"), 5, 0)
            self.combo_mode = QComboBox()
            self.combo_mode.addItem("Isosuperfície", "isosurface")
            self.combo_mode.addItem("Nuvem de densidade", "density_volume")
            self.combo_mode.addItem("Nuvem de pontos", "points")
            self.combo_mode.addItem("Grade de pontos", "grid_points")
            self.combo_mode.currentIndexChanged.connect(self.on_mode_changed)
            layout.addWidget(self.combo_mode, 5, 1, 1, 2)

            # Iso value
            self.label_render_parameter = QLabel("Isosuperfície:")
            layout.addWidget(self.label_render_parameter, 6, 0)
            self.slider_iso = QSlider(Qt.Horizontal)
            self.slider_iso.setRange(0, 200)
            
            # Converte o valor do config (ex: 0.12 -> 12)
            initial_iso_slider_val = self.iso_control_value
            self.slider_iso.setValue(initial_iso_slider_val) 
            self.slider_iso.sliderMoved.connect(self.on_iso_slider_moved)
            layout.addWidget(self.slider_iso, 6, 1)
            
            self.label_iso = QLabel(f"{ISO_VALUE:.3f}") # Texto sincronizado
            layout.addWidget(self.label_iso, 6, 2)

            # Dinâmica quântica
            layout.addWidget(QLabel("Dinâmica:"), 7, 0)
            self.check_superposition = QCheckBox("Superposição temporal")
            self.check_superposition.stateChanged.connect(
                self.on_superposition_toggled
            )
            layout.addWidget(self.check_superposition, 7, 1, 1, 2)

            # Plano do corte 2D
            # Orbitais com dependência azimutal podem exigir um plano de corte
            # específico; os três planos cartesianos permanecem disponíveis.
            layout.addWidget(QLabel("Plano do corte:"), 8, 0)
            self.combo_plane = QComboBox()
            self.combo_plane.addItems(["xz", "xy", "yz"])
            layout.addWidget(self.combo_plane, 8, 1, 1, 2)

            # Modo de interação física/didática
            layout.addWidget(QLabel("Interação:"), 9, 0)
            self.combo_interaction_mode = QComboBox()
            self.combo_interaction_mode.addItems([
                "Explorar orbitais",
                "Átomo real",
                "Preenchimento manual",
            ])
            self.combo_interaction_mode.currentTextChanged.connect(
                self.on_interaction_mode_changed
            )
            layout.addWidget(self.combo_interaction_mode, 9, 1, 1, 2)

            # Botões
            button_layout = QVBoxLayout()
            self.btn_render = QPushButton("🎨 Renderizar Orbital")
            self.btn_render.setProperty("variant", "primary")
            self.btn_render.clicked.connect(self.on_render_clicked)
            button_layout.addWidget(self.btn_render)

            self.btn_all_filled = QPushButton("📊 Mostrar Preenchidos")
            self.btn_all_filled.setProperty("variant", "accent")
            self.btn_all_filled.clicked.connect(self.on_show_filled)
            button_layout.addWidget(self.btn_all_filled)

            self.btn_rules = QPushButton("✅ Verificar Aufbau • Hund • Pauli")
            self.btn_rules.setProperty("variant", "success")
            self.btn_rules.clicked.connect(self.on_rules_clicked)
            button_layout.addWidget(self.btn_rules)

            self.btn_sequence = QPushButton("📈 Sequência Aufbau")
            self.btn_sequence.setProperty("variant", "warning")
            self.btn_sequence.clicked.connect(self.on_sequence_clicked)
            button_layout.addWidget(self.btn_sequence)

            self.btn_slice2d = QPushButton("📐 Gerar Corte 2D")
            self.btn_slice2d.setProperty("variant", "accent")
            self.btn_slice2d.clicked.connect(self.on_slice2d_clicked)
            button_layout.addWidget(self.btn_slice2d)

            layout.addLayout(button_layout, 10, 0, 1, 3)
            # Mantém todos os controles agrupados no topo. O espaço excedente
            # da janela fica abaixo dos botões, sem criar grandes lacunas.
            layout.setRowStretch(11, 1)

            self._configure_control_tooltips()
            control_panel.setLayout(layout)
            self.control_scroll = QScrollArea()
            self.control_scroll.setObjectName("panelScroll")
            self.control_scroll.setWidgetResizable(True)
            self.control_scroll.setFrameShape(QScrollArea.NoFrame)
            self.control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.control_scroll.setMinimumWidth(320)
            self.control_scroll.setWidget(control_panel)
            self.main_splitter.addWidget(self.control_scroll)

    def _configure_control_tooltips(self):
        """Adiciona explicações curtas aos controles menos óbvios."""
        self.combo_element.setToolTip(
            "Escolha o elemento químico. Z é o número de prótons do núcleo."
        )
        self.slider_n.setToolTip(
            "Número quântico principal n: controla a camada, o tamanho e os nós radiais."
        )
        self.label_n.setToolTip(self.slider_n.toolTip())
        self.slider_l.setToolTip(
            "Número quântico azimutal l: define o tipo e a forma do orbital (s, p, d, f...)."
        )
        self.label_l.setToolTip(self.slider_l.toolTip())
        self.slider_m.setToolTip(
            "Número quântico magnético mₗ: escolhe a orientação espacial, de −l até +l."
        )
        self.label_m.setToolTip(self.slider_m.toolTip())

        mode_tips = (
            "Superfície de amplitude constante; destaca a forma e as fases de ψ.",
            "Volume translúcido: o brilho acompanha |ψ|² e a cor representa a fase.",
            "Amostra pontos segundo |ψ|²; representa uma nuvem de probabilidade.",
            "Mostra pontos do grid acima do limiar; útil para observar a amostragem.",
        )
        for index, tooltip in enumerate(mode_tips):
            self.combo_mode.setItemData(index, tooltip, Qt.ToolTipRole)
        self.combo_mode.setToolTip(mode_tips[0])

        iso_tip = (
            "Limiar da isosuperfície. Valores menores mostram regiões mais difusas; "
            "valores maiores aproximam a superfície do núcleo."
        )
        self.slider_iso.setToolTip(iso_tip)
        self.label_iso.setToolTip(iso_tip)
        self.label_render_parameter.setToolTip(iso_tip)
        self.check_superposition.setToolTip(
            "Combina dois estados espaciais de um mesmo elétron e anima a "
            "densidade |Ψ(t)|² produzida pela interferência. A ocupação "
            "eletrônica do átomo não é modificada."
        )
        self.combo_plane.setToolTip(
            "Plano cartesiano usado nos gráficos de amplitude ψ e probabilidade |ψ|²."
        )

        interaction_tips = (
            "Exibe a forma matemática mesmo quando o orbital está vazio.",
            "Mostra somente a ocupação da configuração fundamental da espécie.",
            "Começa com orbitais vazios para adicionar, remover e promover elétrons.",
        )
        self.combo_interaction_mode.blockSignals(True)
        for index, tooltip in enumerate(interaction_tips):
            self.combo_interaction_mode.setItemData(index, tooltip, Qt.ToolTipRole)
        self.combo_interaction_mode.blockSignals(False)
        self.combo_interaction_mode.setToolTip(interaction_tips[0])

        self.btn_render.setToolTip("Recalcula e enquadra o orbital selecionado em 3D.")
        self.btn_all_filled.setToolTip(
            "Exibe simultaneamente os orbitais que possuem elétrons."
        )
        self.btn_rules.setToolTip(
            "Abre o diagnóstico de Aufbau, Hund e Exclusão de Pauli."
        )
        self.btn_sequence.setToolTip(
            "Mostra a ordem energética usada no preenchimento de Aufbau."
        )
        self.btn_slice2d.setToolTip(
            "Gera os cortes de ψ e |ψ|² no plano cartesiano escolhido."
        )
    
    def create_viewer_panel(self):
        """Cria as áreas 3D e 2D como abas da região central."""
        viewer_panel = QGroupBox("Visualizações do Orbital")
        viewer_panel.setObjectName("viewerCard")
        viewer_panel.setMinimumWidth(420)
        from pyvistaqt import QtInteractor
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 12, 8, 8)

        self.viewer_tabs = QTabWidget()
        self.viewer_tabs.setObjectName("viewerTabs")

        self.viewer_3d_tab = QWidget()
        viewer_3d_layout = QVBoxLayout(self.viewer_3d_tab)
        viewer_3d_layout.setContentsMargins(0, 0, 0, 0)

        self.superposition_panel = QGroupBox("Superposição e evolução temporal")
        self.superposition_panel.setObjectName("dynamicsPanel")
        self.superposition_panel.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Maximum
        )
        self.superposition_panel.setMaximumHeight(158)
        dynamics_layout = QGridLayout(self.superposition_panel)
        dynamics_layout.setContentsMargins(10, 14, 10, 8)
        dynamics_layout.setHorizontalSpacing(8)
        dynamics_layout.setVerticalSpacing(6)

        self.superposition_state_a_label = QLabel("A: 1s")
        self.superposition_state_a_label.setToolTip(
            "Estado A: orbital definido pelos controles quânticos à esquerda."
        )
        dynamics_layout.addWidget(self.superposition_state_a_label, 0, 0)
        dynamics_layout.addWidget(QLabel("B:"), 0, 1)
        self.combo_superposition_state_b = QComboBox()
        self.combo_superposition_state_b.setMinimumContentsLength(12)
        self.combo_superposition_state_b.setToolTip(
            "Segundo estado da combinação coerente. Escolha energias diferentes "
            "para observar evolução da densidade. A lista usa níveis próximos "
            "ao estado A para preservar resolução e desempenho."
        )
        dynamics_layout.addWidget(
            self.combo_superposition_state_b, 0, 2, 1, 2
        )
        self.btn_superposition_play = QPushButton("Pausar")
        self.btn_superposition_play.setFixedWidth(92)
        self.btn_superposition_play.setToolTip(
            "Inicia ou pausa a evolução visual da superposição."
        )
        self.btn_superposition_play.clicked.connect(
            self.toggle_superposition_playback
        )
        dynamics_layout.addWidget(self.btn_superposition_play, 0, 4)
        self.btn_superposition_reset = QPushButton("Reiniciar")
        self.btn_superposition_reset.setFixedWidth(92)
        self.btn_superposition_reset.setToolTip(
            "Retorna a fase relativa a 0° e redesenha a superposição."
        )
        self.btn_superposition_reset.clicked.connect(
            self.reset_superposition_phase
        )
        dynamics_layout.addWidget(self.btn_superposition_reset, 0, 5)
        self.check_superposition_phase_colors = QCheckBox("Fase em cores")
        self.check_superposition_phase_colors.setChecked(True)
        self.check_superposition_phase_colors.setToolTip(
            "Na nuvem de densidade, o brilho continua representando |Ψ|² e "
            "a cor passa a representar arg(Ψ). O estado A define a referência "
            "de fase. Desative para usar somente a cor ciano."
        )
        self.check_superposition_phase_colors.stateChanged.connect(
            self.on_superposition_phase_coloring_changed
        )
        dynamics_layout.addWidget(
            self.check_superposition_phase_colors, 0, 6
        )

        dynamics_layout.addWidget(QLabel("Peso de B:"), 1, 0)
        self.slider_superposition_weight = QSlider(Qt.Horizontal)
        self.slider_superposition_weight.setRange(0, 100)
        self.slider_superposition_weight.setValue(50)
        self.slider_superposition_weight.setToolTip(
            "Probabilidade associada ao estado B; 50% cria uma mistura equilibrada."
        )
        dynamics_layout.addWidget(
            self.slider_superposition_weight, 1, 1, 1, 2
        )
        self.label_superposition_weight = QLabel("50%")
        dynamics_layout.addWidget(self.label_superposition_weight, 1, 3)
        dynamics_layout.addWidget(QLabel("Ritmo visual:"), 1, 4)
        self.slider_superposition_speed = QSlider(Qt.Horizontal)
        self.slider_superposition_speed.setRange(10, 150)
        self.slider_superposition_speed.setValue(50)
        self.slider_superposition_speed.setToolTip(
            "Velocidade desacelerada da animação. Não representa a escala de "
            "tempo física, que é informada abaixo."
        )
        dynamics_layout.addWidget(self.slider_superposition_speed, 1, 5)
        self.label_superposition_speed = QLabel("0,50 ciclo/s")
        self.label_superposition_speed.setMinimumWidth(78)
        dynamics_layout.addWidget(self.label_superposition_speed, 1, 6)

        self.superposition_status = QLabel()
        self.superposition_status.setObjectName("dynamicsStatus")
        self.superposition_status.setWordWrap(True)
        self.superposition_status.setMaximumHeight(58)
        dynamics_layout.addWidget(self.superposition_status, 2, 0, 1, 7)
        dynamics_layout.setColumnStretch(2, 1)
        dynamics_layout.setColumnStretch(5, 1)

        self.slider_superposition_weight.valueChanged.connect(
            self.on_superposition_weight_value_changed
        )
        self.slider_superposition_weight.sliderReleased.connect(
            self.on_superposition_weight_changed
        )
        self.slider_superposition_speed.valueChanged.connect(
            self.on_superposition_speed_changed
        )
        self.combo_superposition_state_b.currentIndexChanged.connect(
            self.on_superposition_state_b_changed
        )
        self.populate_superposition_states()
        self.superposition_panel.setVisible(False)
        viewer_3d_layout.addWidget(self.superposition_panel)

        self.pyvista_widget = QtInteractor(self.viewer_3d_tab)
        self.simulator.scene.use_plotter(self.pyvista_widget)
        try:
            self.pyvista_widget.set_background("#07101d")
        except Exception:
            pass
        if hasattr(self.simulator, 'visible_orbitals'):
            self.simulator.visible_orbitals.clear()

        viewer_3d_layout.addWidget(self.pyvista_widget)
        self.phase_legend = QLabel()
        self.phase_legend.setObjectName("phaseLegend")
        self.phase_legend.setAlignment(Qt.AlignCenter)
        self.phase_legend.setWordWrap(True)
        self.phase_legend.setTextFormat(Qt.RichText)
        viewer_3d_layout.addWidget(self.phase_legend)
        self.update_phase_legend(multiple=True)
        viewer_3d_index = self.viewer_tabs.addTab(
            self.viewer_3d_tab, "3D"
        )
        self.viewer_tabs.setTabToolTip(
            viewer_3d_index, "Forma tridimensional e fases da função de onda"
        )

        # O corte ocupa a mesma área nobre do 3D. As duas representações 2D
        # continuam separadas em subabas para comparação sem sobrecarregar a UI.
        self.viewer_2d_tab = QWidget()
        self.viewer_2d_tab.setObjectName("sliceWorkspace")
        viewer_2d_layout = QVBoxLayout(self.viewer_2d_tab)
        viewer_2d_layout.setContentsMargins(8, 8, 8, 8)
        viewer_2d_layout.setSpacing(8)

        self.slice_status = QLabel(
            "Escolha o plano à esquerda e clique em “Gerar Corte 2D”."
        )
        self.slice_status.setObjectName("sliceStatus")
        self.slice_status.setWordWrap(True)
        viewer_2d_layout.addWidget(self.slice_status)

        self.slice_tabs = QTabWidget()
        self.slice_tabs.setObjectName("sliceTabs")
        self.slice_amp_tab = QWidget()
        self.slice_prob_tab = QWidget()
        self.slice_amp_layout = QVBoxLayout(self.slice_amp_tab)
        self.slice_prob_layout = QVBoxLayout(self.slice_prob_tab)
        self.slice_amp_layout.setContentsMargins(6, 6, 6, 6)
        self.slice_prob_layout.setContentsMargins(6, 6, 6, 6)
        self.slice_placeholders = {
            'amplitude': QLabel("A amplitude ψ aparecerá aqui."),
            'probability': QLabel("A probabilidade |ψ|² aparecerá aqui."),
        }
        for placeholder in self.slice_placeholders.values():
            placeholder.setObjectName("emptySlice")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setWordWrap(True)
        self.slice_amp_layout.addWidget(self.slice_placeholders['amplitude'])
        self.slice_prob_layout.addWidget(self.slice_placeholders['probability'])
        self.slice_tabs.addTab(self.slice_amp_tab, "Amplitude ψ")
        self.slice_tabs.addTab(self.slice_prob_tab, "Probabilidade |ψ|²")
        viewer_2d_layout.addWidget(self.slice_tabs, stretch=1)
        viewer_2d_index = self.viewer_tabs.addTab(self.viewer_2d_tab, "2D")
        self.viewer_tabs.setTabToolTip(
            viewer_2d_index, "Amplitude ψ e probabilidade |ψ|² em um plano"
        )

        from ui.radial_distribution import RadialDistributionWidget
        self.radial_widget = RadialDistributionWidget()
        radial_tab_index = self.viewer_tabs.addTab(
            self.radial_widget, "Radial"
        )
        self.viewer_tabs.setTabToolTip(
            radial_tab_index,
            "Função radial, probabilidade, nós e raios característicos",
        )

        from ui.energy_diagram import EnergyDiagramWidget
        self.energy_widget = EnergyDiagramWidget()
        energy_tab_index = self.viewer_tabs.addTab(
            self.energy_widget, "Energia"
        )
        self.viewer_tabs.setTabToolTip(
            energy_tab_index,
            "Níveis aproximados, absorção, emissão e regra parcial de seleção",
        )

        layout.addWidget(self.viewer_tabs)
        viewer_panel.setLayout(layout)
        self.main_splitter.addWidget(viewer_panel)

    def create_info_panel(self):
        """Cria o painel de informações."""
        info_panel = QGroupBox("Informações do Orbital")
        info_panel.setObjectName("sideCard")
        info_panel.setMinimumWidth(390)
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(9)
        
        # Título do orbital
        self.label_orbital_name = QLabel()
        self.label_orbital_name.setObjectName("orbitalTitle")
        self.label_orbital_name.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self.label_orbital_name.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_orbital_name)

        self.quantum_state_card = QLabel()
        self.quantum_state_card.setObjectName("quantumStateCard")
        self.quantum_state_card.setWordWrap(True)
        self.quantum_state_card.setTextFormat(Qt.RichText)
        
        # Informações detalhadas e diagrama didático das regras de ocupação.
        self.info_tabs = QTabWidget()
        self.info_tabs.setObjectName("infoTabs")
        self.text_info = QTextEdit()
        self.text_info.setReadOnly(True)
        self.text_info.setFont(QFont("Cascadia Mono", 9))

        self.data_tab = QWidget()
        data_layout = QVBoxLayout(self.data_tab)
        data_layout.setContentsMargins(8, 8, 8, 8)
        data_layout.setSpacing(8)
        data_layout.addWidget(self.quantum_state_card)
        data_layout.addWidget(self.text_info, stretch=1)

        self.rules_text = QTextEdit()
        self.rules_text.setReadOnly(True)
        self.rules_text.setFont(QFont("Cascadia Mono", 9))

        self.data_tab_index = self.info_tabs.addTab(self.data_tab, "Dados")
        self.info_tabs.setTabToolTip(
            self.data_tab_index, "Números quânticos e parâmetros do orbital selecionado"
        )
        self.rules_tab_index = self.info_tabs.addTab(self.rules_text, "Regras")
        self.info_tabs.setTabToolTip(
            self.rules_tab_index, "Regras de Aufbau, Hund, Pauli e spins"
        )

        self.manual_tab = QWidget()
        manual_tab_layout = QVBoxLayout(self.manual_tab)
        manual_tab_layout.setContentsMargins(0, 0, 0, 0)
        self.manual_scroll = QScrollArea()
        self.manual_scroll.setObjectName("manualScroll")
        self.manual_scroll.setWidgetResizable(True)
        self.manual_scroll.setFrameShape(QScrollArea.NoFrame)
        self.manual_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.manual_content = QWidget()
        self.manual_content.setObjectName("manualContent")
        manual_layout = QVBoxLayout(self.manual_content)
        manual_layout.setContentsMargins(6, 6, 6, 6)
        manual_layout.setSpacing(5)
        self.manual_scroll.setWidget(self.manual_content)
        manual_tab_layout.addWidget(self.manual_scroll)
        self.manual_help = QLabel(
            "1. Clique em um orbital abaixo.  2. Adicione ↑ ou ↓.  "
            "Para uma excitação, escolha o destino e use 'Mover elétron'."
        )
        self.manual_help.setObjectName("helpBanner")
        self.manual_help.setWordWrap(True)
        manual_layout.addWidget(self.manual_help)
        self.manual_status = QLabel()
        self.manual_status.setObjectName("manualStatus")
        self.manual_status.setWordWrap(True)
        self.manual_status.setFont(QFont("Arial", 9))
        manual_layout.addWidget(self.manual_status)

        self.manual_table = QTableWidget(0, 2)
        self.manual_table.setHorizontalHeaderLabels(
            ["Energia / orbital", "Ocupação"]
        )
        self.manual_table.verticalHeader().setVisible(False)
        self.manual_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.manual_table.setShowGrid(False)
        self.manual_table.setAlternatingRowColors(False)
        self.manual_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.manual_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.manual_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.manual_table.cellClicked.connect(self.on_manual_cell_clicked)
        self.manual_table.setMinimumHeight(120)
        self.manual_table.setMaximumHeight(160)
        manual_layout.addWidget(self.manual_table, stretch=1)

        self.btn_show_levels = QPushButton("Mostrar próximos níveis de energia")
        self.btn_show_levels.setProperty("variant", "accent")
        self.btn_show_levels.clicked.connect(self.on_show_more_levels)
        manual_layout.addWidget(self.btn_show_levels)

        move_layout = QGridLayout()
        move_layout.setHorizontalSpacing(8)
        move_layout.addWidget(QLabel("Mover/promover elétron — origem:"), 0, 0, 1, 2)
        self.combo_move_source = QComboBox()
        move_layout.addWidget(self.combo_move_source, 1, 0)
        self.btn_move_electron = QPushButton("Mover elétron → selecionado")
        self.btn_move_electron.setProperty("variant", "warning")
        self.btn_move_electron.clicked.connect(self.on_manual_move_electron)
        move_layout.addWidget(self.btn_move_electron, 1, 1)
        move_layout.setColumnStretch(0, 1)
        manual_layout.addLayout(move_layout)

        self.manual_legend = QLabel(
            "E1 é a menor energia. Verde: fundamental ocupado  |  "
            "Amarelo: excitado ocupado  |  Azul: selecionado vazio  |  "
            "Branco: disponível"
        )
        self.manual_legend.setWordWrap(True)
        self.manual_legend.setFont(QFont("Arial", 8))
        manual_layout.addWidget(self.manual_legend)

        manual_buttons = QGridLayout()
        self.btn_spin_up = QPushButton("Adicionar ↑")
        self.btn_spin_down = QPushButton("Adicionar ↓")
        self.btn_remove_electron = QPushButton("Remover")
        self.btn_undo_manual = QPushButton("Desfazer")
        self.btn_reset_manual = QPushButton("Reiniciar")
        self.btn_auto_fill = QPushButton("Preencher automaticamente")
        self.btn_verify_manual = QPushButton("Verificar configuração")
        self.btn_spin_up.setProperty("variant", "primary")
        self.btn_spin_down.setProperty("variant", "primary")
        self.btn_remove_electron.setProperty("variant", "danger")
        self.btn_auto_fill.setProperty("variant", "success")
        self.btn_verify_manual.setProperty("variant", "accent")
        self.btn_spin_up.clicked.connect(
            lambda: self.on_manual_add_electron(Orbital.SPIN_UP)
        )
        self.btn_spin_down.clicked.connect(
            lambda: self.on_manual_add_electron(Orbital.SPIN_DOWN)
        )
        self.btn_remove_electron.clicked.connect(self.on_manual_remove_electron)
        self.btn_undo_manual.clicked.connect(self.on_manual_undo)
        self.btn_reset_manual.clicked.connect(self.on_manual_reset)
        self.btn_auto_fill.clicked.connect(self.on_manual_auto_fill)
        self.btn_verify_manual.clicked.connect(self.on_manual_verify)
        manual_buttons.addWidget(self.btn_spin_up, 0, 0)
        manual_buttons.addWidget(self.btn_spin_down, 0, 1)
        manual_buttons.addWidget(self.btn_remove_electron, 1, 0)
        manual_buttons.addWidget(self.btn_undo_manual, 1, 1)
        manual_buttons.addWidget(self.btn_reset_manual, 2, 0)
        manual_buttons.addWidget(self.btn_auto_fill, 2, 1)
        manual_buttons.addWidget(self.btn_verify_manual, 3, 0, 1, 2)
        manual_layout.addLayout(manual_buttons)

        self.manual_buttons = [
            self.btn_spin_up, self.btn_spin_down, self.btn_remove_electron,
            self.btn_undo_manual, self.btn_reset_manual, self.btn_auto_fill,
            self.btn_verify_manual, self.btn_show_levels, self.btn_move_electron,
        ]
        self.manual_tab_index = self.info_tabs.addTab(
            self.manual_tab, "Elétrons"
        )
        self.info_tabs.setTabToolTip(
            self.manual_tab_index, "Preenchimento eletrônico manual"
        )
        self.info_tabs.setTabEnabled(self.manual_tab_index, False)

        from ui.scientific_references import ScientificReferencesWidget
        self.references_widget = ScientificReferencesWidget()
        self.references_tab_index = self.info_tabs.addTab(
            self.references_widget, "Fontes"
        )
        self.info_tabs.setTabToolTip(
            self.references_tab_index,
            "Referências científicas usadas como base e fontes para consulta",
        )

        info_tab_bar = self.info_tabs.tabBar()
        info_tab_bar.setExpanding(False)
        info_tab_bar.setUsesScrollButtons(False)
        info_tab_bar.setElideMode(Qt.ElideRight)
        layout.addWidget(self.info_tabs, stretch=3)
        
        # Descrição
        self.label_description = QLabel()
        self.label_description.setWordWrap(True)
        self.label_description.setFont(QFont("Segoe UI", 9))
        layout.addWidget(self.label_description)
        
        info_panel.setLayout(layout)
        self.main_splitter.addWidget(info_panel)
    
    # CALLBACKS DOS SLIDERS
    
    def _charge_option_text(self, Z, charge):
        electrons = Z - charge
        charge_text = f"{charge:+d}".replace("-", "−")
        if charge == 0:
            species = "Neutro"
            charge_text = "0"
        elif charge > 0:
            species = "Cátion"
        elif Z == 1 and charge < -1:
            species = "Ânion formal ⚠"
        else:
            species = "Ânion"
        electron_word = "elétron" if electrons == 1 else "elétrons"
        return f"{species} ({charge_text}) · {electrons} {electron_word}"

    def _charge_option_tooltip(self, Z, charge):
        if charge == 0:
            return "Átomo neutro: número de prótons igual ao de elétrons."
        if charge > 0:
            return f"Cátion: {charge} elétron(s) removido(s) do átomo neutro."
        if Z == 1 and charge < -1:
            return (
                "Configuração eletrônica formal para exploração. "
                "Esta espécie de hidrogênio não é um íon isolado estável."
            )
        return f"Ânion: {abs(charge)} elétron(s) adicionado(s) ao átomo neutro."

    def _populate_charge_options(self, Z, selected=0):
        from physics.screening import max_supported_electrons

        minimum = max(-3, Z - max_supported_electrons())
        maximum = min(3, Z)
        charges = [
            charge for charge in (0, 1, 2, 3, -1, -2, -3)
            if minimum <= charge <= maximum
        ]
        self.combo_charge.blockSignals(True)
        self.combo_charge.clear()
        for charge in charges:
            self.combo_charge.addItem(self._charge_option_text(Z, charge), charge)
            index = self.combo_charge.count() - 1
            self.combo_charge.setItemData(
                index, self._charge_option_tooltip(Z, charge), Qt.ToolTipRole
            )
        index = self.combo_charge.findData(selected)
        self.combo_charge.setCurrentIndex(index if index >= 0 else 0)
        self.combo_charge.blockSignals(False)
        self._update_charge_tooltip()

    def _current_charge(self):
        charge = self.combo_charge.currentData()
        return int(charge) if charge is not None else 0

    def _set_charge(self, charge):
        index = self.combo_charge.findData(charge)
        if index >= 0:
            self.combo_charge.setCurrentIndex(index)

    def _update_charge_tooltip(self):
        index = self.combo_charge.currentIndex()
        tooltip = self.combo_charge.itemData(index, Qt.ToolTipRole) or ""
        self.combo_charge.setToolTip(tooltip)

    def _new_manual_configuration(self):
        atom = self.simulator.atom
        return ManualElectronConfiguration(atom.Z, charge=atom.charge)

    def _replace_atom(self, Z, charge, feedback):
        from atom.atom import Atom

        self.simulator.atom = Atom(Z=Z, charge=charge)
        if self.manual_config is not None or self.interaction_mode == "Preenchimento manual":
            self.manual_config = self._new_manual_configuration()
            self.manual_feedback = feedback
            self.manual_extra_levels = 0
        self.update_limits()
        self.update_filling_diagram()
        self.update_manual_panel()


    def on_element_changed(self, index):
        Z = self.combo_element.currentData()
        if Z is None:
            return
        self._populate_charge_options(Z, selected=0)
        self._replace_atom(Z, 0, "Novo elemento neutro; configuração manual reiniciada vazia.")

    def on_charge_changed(self, index):
        """Reconstrói o átomo mantendo Z e alterando o total de elétrons."""
        Z = self.combo_element.currentData()
        charge = self.combo_charge.itemData(index)
        if Z is None or charge is None:
            return
        self._update_charge_tooltip()
        charge = int(charge)
        if Z - charge == 0:
            feedback = (
                "Esta espécie não possui elétrons. Escolha 'Neutro' ou um "
                "ânion para inserir elétrons."
            )
        elif Z == 1 and charge < -1:
            feedback = (
                "Configuração formal para exploração: esta espécie de "
                "hidrogênio não é um íon isolado estável."
            )
        else:
            feedback = "Carga alterada; configuração manual reiniciada vazia."
        self._replace_atom(
            Z, charge, feedback
        )

    def on_interaction_mode_changed(self, mode):
        """Alterna entre exploração, átomo real e construção manual."""
        index = self.combo_interaction_mode.currentIndex()
        tooltip = self.combo_interaction_mode.itemData(index, Qt.ToolTipRole)
        if tooltip:
            self.combo_interaction_mode.setToolTip(tooltip)
        self.interaction_mode = mode
        if mode == "Preenchimento manual":
            self.info_tabs.setTabEnabled(self.manual_tab_index, True)
            if (self.manual_config is None
                    or self.manual_config.atomic_number != self.simulator.atom.Z
                    or self.manual_config.charge != self.simulator.atom.charge):
                self.manual_config = self._new_manual_configuration()
                self.manual_feedback = "Configuração manual iniciada com todos os orbitais vazios."
                self.manual_extra_levels = 0
            self.info_tabs.setCurrentWidget(self.manual_tab)
        elif mode == "Átomo real":
            self.info_tabs.setCurrentWidget(self.rules_text)
            self.info_tabs.setTabEnabled(self.manual_tab_index, False)
        else:
            self.info_tabs.setCurrentWidget(self.data_tab)
            self.info_tabs.setTabEnabled(self.manual_tab_index, False)
        self.update_manual_panel()
        self.update_filling_diagram()
        self.on_render_clicked()

    def on_n_changed(self):
        """Quando muda o nível quântico n."""
        n = self.slider_n.value()
        self.label_n.setText(f"n = {n}")
        
        # Limitar l dinamicamente baseado em n
        max_l = n - 1
        self.slider_l.setMaximum(max_l)
        if self.slider_l.value() > max_l:
            self.slider_l.setValue(max_l)
        
        # Apenas chama o próximo da cadeia de validação
        self.on_l_changed()
    
    def on_l_changed(self):
        """Quando muda o tipo orbital l."""
        from orbitals.orbital_types import get_orbital_type
        
        l = self.slider_l.value()
        orbital_type = get_orbital_type(l)
        self.label_l.setText(f"l = {l} ({orbital_type.letter})")
        
        # Limitar m dinamicamente baseado em l
        self.slider_m.setRange(-l, l)
        if abs(self.slider_m.value()) > l:
            self.slider_m.setValue(0)
        
        # Finaliza a cadeia de atualização com segurança
        self.on_m_changed()

    
    def on_m_changed(self):
        """Quando muda a orientação m."""
        m = self.slider_m.value()
        self.label_m.setText(f"m = {m:+d}")
        
        # Atualiza os painéis dependentes da orientação selecionada.
        if hasattr(self, 'rules_text'):
            self.update_filling_diagram()
        if hasattr(self, 'manual_table'):
            self.update_manual_panel()
        self.schedule_update()


    def on_iso_slider_moved(self, value):
        """Registra a alteração do limiar ou do brilho volumétrico."""
        if self.combo_mode.currentData() == 'density_volume':
            self.volume_brightness_percent = value
        else:
            self.iso_control_value = value
            self.iso_manually_set = True
        self.schedule_update()

    def populate_superposition_states(self, preferred=None):
        """Atualiza o estado B sem permitir que ele seja idêntico ao estado A."""
        if not hasattr(self, 'combo_superposition_state_b'):
            return
        state_a = self.selected_quantum_numbers()
        primary_changed = state_a != self.superposition_primary_state
        if primary_changed:
            self.superposition_primary_state = state_a
            self.superposition_phase = 0.0
            self.superposition_cache = None
            self.superposition_cache_key = None
        previous = (
            None if primary_changed
            else preferred or self.combo_superposition_state_b.currentData()
        )

        combo = self.combo_superposition_state_b
        combo.blockSignals(True)
        combo.clear()
        minimum_n = max(1, state_a[0] - 2)
        maximum_n = min(MAX_N, state_a[0] + 2)
        for n in range(minimum_n, maximum_n + 1):
            for l in range(n):
                for m in range(-l, l + 1):
                    state = (n, l, m)
                    if state != state_a:
                        combo.addItem(quantum_label(n, l, m), state)

        target = previous if previous != state_a else None
        available_default = (
            (2, 1, 0) if state_a[0] == 1
            else (state_a[0] - 1, 0, 0)
        )
        if target is None or not minimum_n <= target[0] <= maximum_n:
            target = available_default
        target_index = next(
            (
                index for index in range(combo.count())
                if combo.itemData(index) == target
            ),
            0,
        )
        if combo.count():
            combo.setCurrentIndex(target_index)
        combo.blockSignals(False)
        self.superposition_state_a_label.setText(
            f"A: {quantum_label(*state_a)}"
        )

    def current_superposition_context(self):
        """Retorna elétrons e configuração usados nas aproximações de energia."""
        atom = self.simulator.atom
        if self.interaction_mode == "Preenchimento manual" and self.manual_config:
            return (
                self.manual_config.electron_count,
                self.manual_config.subshell_occupancy(),
            )
        return atom.N_electrons, atom.get_subshell_occupancy()

    def prepare_superposition(self):
        """Prepara as partes espaciais e as escalas de energia da animação."""
        from physics.energy_levels import approximate_orbital_energy
        from physics.superposition import superposition_dynamics

        state_a = self.selected_quantum_numbers()
        state_b = self.combo_superposition_state_b.currentData()
        if state_b is None or state_a == state_b:
            return False
        electron_count, configuration = self.current_superposition_context()
        energy_a, z_eff_a = approximate_orbital_energy(
            self.simulator.atom.Z, state_a[0], state_a[1],
            electron_count=electron_count, configuration=configuration,
        )
        energy_b, z_eff_b = approximate_orbital_energy(
            self.simulator.atom.Z, state_b[0], state_b[1],
            electron_count=electron_count, configuration=configuration,
        )
        cache_key = (
            state_a, state_b, round(z_eff_a, 8), round(z_eff_b, 8),
            self.simulator.atom.Z, electron_count,
        )
        if cache_key != self.superposition_cache_key:
            orbital_a = Orbital(*state_a, electrons=1, Z_eff=z_eff_a)
            orbital_b = Orbital(*state_b, electrons=1, Z_eff=z_eff_b)
            animation_grid_size = min(
                72, 56 + 4 * max(0, max(state_a[0], state_b[0]) - 2)
            )
            self.superposition_cache = (
                self.simulator.renderer.prepare_superposition(
                    orbital_a, orbital_b, grid_size=animation_grid_size,
                )
            )
            self.superposition_cache_key = cache_key
        self.superposition_dynamics = superposition_dynamics(
            energy_a, energy_b,
        )
        return True

    @staticmethod
    def format_quantum_period(seconds):
        if not np.isfinite(seconds):
            return "∞"
        if seconds < 1e-15:
            return f"{seconds * 1e18:.2f} as"
        if seconds < 1e-12:
            return f"{seconds * 1e15:.2f} fs"
        if seconds < 1e-9:
            return f"{seconds * 1e12:.2f} ps"
        return f"{seconds:.3e} s"

    def update_superposition_status(self, *_):
        """Explica a escala física e distingue-a do ritmo visual."""
        if not hasattr(self, 'superposition_status'):
            return
        dynamics = self.superposition_dynamics
        speed = self.slider_superposition_speed.value() / 100.0
        if dynamics is None:
            self.superposition_status.setText(
                "Escolha dois estados para calcular a evolução temporal."
            )
            return
        state_b = self.combo_superposition_state_b.currentData()
        labels = (
            quantum_label(*self.selected_quantum_numbers()),
            quantum_label(*state_b) if state_b else "—",
        )
        weight_b = self.slider_superposition_weight.value()
        weight_a = 100 - weight_b
        if dynamics.is_stationary:
            evolution = (
                "Estados degenerados nesta aproximação: ΔE≈0 e a densidade "
                "permanece estacionária."
            )
        else:
            evolution = (
                f"ΔE={dynamics.delta_energy_ev:+.4f} eV • "
                f"f={dynamics.beat_frequency_hz:.3e} Hz • "
                f"T={self.format_quantum_period(dynamics.beat_period_s)} • "
                f"fase={np.degrees(self.superposition_phase) % 360:.0f}°"
            )
        self.superposition_status.setText(
            f"Estados: {labels[0]} + {labels[1]} &nbsp; | &nbsp; "
            f"pesos |c<sub>A</sub>|²={weight_a}% e "
            f"|c<sub>B</sub>|²={weight_b}% &nbsp; | &nbsp; "
            f"{evolution}<br><b>Ritmo visual desacelerado:</b> "
            f"{speed:.2f} ciclo/s; ele não é o tempo físico do elétron."
        )

    def on_superposition_weight_value_changed(self, value):
        """Atualiza a leitura do peso sem recalcular a malha durante o arraste."""
        self.label_superposition_weight.setText(f"{value}%")
        self.update_superposition_status()

    def on_superposition_speed_changed(self, value):
        """Exibe o ritmo visual escolhido junto ao respectivo controle."""
        self.label_superposition_speed.setText(
            f"{value / 100.0:.2f} ciclo/s".replace(".", ",")
        )
        self.update_superposition_status()

    def render_superposition_frame(self, reset_camera=False):
        """Atualiza um quadro de |Ψ(t)|² sem recalcular os orbitais base."""
        if self.superposition_rendering or not self.check_superposition.isChecked():
            return
        self.superposition_rendering = True
        try:
            if not self.prepare_superposition():
                return
            volume_mode = self.simulator.renderer.mode == 'density_volume'
            mesh = self.simulator.renderer.render_superposition(
                self.superposition_cache,
                weight_b=self.slider_superposition_weight.value() / 100.0,
                relative_phase_rad=self.superposition_phase,
                iso_value=self.slider_iso.value() / 100.0,
                as_volume=volume_mode,
                phase_coloring=(
                    volume_mode
                    and self.check_superposition_phase_colors.isChecked()
                ),
            )
            if mesh is None or mesh.n_points == 0:
                self.superposition_status.setText(
                    "A isosuperfície não apareceu com este limiar. "
                    "Reduza o controle de isosuperfície."
                )
                return
            if reset_camera:
                self.simulator.scene.clear_orbital_meshes()
            self.simulator.scene.add_orbital_mesh(
                mesh, "superposition_density", (0.24, 0.88, 1.0), 0.82,
                volume_brightness=self.simulator.renderer.volume_brightness,
            )
            if reset_camera:
                self.simulator.scene.reset_camera()
            else:
                self.simulator.scene.update()
            self.superposition_status_tick += 1
            if reset_camera or self.superposition_status_tick % 3 == 0:
                self.update_superposition_status()
            if reset_camera:
                self.update_phase_legend()
        finally:
            self.superposition_rendering = False

    def set_superposition_playing(self, playing):
        """Inicia ou pausa o relógio visual da superposição."""
        if playing and (
                not self.check_superposition.isChecked()
                or not self.prepare_superposition()
                or self.superposition_dynamics.is_stationary
        ):
            playing = False
        if playing:
            self.superposition_last_tick = time.monotonic()
            self.superposition_timer.start()
            self.btn_superposition_play.setText("Pausar")
        else:
            self.superposition_timer.stop()
            self.superposition_last_tick = None
            self.btn_superposition_play.setText("Reproduzir")
        self.update_superposition_status()

    def toggle_superposition_playback(self):
        self.set_superposition_playing(not self.superposition_timer.isActive())

    def advance_superposition(self):
        """Avança a fase em escala desacelerada e redesenha a densidade."""
        now = time.monotonic()
        elapsed = (
            0.12 if self.superposition_last_tick is None
            else min(0.5, now - self.superposition_last_tick)
        )
        self.superposition_last_tick = now
        visual_cycles_per_second = self.slider_superposition_speed.value() / 100.0
        self.superposition_phase = (
            self.superposition_phase
            + 2.0 * np.pi * visual_cycles_per_second * elapsed
        ) % (2.0 * np.pi)
        self.render_superposition_frame()

    def reset_superposition_phase(self):
        self.superposition_phase = 0.0
        self.render_superposition_frame()

    def on_superposition_weight_changed(self):
        self.render_superposition_frame()

    def on_superposition_phase_coloring_changed(self, _state):
        """Alterna entre densidade monocromática e fase relativa em cores."""
        if self.check_superposition.isChecked():
            self.render_superposition_frame()
            self.update_phase_legend()

    def on_superposition_state_b_changed(self):
        self.superposition_phase = 0.0
        self.superposition_cache = None
        self.superposition_cache_key = None
        if self.check_superposition.isChecked():
            self.render_superposition_frame(reset_camera=True)
            self.set_superposition_playing(True)
            self.update_info(*self.selected_quantum_numbers())

    def on_superposition_toggled(self, state):
        """Alterna entre um orbital estacionário e a dinâmica de dois estados."""
        enabled = state == Qt.Checked
        self.superposition_panel.setVisible(enabled)
        supported_modes = {'isosurface', 'density_volume'}
        for index in range(self.combo_mode.count()):
            item = self.combo_mode.model().item(index)
            if item is not None:
                item.setEnabled(
                    not enabled or self.combo_mode.itemData(index) in supported_modes
                )
        if enabled:
            self.combo_mode.blockSignals(True)
            density_index = self.combo_mode.findData('density_volume')
            self.combo_mode.setCurrentIndex(max(0, density_index))
            self.combo_mode.blockSignals(False)
            self.render_parameter_mode = 'density_volume'
            self.configure_render_parameter_control('density_volume')
            self.simulator.renderer.set_mode('density_volume')
            self.viewer_tabs.setCurrentWidget(self.viewer_3d_tab)
            self.populate_superposition_states()
            self.superposition_phase = 0.0
            self.superposition_cache = None
            self.superposition_cache_key = None
            self.render_superposition_frame(reset_camera=True)
            self.set_superposition_playing(True)
        else:
            self.set_superposition_playing(False)
            self.superposition_cache = None
            self.superposition_cache_key = None
            self.on_render_clicked()
        self.update_info(*self.selected_quantum_numbers())

    def on_mode_changed(self, index):
        """Quando muda o modo de renderização."""
        mode = self.combo_mode.itemData(index)
        if mode is None:
            return
        tooltip = self.combo_mode.itemData(index, Qt.ToolTipRole)
        if tooltip:
            self.combo_mode.setToolTip(tooltip)
        if self.render_parameter_mode == 'density_volume':
            self.volume_brightness_percent = self.slider_iso.value()
        else:
            self.iso_control_value = self.slider_iso.value()
        self.render_parameter_mode = mode
        self.configure_render_parameter_control(mode)
        self.check_superposition_phase_colors.setEnabled(
            mode == 'density_volume'
        )
        if self.check_superposition.isChecked():
            self.simulator.renderer.set_mode(mode)
            self.render_superposition_frame(reset_camera=True)
        else:
            # A própria ação abaixo renderiza o estado selecionado; chamar
            # Simulator.set_render_mode aqui faria uma renderização duplicada.
            self.simulator.renderer.set_mode(mode)
            self.on_render_clicked()

    def configure_render_parameter_control(self, mode):
        """Reutiliza o mesmo espaço para limiar de superfície ou brilho."""
        self.slider_iso.blockSignals(True)
        if mode == 'density_volume':
            self.label_render_parameter.setText("Brilho:")
            self.slider_iso.setRange(20, 200)
            self.slider_iso.setValue(self.volume_brightness_percent)
            self.label_iso.setText(f"{self.volume_brightness_percent}%")
            tip = (
                "Ganho visual da nuvem. Regiões brilhantes possuem maior |ψ|²; "
                "o efeito não representa emissão de luz."
            )
            self.simulator.set_volume_brightness(
                self.volume_brightness_percent / 100.0
            )
        else:
            self.label_render_parameter.setText("Isosuperfície:")
            self.slider_iso.setRange(0, 200)
            self.slider_iso.setValue(self.iso_control_value)
            self.label_iso.setText(f"{self.iso_control_value / 100.0:.3f}")
            tip = (
                "Limiar da isosuperfície. Valores menores mostram regiões mais "
                "difusas; valores maiores aproximam a superfície do núcleo."
            )
        self.slider_iso.setToolTip(tip)
        self.label_iso.setToolTip(tip)
        self.label_render_parameter.setToolTip(tip)
        self.slider_iso.blockSignals(False)
    
    def schedule_update(self):
        """Agenda uma atualização (debounce)."""
        self.timer_pending = True
        interval = (
            180 if hasattr(self, 'combo_mode')
            and self.combo_mode.currentData() == 'density_volume'
            else 100
        )
        self.update_timer.setInterval(interval)
        if not self.update_timer.isActive():
            self.update_timer.start()
    
    def on_slider_changed(self):
        """Timer callback para debouncing."""
        if self.timer_pending:
            self.on_render_clicked()
            self.timer_pending = False
            self.update_timer.stop()
    
    def on_render_clicked(self):
        """Renderiza o orbital selecionado."""
        # Descarta o corte associado à seleção anterior.
        if hasattr(self, 'slice_canvases'):
            self.clear_slice_figures()
        n = self.slider_n.value()
        l = self.slider_l.value()
        m = self.slider_m.value()
        
        # Validar m
        if abs(m) > l:
            m = 0
            self.slider_m.setValue(0)
        
        # Validar l
        if l >= n:
            l = n - 1
            self.slider_l.setValue(l)
        
        render_mode = self.combo_mode.currentData()
        if render_mode == 'density_volume':
            self.volume_brightness_percent = self.slider_iso.value()
            self.label_iso.setText(f"{self.volume_brightness_percent}%")
            self.simulator.set_volume_brightness(
                self.volume_brightness_percent / 100.0
            )
        else:
            iso_val = self.slider_iso.value() / 100.0
            self.iso_control_value = self.slider_iso.value()
            self.label_iso.setText(f"{iso_val:.3f}")
            # Uma escolha manual substitui o limiar adaptativo do renderizador.
            if self.iso_manually_set:
                self.simulator.set_iso_value(iso_val)
        
        # Renderizar o estado estacionário ou a combinação temporal ativa.
        if self.check_superposition.isChecked():
            self.populate_superposition_states(
                preferred=self.combo_superposition_state_b.currentData()
            )
            self.render_superposition_frame(reset_camera=True)
        else:
            self.visualize_orbital(n, l, m)
        
        # Atualizar info
        self.update_info(n, l, m)
        self.update_filling_diagram()
        self.update_radial_distribution()
        self.update_energy_diagram()

    def update_radial_distribution(self):
        """Sincroniza os gráficos radiais com o orbital selecionado."""
        if not hasattr(self, "radial_widget"):
            return
        n, l, m = self.selected_quantum_numbers()
        z_eff = self.effective_charge_for_state(n, l)
        occupied = self.selected_occupancy_orbital(n, l, m)
        electron_count = occupied.electrons if occupied else 0
        self.radial_widget.update_state(
            n=n,
            l=l,
            z_eff=z_eff,
            orbital_label=quantum_label(n, l, m),
            electron_count=electron_count,
            interaction_mode=self.interaction_mode,
        )

    def update_energy_diagram(self):
        """Sincroniza níveis e transições com a configuração exibida."""
        if not hasattr(self, 'energy_widget'):
            return
        atom = self.simulator.atom
        if self.interaction_mode == "Preenchimento manual" and self.manual_config:
            configuration = self.manual_config.subshell_occupancy()
            electron_count = self.manual_config.electron_count
            orbitals = self.manual_config.orbitals
        else:
            configuration = atom.get_subshell_occupancy()
            electron_count = atom.N_electrons
            orbitals = atom.orbitals
        occupied_states = [
            (orbital.n, orbital.l, orbital.m, spin)
            for orbital in orbitals
            for spin in orbital.electron_spins
        ]
        self.energy_widget.update_state(
            atom.Z,
            electron_count,
            configuration,
            self.selected_quantum_numbers(),
            atom.ion_label,
            occupied_states=occupied_states,
        )
    
    def on_slice2d_clicked(self):
        """
        Mostra (ou atualiza) o corte na aba 2D da região central, usando o
        orbital selecionado e o plano escolhido. A cena 3D é preservada.
        """
        n = self.slider_n.value()
        l = self.slider_l.value()
        m = self.slider_m.value()
        if abs(m) > l:
            m = 0
        plane = self.combo_plane.currentText()

        Z_eff = self.effective_charge_for_state(n, l)

        occupied = self.selected_occupancy_orbital(n, l, m)
        electron_count = occupied.electrons if occupied else 0
        if self.interaction_mode != "Explorar orbitais" and electron_count == 0:
            self.clear_slice_figures("orbital vazio — densidade eletrônica zero")
            self.viewer_tabs.setCurrentWidget(self.viewer_2d_tab)
            self.label_description.setText(
                "Orbital vazio: o corte físico possui densidade eletrônica zero. "
                "Use 'Explorar orbitais' para visualizar sua forma matemática."
            )
            return

        orbital = Orbital(
            n=n, l=l, m=m, electrons=electron_count, Z_eff=Z_eff
        )

        try:
            figures = self.simulator.renderer.render_slice_2d_panels(orbital, plane=plane)
        except Exception as e:
            print(f"⚠ Erro ao gerar corte 2D: {e}")
            return

        self.slice_status.setText(
            f"Corte do orbital {quantum_label(n, l, m)} no plano {plane}."
        )
        self.show_slice_figures(figures)
        self.viewer_tabs.setCurrentWidget(self.viewer_2d_tab)

    def show_slice_figures(self, figures):
        """Mostra amplitude e probabilidade nas subabas da área 2D."""
        self._set_slice_canvas('amplitude', figures['amplitude'], self.slice_amp_layout)
        self._set_slice_canvas('probability', figures['probability'], self.slice_prob_layout)

    def _set_slice_canvas(self, key, fig, layout):
        old_canvas = self.slice_canvases.get(key)
        if old_canvas is not None:
            layout.removeWidget(old_canvas)
            old_canvas.setParent(None)
            plt.close(old_canvas.figure)

        placeholder = self.slice_placeholders.get(key)
        if placeholder is not None:
            placeholder.hide()

        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)
        canvas.draw()
        self.slice_canvases[key] = canvas

    def clear_slice_figures(self, reason=None):
        """Remove cortes antigos para não associá-los ao orbital atual."""
        for key, canvas in list(self.slice_canvases.items()):
            if canvas is None:
                continue
            layout = self.slice_amp_layout if key == 'amplitude' else self.slice_prob_layout
            layout.removeWidget(canvas)
            canvas.setParent(None)
            plt.close(canvas.figure)
            self.slice_canvases[key] = None
        message = reason or "Seleção alterada. Gere novamente o corte 2D."
        self.slice_status.setText(message[0].upper() + message[1:] + ".")
        for key, placeholder in self.slice_placeholders.items():
            quantity = "amplitude ψ" if key == 'amplitude' else "probabilidade |ψ|²"
            placeholder.setText(f"Sem {quantity} para a seleção atual.")
            placeholder.show()

    def on_show_filled(self):
        """Mostra os orbitais preenchidos."""
        if self.check_superposition.isChecked():
            self.check_superposition.setChecked(False)
        if self.interaction_mode == "Preenchimento manual" and self.manual_config:
            self.show_manual_filled()
            self.update_manual_panel()
            self.info_tabs.setCurrentWidget(self.manual_tab)
        else:
            self.simulator.update_atom_display()
            self.update_info_filled()
        self.update_phase_legend(multiple=True)

    def on_rules_clicked(self):
        """Abre o diagnóstico visual de Aufbau, Hund e Pauli."""
        self.update_filling_diagram()
        self.info_tabs.setCurrentWidget(self.rules_text)

    def selected_quantum_numbers(self):
        return self.slider_n.value(), self.slider_l.value(), self.slider_m.value()

    def selected_occupancy_orbital(self, n=None, l=None, m=None):
        if n is None:
            n, l, m = self.selected_quantum_numbers()
        if self.interaction_mode == "Preenchimento manual" and self.manual_config:
            return self.manual_config.get_orbital(n, l, m)
        return self.simulator.atom.get_orbital_by_quantum_numbers(n, l, m)

    def effective_charge_for_state(self, n, l):
        """Retorna Z_eff usando a configuração exibida na interface."""
        from physics.screening import orbital_state_effective_charge

        atom = self.simulator.atom
        if self.interaction_mode == "Preenchimento manual" and self.manual_config:
            configuration = self.manual_config.subshell_occupancy()
            electron_count = self.manual_config.electron_count
        else:
            configuration = atom.get_subshell_occupancy()
            electron_count = atom.N_electrons
        return orbital_state_effective_charge(
            atom.Z, n, l,
            electron_count=electron_count,
            configuration=configuration,
        )

    def select_quantum_numbers(self, n, l, m):
        """Sincroniza uma caixa clicada com os três controles quânticos."""
        for slider in (self.slider_n, self.slider_l, self.slider_m):
            slider.blockSignals(True)
        try:
            self.slider_n.setValue(n)
            self.slider_l.setRange(0, n - 1)
            self.slider_l.setValue(l)
            self.slider_m.setRange(-l, l)
            self.slider_m.setValue(m)
            from orbitals.orbital_types import get_orbital_type
            self.label_n.setText(f"n = {n}")
            self.label_l.setText(f"l = {l} ({get_orbital_type(l).letter})")
            self.label_m.setText(f"m = {m:+d}")
        finally:
            for slider in (self.slider_n, self.slider_l, self.slider_m):
                slider.blockSignals(False)
        self.update_manual_panel()
        self.update_filling_diagram()
        self.on_render_clicked()

    def on_manual_cell_clicked(self, row, column):
        item = self.manual_table.item(row, column)
        quantum_numbers = item.data(Qt.UserRole) if item else None
        if quantum_numbers:
            self.select_quantum_numbers(*quantum_numbers)

    def _finish_manual_action(self, result):
        details = [result.message]
        details.extend(result.warnings)
        self.manual_feedback = "\n".join(details)
        self.update_manual_panel()
        self.update_filling_diagram()
        self.on_render_clicked()
        self.info_tabs.setCurrentWidget(self.manual_tab)

    def on_manual_add_electron(self, spin):
        if not self.manual_config or self.interaction_mode != "Preenchimento manual":
            return
        n, l, m = self.selected_quantum_numbers()
        self._finish_manual_action(self.manual_config.add_electron(n, l, m, spin))

    def on_manual_remove_electron(self):
        if not self.manual_config or self.interaction_mode != "Preenchimento manual":
            return
        n, l, m = self.selected_quantum_numbers()
        self._finish_manual_action(self.manual_config.remove_electron(n, l, m))

    def on_show_more_levels(self):
        if not self.manual_config or self.interaction_mode != "Preenchimento manual":
            return
        self.manual_extra_levels += 2
        self.manual_feedback = (
            "Novos níveis de maior energia foram exibidos. "
            "Clique em uma caixa para selecioná-la."
        )
        self.update_manual_panel()
        self.update_filling_diagram()
        self.info_tabs.setCurrentWidget(self.manual_tab)

    def on_manual_move_electron(self):
        if not self.manual_config or self.interaction_mode != "Preenchimento manual":
            return
        source = self.combo_move_source.currentData()
        target = self.selected_quantum_numbers()
        if source is None:
            self.manual_feedback = (
                "Não há outro orbital ocupado para usar como origem. "
                "Adicione primeiro um elétron."
            )
            self.update_manual_panel()
            return
        self._finish_manual_action(
            self.manual_config.move_electron(source, target)
        )

    def on_manual_undo(self):
        if not self.manual_config or self.interaction_mode != "Preenchimento manual":
            return
        self.manual_feedback = (
            "Última operação desfeita."
            if self.manual_config.undo() else "Não há operações para desfazer."
        )
        self.update_manual_panel()
        self.update_filling_diagram()
        self.on_render_clicked()

    def on_manual_reset(self):
        if not self.manual_config or self.interaction_mode != "Preenchimento manual":
            return
        self.manual_config.reset()
        self.manual_feedback = "Todos os orbitais foram esvaziados."
        self.update_manual_panel()
        self.update_filling_diagram()
        self.on_render_clicked()

    def on_manual_auto_fill(self):
        if not self.manual_config or self.interaction_mode != "Preenchimento manual":
            return
        self.manual_config.fill_ground_state()
        self.manual_feedback = "Configuração fundamental preenchida automaticamente."
        self.update_manual_panel()
        self.update_filling_diagram()
        self.on_render_clicked()

    def on_manual_verify(self):
        if not self.manual_config or self.interaction_mode != "Preenchimento manual":
            return
        checks = self.manual_config.validate_rules()
        mark = lambda valid: "✓" if valid else "⚠"
        self.manual_feedback = (
            f"{self.manual_config.result_description()}\n"
            f"Aufbau {mark(checks['aufbau'])}  "
            f"Hund {mark(checks['hund'])}  "
            f"Pauli {mark(checks['pauli'])}"
        )
        self.update_manual_panel()
        self.update_filling_diagram()
        self.info_tabs.setCurrentWidget(self.manual_tab)
    
    def on_sequence_clicked(self):
        """Mostra a sequência de Aufbau."""
        from physics.screening import get_orbital_sequence
        from utils.helpers import quantum_label
        
        seq = get_orbital_sequence()
        info = "SEQUÊNCIA DE AUFBAU (Ordem de Preenchimento)\n"
        info += "=" * 50 + "\n\n"
        
        count = 0
        for n, l in seq[:20]:
            label = quantum_label(n, l)
            max_e = 2 * (2*l + 1)
            count += 1
            info += f"{count:2d}. {label:5s}  (máx {max_e:2d} e⁻)\n"
            
            if count == 10:
                info += "\n"
        
        self.text_info.setText(info)
        self.info_tabs.setCurrentWidget(self.data_tab)
    
    # RENDERIZAÇÃO DE ORBITAIS

    @staticmethod
    def _color_to_hex(color):
        """Converte uma cor RGB normalizada para hexadecimal."""
        values = [max(0, min(255, round(component * 255))) for component in color]
        return "#{:02x}{:02x}{:02x}".format(*values[:3])

    def update_phase_legend(self, color=None, electron_count=None, multiple=False):
        """Explica a convenção de cores da representação 3D atual."""
        if not hasattr(self, 'phase_legend'):
            return

        render_mode = self.combo_mode.currentData() if hasattr(self, 'combo_mode') else "isosurface"
        if (
                hasattr(self, 'check_superposition')
                and self.check_superposition.isChecked()
        ):
            if render_mode == 'density_volume':
                phase_colors = (
                    hasattr(self, 'check_superposition_phase_colors')
                    and self.check_superposition_phase_colors.isChecked()
                )
                if phase_colors:
                    self.phase_legend.setText(
                        "<b>SUPERPOSIÇÃO — DENSIDADE E FASE RELATIVA</b> &nbsp; "
                        "brilho = |Ψ|²; cor = arg(Ψ), com A como referência. "
                        "<span style='color:#19ffff'>● 0</span> &nbsp; "
                        "<span style='color:#8c19ff'>● π/2</span> &nbsp; "
                        "<span style='color:#ff1919'>● π</span> &nbsp; "
                        "<span style='color:#8cff19'>● 3π/2</span>. "
                        "O brilho não representa emissão de luz."
                    )
                else:
                    self.phase_legend.setText(
                        "<b>SUPERPOSIÇÃO — NUVEM |Ψ(t)|²</b> &nbsp; "
                        "mais brilho = maior densidade de probabilidade; a cor "
                        "ciano é apenas uma convenção visual. A animação está "
                        "desacelerada."
                    )
            else:
                self.phase_legend.setText(
                    "<b>SUPERPOSIÇÃO TEMPORAL — |Ψ(t)|²</b> &nbsp; "
                    "A superfície ciano representa densidade de probabilidade. "
                    "A mudança de forma vem da interferência entre A e B; "
                    "a animação está desacelerada."
                )
            return

        if electron_count == 0 and self.interaction_mode != "Explorar orbitais":
            self.phase_legend.setText(
                "<b>LEGENDA 3D</b> &nbsp; Orbital vazio: nenhuma densidade eletrônica é exibida."
            )
            return

        if render_mode == 'density_volume':
            empty_note = (
                " Orbital vazio: trata-se apenas da forma matemática disponível."
                if electron_count == 0 else ""
            )
            self.phase_legend.setText(
                "<b>NUVEM DE DENSIDADE</b> &nbsp; brilho ∝ |ψ|²; as cores "
                "representam a fase de ψ. O brilho não é emissão de luz."
                f"{empty_note}"
            )
            return

        if render_mode == "points":
            if electron_count == 0:
                empty_hex = self._color_to_hex(ORBITAL_EMPTY_POSITIVE_COLOR)
                self.phase_legend.setText(
                    "<b>ORBITAL VAZIO — PRÉVIA MATEMÁTICA</b> &nbsp; "
                    f"<span style='color:{empty_hex}'>● |ψ|²</span> &nbsp; "
                    "A cor é um destaque visual; não representa um elétron presente."
                )
                return
            color_note = "as cores identificam os tipos orbitais" if multiple else "a cor identifica o tipo orbital"
            self.phase_legend.setText(
                f"<b>LEGENDA 3D</b> &nbsp; Nuvem de probabilidade |ψ|²: {color_note}; "
                "o sinal da fase não é representado."
            )
            return

        negative_color = (
            ORBITAL_EMPTY_NEGATIVE_COLOR
            if electron_count == 0 else ORBITAL_NEGATIVE_PHASE_COLOR
        )
        negative_hex = self._color_to_hex(negative_color)
        if multiple:
            self.phase_legend.setText(
                "<b>LEGENDA 3D</b> &nbsp; "
                "<span style='color:#33ccff'>● s</span> &nbsp; "
                "<span style='color:#ff8c00'>● p</span> &nbsp; "
                "<span style='color:#cc33ff'>● d</span> &nbsp; "
                "<span style='color:#33ff66'>● f</span> &nbsp; "
                f"= ψ positivo &nbsp; <span style='color:{negative_hex}'>● branco</span> = ψ negativo"
            )
            return

        positive_color = color or ORBITAL_EMPTY_POSITIVE_COLOR
        positive_hex = self._color_to_hex(positive_color)
        empty_note = (
            " &nbsp; <b>ORBITAL VAZIO — prévia matemática, sem elétron.</b>"
            if electron_count == 0 else ""
        )
        self.phase_legend.setText(
            "<b>LEGENDA 3D — fase de ψ</b> &nbsp; "
            f"<span style='color:{positive_hex}'>● ψ &gt; 0</span> &nbsp; "
            f"<span style='color:{negative_hex}'>● ψ &lt; 0</span>"
            f"{empty_note}"
        )
    
    def visualize_orbital(self, n: int, l: int, m: int = 0):
        """Renderiza um orbital específico (n, l, m)."""
        Z_eff = self.effective_charge_for_state(n, l)

        occupied = self.selected_occupancy_orbital(n, l, m)
        electron_count = occupied.electrons if occupied else 0

        if self.interaction_mode != "Explorar orbitais" and electron_count == 0:
            self.simulator.scene.clear_orbital_meshes()
            self.update_phase_legend(electron_count=0)
            self.slice_status.setText(
                "Orbital vazio: o corte 2D possui densidade eletrônica zero."
            )
            self.simulator.scene.update()
            return

        orbital = Orbital(
            n=n, l=l, m=m, electrons=electron_count, Z_eff=Z_eff
        )
        orbital_label = quantum_label(n, l, m)
        if electron_count == 0:
            orbital_label += "_forma_vazia"

        # Mantém o ator quando o estado não mudou. No modo volumétrico isso
        # permite atualizar apenas os voxels RGBA em vez de reconstruir o VTK.
        for existing_id in list(self.simulator.scene.orbital_meshes):
            if existing_id != orbital_label:
                self.simulator.scene.remove_orbital_mesh(existing_id)
        
        # Renderiza
        mesh = self.simulator.renderer.render_orbital(orbital)
        
        # Adiciona na cena
        color = getattr(orbital, 'color', (0.2, 0.8, 1.0))
        opacity = 0.68 if electron_count == 0 else 0.8
        negative_color = None
        if electron_count == 0:
            color = ORBITAL_EMPTY_POSITIVE_COLOR
            negative_color = ORBITAL_EMPTY_NEGATIVE_COLOR
        self.update_phase_legend(color=color, electron_count=electron_count)
        self.simulator.scene.add_orbital_mesh(
            mesh, orbital_label, color, opacity,
            negative_color=negative_color,
            volume_brightness=self.simulator.renderer.volume_brightness,
        )
        
        # Enquadra a malha realmente gerada; o alcance teórico pode conter
        # regiões onde a amplitude já é desprezível e afastar demais a câmera.
        try:
            self.simulator.scene.reset_camera()
        except:
            range_max = self.simulator.renderer._get_range_for_orbital(orbital)
            self.simulator.scene.set_camera_for_range(range_max)
        self.simulator.scene.update()

    def show_manual_filled(self):
        """Renderiza apenas os orbitais ocupados na construção do usuário."""
        from physics.screening import orbital_state_effective_charge

        self.simulator.scene.clear_orbital_meshes()
        max_range = 1.0
        configuration = self.manual_config.subshell_occupancy()
        for source in self.manual_config.orbitals:
            if source.electrons == 0:
                continue
            Z_eff = orbital_state_effective_charge(
                self.simulator.atom.Z, source.n, source.l,
                electron_count=self.manual_config.electron_count,
                configuration=configuration,
            )
            orbital = Orbital(
                n=source.n, l=source.l, m=source.m,
                electrons=source.electrons, Z_eff=Z_eff,
            )
            mesh = self.simulator.renderer.render_orbital(orbital)
            label = f"manual_{quantum_label(source.n, source.l, source.m)}"
            self.simulator.scene.add_orbital_mesh(
                mesh, label, orbital.color,
                0.75 if source.electrons == 2 else 0.55,
                volume_brightness=self.simulator.renderer.volume_brightness,
            )
            max_range = max(
                max_range,
                self.simulator.renderer._get_range_for_orbital(orbital),
            )
        self.simulator.scene.set_camera_for_range(max_range)
        self.simulator.scene.update()
        self.update_phase_legend(multiple=True)

    
    def update_limits(self):
        """
        Seleciona a camada ocupada mais externa e sincroniza os limites de l e m.
        """
        if not self.simulator.atom:
            return

        # Evita renderizações intermediárias enquanto os controles são sincronizados.
        self.slider_n.blockSignals(True)
        self.slider_l.blockSignals(True)
        self.slider_m.blockSignals(True)

        try:
            # Nível principal mais externo que possui elétrons.
            max_n_occupied = max(
                (orb.n for orb in self.simulator.atom.orbitals if orb.electrons > 0),
                default=1
            )

            # Garante que não ultrapasse MAX_N (definido no config)
            if max_n_occupied > MAX_N:
                max_n_occupied = MAX_N

            # Seleciona o nível principal externo.
            self.slider_n.setValue(max_n_occupied)
            self.label_n.setText(f"n = {max_n_occupied}")

            # Reinicia o tipo de orbital em s.
            max_l = max_n_occupied - 1
            self.slider_l.setRange(0, max_l)
            self.slider_l.setValue(0)
            from orbitals.orbital_types import get_orbital_type
            self.label_l.setText(f"l = 0 ({get_orbital_type(0).letter})")

            # Para l=0, a única orientação permitida é m=0.
            self.slider_m.setRange(0, 0)   # l=0 → m só pode ser 0
            self.slider_m.setValue(0)
            self.label_m.setText("m = +0")

        finally:
            # Restaura os sinais mesmo se a sincronização falhar.
            self.slider_n.blockSignals(False)
            self.slider_l.blockSignals(False)
            self.slider_m.blockSignals(False)

        # Renderiza uma única vez após validar todos os controles.
        self.on_render_clicked()

    
    # ATUALIZAÇÃO DE INFORMAÇÕES

    def update_quantum_state_card(self, n, l, m, orbital):
        """Mostra os quatro números quânticos de cada elétron selecionado."""
        spins = orbital.electron_spins if orbital is not None else ()
        base = f"n={n}, l={l}, mₗ={m:+d}"
        if not spins:
            detail = (
                f"<b>Orbital vazio:</b> ({base}). mₛ não se aplica enquanto "
                "não houver elétron; os estados disponíveis são +½ e −½."
            )
        else:
            states = []
            for index, spin in enumerate(spins, start=1):
                spin_text = "+½" if spin > 0 else "−½"
                states.append(
                    f"Elétron {index}: ({base}, mₛ={spin_text})"
                )
            detail = "<br>".join(states)
        self.quantum_state_card.setText(
            f"<b>ESTADO QUÂNTICO SELECIONADO</b><br>{detail}"
        )
    
    def update_info(self, n: int, l: int, m: int):
        """Atualiza o painel de informações."""
        from orbitals.orbital_types import get_orbital_type, ORBITAL_TYPES
        
        # Nome do orbital
        orbital_type = get_orbital_type(l)
        orbital_name = quantum_label(n, l, m)
        occupied_orbital = self.selected_occupancy_orbital(n, l, m)
        electron_count = occupied_orbital.electrons if occupied_orbital else 0
        occupancy_label = "VAZIO" if electron_count == 0 else f"{electron_count} e⁻"
        superposition_enabled = (
            hasattr(self, 'check_superposition')
            and self.check_superposition.isChecked()
            and hasattr(self, 'combo_superposition_state_b')
        )
        state_b = (
            self.combo_superposition_state_b.currentData()
            if superposition_enabled else None
        )
        state_b_name = quantum_label(*state_b) if state_b else None
        if state_b_name:
            self.label_orbital_name.setText(
                f"{orbital_name} + {state_b_name} — |Ψ(t)|²"
            )
            self.quantum_state_card.setText(
                "<b>SUPERPOSIÇÃO TEMPORAL</b><br>"
                f"Estado A: {orbital_name} (n={n}, l={l}, m={m:+d})<br>"
                f"Estado B: {state_b_name} "
                f"(n={state_b[0]}, l={state_b[1]}, m={state_b[2]:+d})<br>"
                "Estado espacial preparado de um elétron; a ocupação do átomo "
                "não foi modificada."
            )
        else:
            self.label_orbital_name.setText(f"{orbital_name} — {occupancy_label}")
            self.update_quantum_state_card(n, l, m, occupied_orbital)
        
        # Informações detalhadas
        orbital_type = get_orbital_type(l)
        Z_eff = self.effective_charge_for_state(n, l)
        
        if state_b_name:
            info = "Visualização: superposição temporal |Ψ(t)|²\n"
            info += f"Estado preparado: {orbital_name} + {state_b_name}\n"
            info += (
                f"Ocupação de {orbital_name} na configuração: "
                f"{occupancy_label}\n"
            )
            info += "A construção visual não altera a configuração eletrônica.\n"
        else:
            info = f"Modo: {self.interaction_mode}\n"
            info += f"Estado do orbital: {occupancy_label}\n"
        if not state_b_name:
            if self.interaction_mode == "Explorar orbitais" and electron_count == 0:
                info += "Visualização: forma matemática de um estado disponível.\n"
                info += "Densidade eletrônica real deste orbital: zero.\n"
            elif self.interaction_mode != "Explorar orbitais" and electron_count == 0:
                info += "Visualização 3D: ocultada porque o orbital está vazio.\n"
            elif self.interaction_mode == "Preenchimento manual":
                info += "Ocupação definida pela construção do usuário.\n"
            else:
                info += "Ocupação pertencente à configuração fundamental.\n"

        info += f"\nNúmeros Quânticos:\n"
        info += f"  n (principal)  = {n}    (nível de energia)\n"
        info += f"  l (azimutal)   = {l}    (tipo: {orbital_type.letter})\n"
        info += f"  m (magnético)  = {m:+d}  (orientação)\n"
        spin_symbols = (
            occupied_orbital.spin_symbols if electron_count else "vazio"
        )
        info += f"  mₛ (spin)      = ±½   (ocupação: {spin_symbols})\n"
        info += f"                    veja a aba 'Regras e spins'\n\n"
        
        info += f"Parâmetros do Elemento:\n"
        info += f"  Z (nuclear)    = {self.simulator.atom.Z}\n"
        info += f"  Carga          = {self.simulator.atom.charge:+d}\n"
        info += f"  Elétrons       = {self.simulator.atom.N_electrons}\n"
        info += f"  Z_eff          = {Z_eff:.2f}\n"
        info += f"  Espécie        = {self.simulator.atom.ion_label}\n\n"
        
        info += f"Características:\n"
        info += f"  {orbital_type.description}\n"
        info += f"  Capacidade: {orbital_type.max_electrons} e⁻ (subnível)\n"
        info += f"  Degeneração: {orbital_type.degeneracy} orbitais\n"
        
        self.text_info.setText(info)
        
        # Descrição
        descriptions = {
            (1, 0): "🔵 Esférico perfeito, menor e mais energético.",
            (2, 0): "🔵 Esférico com nó radial interno.",
            (2, 1): "🥁 Haltere em 3 orientações (x, y, z).",
            (3, 0): "🔵 Esférico com 2 nós radiais.",
            (3, 1): "🥁 Halteres maiores e mais afastados.",
            (3, 2): "🍀 Trevos com 4-5 lobos complexos.",
            (4, 0): "🔵 Esférico com 3 nós radiais.",
            (4, 1): "🥁 Halteres ainda maiores.",
        }
        
        if state_b_name:
            desc = (
                f"Superposição temporal {orbital_name} + {state_b_name}. "
                "A superfície mostra |Ψ(t)|²; a configuração eletrônica do "
                "átomo permanece inalterada."
            )
        else:
            desc = descriptions.get((n, l), "Orbital de alta energia")
            if electron_count == 0:
                desc += " Estado atualmente vazio."
        self.label_description.setText(desc)
    
    def update_info_filled(self):
        """Atualiza info mostrando configuração eletrônica."""
        atom = self.simulator.atom
        config = atom.get_electron_config()
        
        info = f"CONFIGURAÇÃO ELETRÔNICA\n"
        info += f"{'=' * 50}\n\n"
        info += f"Espécie: {atom.get_element_name()} ({atom.ion_label})\n"
        info += f"Z = {atom.Z} | Carga = {atom.charge:+d} | Elétrons = {atom.N_electrons}\n\n"
        info += f"Configuração:\n{config}\n\n"
        info += f"Elétrons de valência: {atom.get_valence_electrons()}\n\n"
        info += f"Orbitais preenchidos:\n"
        
        for orbital in atom.orbitals:
            if orbital.electrons > 0:
                label = quantum_label(orbital.n, orbital.l, orbital.m)
                info += f"  {label:5s} — {orbital.electrons} e⁻ [{orbital.spin_symbols}]\n"
        
        self.text_info.setText(info)
        self.info_tabs.setCurrentWidget(self.data_tab)

    def update_filling_diagram(self):
        """Atualiza o diagrama orbital e o resultado das três verificações."""
        atom = self.simulator.atom
        if atom is None or not hasattr(self, 'rules_text'):
            return

        n, l, m = self.selected_quantum_numbers()
        mark = lambda valid: "✓" if valid else "⚠"

        if self.interaction_mode == "Preenchimento manual" and self.manual_config:
            checks = self.manual_config.validate_rules()
            selected = self.manual_config.get_orbital(n, l, m)
            selected_count = selected.electrons if selected else 0
            info = (
                f"MODO DE PREENCHIMENTO MANUAL — {atom.ion_label}\n"
                f"Elétrons: {self.manual_config.electron_count}/"
                f"{self.manual_config.target_electrons}\n"
                f"Configuração: {self.manual_config.configuration_string()}\n\n"
                f"{mark(checks['aufbau'])} AUFBAU — ordem de menor energia\n"
                f"{mark(checks['hund'])} HUND   — ocupação simples antes dos pares\n"
                f"{mark(checks['pauli'])} PAULI  — spins distintos, máximo de dois\n\n"
                f"Selecionado: {quantum_label(n, l, m)} | "
                f"{selected_count} e⁻ [{selected.spin_symbols if selected_count else 'vazio'}]\n\n"
                f"{self.manual_diagram_text()}\n\n"
                f"{self.manual_config.result_description()}\n"
                "Legenda: ↑ mₛ=+½   ↓ mₛ=-½"
            )
            self.rules_text.setPlainText(info)
            return

        checks = atom.validate_filling_rules()
        selected = atom.get_orbital_by_quantum_numbers(n, l, m)
        selected_count = selected.electrons if selected else 0
        selected_state = (
            f"{selected_count} e⁻ [{selected.spin_symbols}]"
            if selected_count else "vazio"
        )
        exception_note = (
            "\nNota: exceção energética da configuração fundamental aplicada.\n"
            if atom.has_configuration_exception else ""
        )
        exploration_note = (
            "Forma matemática disponível; não contribui para a densidade real.\n"
            if self.interaction_mode == "Explorar orbitais" and selected_count == 0
            else ""
        )
        info = (
            f"{atom.get_element_name()} ({atom.ion_label}) — "
            f"Z = {atom.Z}, carga = {atom.charge:+d}, elétrons = {atom.N_electrons}\n"
            f"Configuração: {atom.get_electron_config()}\n"
            f"{exception_note}\n"
            f"{mark(checks['aufbau'])} AUFBAU — subníveis ocupados na ordem de energia\n"
            f"{mark(checks['hund'])} HUND   — primeiro um ↑ em cada orbital degenerado\n"
            f"{mark(checks['pauli'])} PAULI  — no máximo ↑↓ por orbital\n\n"
            f"SELECIONADO: {quantum_label(n, l, m)} — {selected_state}\n"
            f"{exploration_note}\n"
            "DIAGRAMA DE ORBITAIS\n"
            "Cada caixa representa um valor de mₗ.\n\n"
            f"{atom.get_orbital_diagram()}\n\n"
            "Legenda: ↑ mₛ=+½   ↓ mₛ=-½"
        )
        self.rules_text.setPlainText(info)

    def manual_diagram_text(self):
        """Diagrama textual da configuração editável, incluindo caixas vazias."""
        if not self.manual_config:
            return ""
        from orbitals.orbital_types import get_orbital_type

        selected_numbers = self.selected_quantum_numbers()
        lines = []
        selected_subshell = self.selected_quantum_numbers()[:2]
        for n, l in self.manual_config.visible_subshells(
                selected_subshell, self.manual_extra_levels):
            orbitals = [
                orbital for orbital in self.manual_config.orbitals
                if orbital.n == n and orbital.l == l
            ]
            total = sum(orbital.electrons for orbital in orbitals)
            boxes = []
            for orbital in orbitals:
                box = f"[{orbital.spin_symbols:<2}]"
                if (orbital.n, orbital.l, orbital.m) == selected_numbers:
                    box = f"<{box}>"
                boxes.append(box)
            label = f"{n}{get_orbital_type(l).letter}{total}"
            lines.append(f"{label:<5} {' '.join(boxes)}")
            if len(orbitals) > 1:
                magnetic = " ".join(f"{orbital.m:+d}".center(4) for orbital in orbitals)
                lines.append(f"{'mₗ':<5} {magnetic}")
        return "\n".join(lines)

    def update_manual_panel(self):
        """Reconstrói as caixas clicáveis e os contadores do modo manual."""
        if not hasattr(self, 'manual_table'):
            return

        active = self.interaction_mode == "Preenchimento manual"
        for button in self.manual_buttons:
            button.setEnabled(active)
        self.manual_table.setEnabled(active)

        if not active or not self.manual_config:
            self.manual_table.setRowCount(0)
            self.combo_move_source.clear()
            self.btn_move_electron.setEnabled(False)
            self.manual_status.setProperty("state", "inactive")
            self.manual_status.style().unpolish(self.manual_status)
            self.manual_status.style().polish(self.manual_status)
            self.manual_status.setText(
                "Selecione o modo 'Preenchimento manual' para começar com os "
                "orbitais vazios e inserir elétrons."
            )
            return

        from orbitals.orbital_types import get_orbital_type
        from physics.screening import get_orbital_sequence

        checks = self.manual_config.validate_rules()
        mark = lambda valid: "✓" if valid else "⚠"
        if self.manual_config.target_electrons == 0:
            state_banner = f"{self.simulator.atom.ion_label} — ESPÉCIE SEM ELÉTRONS"
            banner_style = "inactive"
        elif self.manual_config.is_complete and checks["ground_state"]:
            state_banner = "ESTADO FUNDAMENTAL CORRETO ✓"
            banner_style = "ground"
        elif self.manual_config.is_complete:
            state_banner = "ESTADO EXCITADO PERMITIDO ⚠"
            banner_style = "excited"
        else:
            state_banner = "CONFIGURAÇÃO EM CONSTRUÇÃO"
            banner_style = "building"
        status = (
            f"{state_banner}\n"
            f"Elétrons: {self.manual_config.electron_count}/"
            f"{self.manual_config.target_electrons} | "
            f"Restantes: {self.manual_config.remaining_electrons}\n"
            f"Configuração: {self.manual_config.configuration_string()}\n"
            f"Aufbau {mark(checks['aufbau'])}  Hund {mark(checks['hund'])}  "
            f"Pauli {mark(checks['pauli'])}"
        )
        if self.manual_feedback:
            status += f"\n{self.manual_feedback}"
        self.manual_status.setProperty("state", banner_style)
        self.manual_status.style().unpolish(self.manual_status)
        self.manual_status.style().polish(self.manual_status)
        self.manual_status.setText(status)
        can_add = self.manual_config.remaining_electrons > 0
        self.btn_spin_up.setEnabled(can_add)
        self.btn_spin_down.setEnabled(can_add)
        self.btn_auto_fill.setEnabled(self.manual_config.target_electrons > 0)
        if self.manual_config.target_electrons == 0:
            add_tooltip = (
                "Esta espécie possui zero elétrons. Escolha uma carga com "
                "elétrons disponíveis."
            )
        elif not can_add:
            add_tooltip = (
                "Todos os elétrons já foram colocados. Remova ou mova um "
                "elétron para alterar a configuração."
            )
        else:
            add_tooltip = "Adiciona um elétron ao orbital selecionado."
        self.btn_spin_up.setToolTip(add_tooltip)
        self.btn_spin_down.setToolTip(add_tooltip)
        selected_subshell = self.selected_quantum_numbers()[:2]
        visible = set(self.manual_config.visible_subshells(
            selected_subshell, self.manual_extra_levels
        ))
        all_subshells = get_orbital_sequence()
        more_available = len(visible) < len(all_subshells)
        self.btn_show_levels.setEnabled(more_available)
        self.btn_show_levels.setText(
            "Mostrar próximos níveis de energia"
            if more_available else "Todos os níveis disponíveis estão visíveis"
        )
        orbitals = [
            orbital for orbital in self.manual_config.orbitals
            if (orbital.n, orbital.l) in visible
        ]
        self.manual_table.setRowCount(len(orbitals))
        selected = self.selected_quantum_numbers()
        energy_order = {
            subshell: index + 1
            for index, subshell in enumerate(get_orbital_sequence())
        }
        for row, orbital in enumerate(orbitals):
            label = (
                f"E{energy_order[(orbital.n, orbital.l)]}: "
                f"{orbital.n}{get_orbital_type(orbital.l).letter}  "
                f"mₗ={orbital.m:+d}"
            )
            box = f"[{orbital.spin_symbols:<2}]"
            label_item = QTableWidgetItem(label)
            box_item = QTableWidgetItem(box)
            box_item.setTextAlignment(Qt.AlignCenter)
            quantum_numbers = (orbital.n, orbital.l, orbital.m)
            label_item.setData(Qt.UserRole, quantum_numbers)
            box_item.setData(Qt.UserRole, quantum_numbers)
            follows_ground = (
                orbital.electrons > 0
                and checks["aufbau"] and checks["hund"]
            )
            if follows_ground:
                color = QColor("#183d36")
            elif orbital.electrons:
                color = QColor("#574518")
            elif quantum_numbers == selected:
                color = QColor("#173a5c")
            else:
                color = QColor("#081423")
            if quantum_numbers == selected:
                selected_font = label_item.font()
                selected_font.setBold(True)
                label_item.setFont(selected_font)
                box_item.setFont(selected_font)
                label_item.setToolTip("Orbital selecionado — os botões atuam nesta caixa")
                box_item.setToolTip("Orbital selecionado — os botões atuam nesta caixa")
            label_item.setBackground(color)
            box_item.setBackground(color)
            self.manual_table.setItem(row, 0, label_item)
            self.manual_table.setItem(row, 1, box_item)

        previous_source = self.combo_move_source.currentData()
        self.combo_move_source.blockSignals(True)
        self.combo_move_source.clear()
        for orbital in self.manual_config.orbitals:
            quantum_numbers = (orbital.n, orbital.l, orbital.m)
            if orbital.electrons == 0 or quantum_numbers == selected:
                continue
            label = (
                f"{orbital.n}{get_orbital_type(orbital.l).letter} "
                f"mₗ={orbital.m:+d} [{orbital.spin_symbols}]"
            )
            self.combo_move_source.addItem(label, quantum_numbers)
        if previous_source is not None:
            index = self.combo_move_source.findData(previous_source)
            if index >= 0:
                self.combo_move_source.setCurrentIndex(index)
        self.combo_move_source.blockSignals(False)
        can_move = self.combo_move_source.count() > 0
        self.btn_move_electron.setEnabled(can_move)
        target_name = quantum_label(*selected)
        self.btn_move_electron.setText(f"Mover elétron → {target_name}")


    def closeEvent(self, event):
        if hasattr(self, 'superposition_timer'):
            self.superposition_timer.stop()
        for canvas in self.slice_canvases.values():
            if canvas is not None:
                plt.close(canvas.figure)
        self.simulator.close()
        super().closeEvent(event)


def launch_explorer(simulator):
    """
    Lança a UI de exploração de orbitais.
    
    Parâmetros:
        simulator : objeto Simulator já rodando
    
    Uso:
        app, explorer = launch_explorer(sim)
        app.exec_()
    """
    app = QApplication.instance()  # Pegar app existente ou criar
    if app is None:
        app = QApplication(sys.argv)
    
    explorer = OrbitalExplorer(simulator)
    explorer.show()
    
    return app, explorer


if __name__ == "__main__":
    # Teste
    from atom.atom import Atom
    from simulator.simulator import Simulator
    
    print("Iniciando Orbital Explorer...")
    
    atom = Atom(Z=1)
    sim = Simulator(atom=atom, title="Orbital Explorer — Teste")
    
    app, explorer = launch_explorer(sim)
    
    sys.exit(app.exec_())


