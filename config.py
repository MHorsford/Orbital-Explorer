"""
config.py

Configurações globais do simulador de orbitais atômicos.
Importar daqui evita dependências circulares.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# RENDERIZAÇÃO 3D

GRID_SIZE = 120              # Pontos por dimensão para calcular densidade
GRID_RANGE = 20.0            # Extensão da caixa em unidades de Bohr (±8 a.u.)
ISO_VALUE = 0.06            # Isosurface: |ψ|² = 2% do máximo (0.05 p 0.08)

# Modo de renderização padrão: 'isosurface', 'points' ou 'grid_points'
RENDER_MODE = 'isosurface'
NUM_POINTS_CLOUD = 10000    # Pontos da nuvem Monte Carlo

# VISUALIZAÇÃO

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 900
FPS_TARGET = 30

# Raios visuais das partículas (Angstroms)
PROTON_RADIUS = 0.1
NEUTRON_RADIUS = 0.1
ELECTRON_RADIUS = 0.05

# Cores (RGB 0-1)
COLOR_PROTON = (0.9, 0.3, 0.2)      # Vermelho-coral
COLOR_NEUTRON = (0.4, 0.4, 0.4)     # Cinza
COLOR_ELECTRON = (0.2, 0.6, 0.9)    # Azul
COLOR_NUCLEUS = (0.95, 0.7, 0.1)    # Ouro

# Opacidade padrão dos orbitais
ORBITAL_OPACITY_ISO = 0.45
ORBITAL_OPACITY_VOLUME = 0.5

# UI / CONTROLES

# Intervalo de número atômico no slider
MIN_Z = 1
MAX_Z = 118          # Até o oganessônio

# Intervalo de nível principal
MIN_N = 1
MAX_N = 7

# PALETA DE CORES PARA ORBITAIS

ORBITAL_COLORS = {
    0: (0.2, 0.8, 1.0),      # s → Azul claro
    1: (1.0, 0.55, 0.0),     # p → Laranja
    2: (0.8, 0.2, 1.0),      # d → Roxo
    3: (0.2, 1.0, 0.4),      # f → Verde
}

# Convenção visual das fases da função de onda.
ORBITAL_NEGATIVE_PHASE_COLOR = (0.95, 0.95, 0.95)
ORBITAL_EMPTY_POSITIVE_COLOR = (0.20, 0.88, 1.00)  # Ciano luminoso
ORBITAL_EMPTY_NEGATIVE_COLOR = (1.00, 0.35, 0.78)  # Magenta luminoso

# CENA 3D

# Mostrar eixos de coordenadas?
SHOW_AXES = True

# Mostrar núcleo como esfera central?
SHOW_NUCLEUS = True

# Câmera inicial (posição em unidades de Bohr)
CAMERA_INITIAL_POSITION = (45, 45, 45)
CAMERA_FOCAL_POINT = (0, 0, 0)

# Background
BG_COLOR = (0.05, 0.05, 0.08)       # Azul muito escuro (quase preto)

# Iluminação
AMBIENT_LIGHT_INTENSITY = 0.4
DIRECTIONAL_LIGHT_INTENSITY = 0.8
