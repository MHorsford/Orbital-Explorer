# Guia de Constantes Físicas — Simulador de Orbitais Atômicos

## 📋 Visão Geral

O arquivo `constants.py` centraliza todas as constantes físicas necessárias para o simulador. Isso garante:
- **Precisão**: Valores CODATA 2018 (padrão internacional)
- **Consistência**: Um único lugar para manter valores atualizados
- **Facilidade**: Importar constantes é simples
- **Conversões**: Funções prontas para converter entre unidades

---

## 🚀 Como Usar

### Importação Básica

```python
from physics.constants import A0_METERS, E_CHARGE, HBAR, K_COULOMB

# Usar directamente
raio_bohr = A0_METERS  # 5.292e-11 metros
carga_eletron = E_CHARGE  # 1.602e-19 Coulombs
```

### Conversões

```python
from physics.constants import (
    bohr_to_meters, 
    ev_to_joule, 
    hartree_to_ev
)

# Converter 2 unidades de Bohr para metros
distancia = bohr_to_meters(2.0)  # → 1.058e-10 m

# Converter 13.6 eV para Joules
energia = ev_to_joule(13.6)  # → 2.178e-18 J

# Converter 1 Hartree para eV
E_hartree = hartree_to_ev(1.0)  # → 27.211 eV
```

---

## 📊 Constantes Disponíveis

### Grupo 1: Constantes Fundamentais (SI)

| Constante | Símbolo | Valor | Unidade |
|-----------|---------|-------|---------|
| Carga elementar | `E_CHARGE` | 1.602e-19 | C |
| Massa do elétron | `M_ELECTRON` | 9.109e-31 | kg |
| Massa do próton | `M_PROTON` | 1.673e-27 | kg |
| Constante de Planck | `HBAR` | 1.055e-34 | J·s |
| Const. de Coulomb | `K_COULOMB` | 8.988e+9 | N·m²/C² |
| Velocidade da luz | `C` | 2.998e+8 | m/s |

**Uso típico em `wavefunction.py`:**
```python
from physics.constants import A0_METERS, HBAR, M_ELECTRON

# Cálculo da função de onda
rho = (2 * Z * r) / (n * A0_METERS)
```

---

### Grupo 2: Constantes Atômicas

| Constante | Símbolo | Valor | Notas |
|-----------|---------|-------|-------|
| Raio de Bohr (metros) | `A0_METERS` | 5.292e-11 m | Unidade de comprimento atômico |
| Raio de Bohr (Angstrom) | `A0_ANGSTROM` | 0.5292 Å | Mais conveniente para visualização |
| Energia Rydberg | `RY_EV` | 13.606 eV | Energia de ionização do H |
| Magnetão de Bohr | `MU_B` | 9.285e-24 J/T | Momento magnético do elétron |

**Uso em `orbital.py`:**
```python
from physics.constants import A0_ANGSTROM

# Classe HydrogenWavefunction
self.a0 = A0_ANGSTROM  # Compatível com código existente
```

---

### Grupo 3: Screening (Blindagem Eletrônica)

Para cálculos com átomos multieletrônicos, use:

```python
from physics.constants import SLATER_SCREENING, PENETRATION_FACTOR

# Fator de screening para um elétron em orbital s
screening_s = SLATER_SCREENING['s']  # 0.30

# Fator de penetração para d
penetration_d = PENETRATION_FACTOR['d']  # 0.35
```

**Uso em `screening.py`:**
```python
# Cálculo de carga nuclear efetiva
Z_eff = Z - screening_s * (numero_eletrons_internos)
```

---

### Grupo 4: Visualização e Simulação

| Constante | Valor | Significado |
|-----------|-------|-------------|
| `GRID_SIZE_DEFAULT` | 80 | Resolução da grade 3D (80³ pontos) |
| `GRID_RANGE_DEFAULT` | 8.0 | Extensão da caixa (±8 unidades de Bohr) |
| `ISO_VALUE_DEFAULT` | 0.02 | Isosurface a 2% da densidade máxima |
| `PROTON_RADIUS_VISUAL` | 0.1 Å | Tamanho visual do próton |

**Uso em `renderer.py`:**
```python
from physics.constants import ISO_VALUE_DEFAULT, GRID_SIZE_DEFAULT

# Renderizar isosurface
density_grid, X, Y, Z = wavefunction.evaluate_on_grid(
    n=2, l=1, size=GRID_SIZE_DEFAULT
)
isosurface = create_isosurface(density_grid, ISO_VALUE_DEFAULT)
```

---

## 🔧 Exemplos Práticos

### Exemplo 1: Cálculo de Energia de Ionização

```python
from physics.constants import RY_EV

# Energia de ionização do H (1º nível)
E_ionization = RY_EV / 1**2  # 13.6 eV

# Energia do nível n=2
E_n2 = RY_EV / 2**2  # 3.4 eV

print(f"ΔE (transição 1→2): {E_ionization - E_n2} eV")
```

### Exemplo 2: Cálculo de Força Coulombiana

```python
from physics.constants import K_COULOMB, E_CHARGE, A0_METERS

# Força entre próton e elétron a 1 Bohr de distância
r = A0_METERS
F = K_COULOMB * E_CHARGE**2 / r**2

print(f"Força Coulombiana: {F:.3e} N")
```

### Exemplo 3: Screening de Slater para Carbono

```python
from physics.constants import SLATER_SCREENING

# C (Z=6): [He] 2s² 2p²
# Elétron em 2p enxerga:
#   - 2 elétrons em 2s e 2p (mesmo subnível): 2 × 0.35
#   - 2 elétrons em 1s (interno): 2 × 0.85

Z_C = 6
Z_eff_2p = Z_C - (2 * 0.35 + 2 * 0.85)  # = 3.3

print(f"Carga nuclear efetiva (C, 2p): {Z_eff_2p:.1f}")
```

---

## 📐 Conversões Úteis

### Comprimento
```python
from physics.constants import *

# Conversão interativa
r_bohr = 1.0
print(f"{r_bohr} Bohr = {bohr_to_meters(r_bohr):.3e} m")
print(f"                = {bohr_to_angstrom(r_bohr):.6f} Å")
```

### Energia
```python
from physics.constants import *

# Seguir a cascata de conversões
E_ev = 27.2  # eV
E_j = ev_to_joule(E_ev)
E_hartree = ev_to_hartree(E_ev)

print(f"{E_ev} eV = {E_j:.3e} J = {E_hartree:.3f} Hartree")
```

---

## ⚠️ Notas Importantes

### 1. **Raio de Bohr: Use a Versão Correta**

```python
# ✓ Para cálculos internos (metros)
from physics.constants import A0_METERS

# ✓ Para visualização (Angstroms)  
from physics.constants import A0_ANGSTROM

# ✗ Não misture unidades sem conversão!
```

### 2. **Valores Derivados vs Calculados**

Constantes derivadas são recalculadas a cada execução para garantir consistência:

```python
# Calculado automaticamente a partir de outras constantes
K_COULOMB = 1 / (4 * π * ε₀)

# Isso evita erros de arredondamento
```

### 3. **Screening: Contexto Importa**

Os valores em `SLATER_SCREENING` são **recomendações**. Para precisão máxima:
- Use dados experimentais de ionização
- Considere efeitos relativísticos para átomos pesados
- Ajuste Z_eff baseado na geometria específica

---

## 🧪 Testando as Constantes

```bash
# Execute o arquivo diretamente
python physics/constants.py
```

Saída esperada:
```
======================================================================
CONSTANTES FÍSICAS DO SIMULADOR DE ORBITAIS ATÔMICOS
======================================================================

[CONSTANTES FUNDAMENTAIS]
  Constante de Planck (ℏ): 1.054572e-34 J·s
  Velocidade da luz (c):    2.997925e+08 m/s
  ...
```

---

## 📚 Referências

- **CODATA 2018**: https://physics.nist.gov/cuu/Constants/
- **Atomic Spectra Database**: https://physics.nist.gov/asd/
- **Griffiths, D. J.**: "Introduction to Quantum Mechanics" (2ed, 2015)
- **Slater Rules**: https://en.wikipedia.org/wiki/Slater%27s_rules

---

## ✅ Checklist de Integração

Ao usar `constants.py` em seus módulos, verifique:

- [ ] Importações no topo do arquivo
- [ ] Conversões de unidades (se necessário)
- [ ] Compatibilidade com código existente
- [ ] Comentários explicativos em cálculos não-óbvios
- [ ] Testes unitários (se mudar valores)

---

**Versão**: 1.0  
**Última atualização**: 2026-06-20  
**Status**: ✅ Pronto para produção
