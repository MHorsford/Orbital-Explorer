# Orbital Explorer

Simulador interativo de orbitais atômicos desenvolvido em Python para apoiar o ensino de física quântica e configuração eletrônica.

O programa permite explorar formas orbitais em 3D, analisar cortes da função de onda em 2D, construir configurações eletrônicas e acompanhar níveis de energia e transições eletrônicas.

## Recursos principais

- Seleção de elementos químicos de `Z = 1` a `Z = 118`.
- Exploração dos números quânticos `n`, `l` e `m`.
- Visualização 3D por isosuperfícies, nuvem de pontos ou grade de pontos.
- Cortes 2D da amplitude `ψ` e da probabilidade `|ψ|²`.
- Gráficos da função radial `Rₙₗ(r)` e da distribuição de probabilidade `P(r)`.
- Identificação de nós radiais e angulares, raio mais provável e raio médio.
- Representação das fases positiva e negativa da função de onda.
- Configuração eletrônica fundamental de átomos neutros, cátions e ânions.
- Aplicação de exceções conhecidas da ordem simples de Aufbau.
- Diagramas de orbitais com spins `↑` e `↓`.
- Verificação de Aufbau, Hund e Pauli.
- Construção manual de configurações a partir de orbitais vazios.
- Identificação de estados fundamentais e estados excitados permitidos.
- Promoção de elétrons entre orbitais no modo manual.
- Cartão do estado selecionado com `n`, `l`, `mₗ` e `mₛ` de cada elétron.
- Diagrama de níveis de energia com ocupação e carga nuclear efetiva estimada.
- Cálculo didático de absorção e emissão, com energia, frequência e comprimento de onda do fóton.
- Diagnóstico de transições E1 por estado eletrônico: `Δl`, `Δmₗ`, conservação de spin e mudança de paridade.
- Comparação opcional das transições de H I até `n=5` com níveis médios avaliados pelo NIST.
- Campo magnético externo `Bz` de `−10 T` a `+10 T`, com deslocamento Zeeman linear dos estados e das transições.
- Superposição coerente de dois estados orbitais de um elétron, com animação desacelerada da densidade `|Ψ(t)|²`.
- Controles de estado B, peso probabilístico, reprodução, pausa, reinício e ritmo visual.

## Modos de interação

### Explorar orbitais

Exibe a forma matemática do orbital selecionado, mesmo quando ele está vazio no elemento escolhido. É o modo indicado para estudar os efeitos de `n`, `l` e `m` sobre a geometria orbital.

### Átomo real

Mostra a ocupação eletrônica fundamental da espécie selecionada. A carga pode ser ajustada para estudar íons. Orbitais vazios não representam densidade eletrônica física nesse modo.

### Preenchimento manual

Inicia com os orbitais vazios e permite adicionar, remover ou promover elétrons. A interface informa, a cada alteração, se a configuração respeita Aufbau, Hund e Pauli.

## Tecnologias

- Python
- NumPy e SciPy
- PyVista e PyVistaQt
- PyQt5
- Matplotlib
- Pandas

## Instalação

Clone o repositório e entre na pasta do projeto:

```bash
git clone https://github.com/MHorsford/Atomic-Orbital-3D-Viewer.git
cd Atomic-Orbital-3D-Viewer
```

É recomendado criar um ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente no Windows:

```powershell
.venv\Scripts\Activate.ps1
```

No Linux ou macOS:

```bash
source .venv/bin/activate
```

Instale as dependências:

```bash
python -m pip install -r requirements.txt
```

## Execução

Inicie a interface gráfica com:

```bash
python main.py
```

Na interface:

1. Escolha o elemento químico.
2. Selecione o modo de interação.
3. Ajuste os números quânticos do orbital.
4. Use as abas centrais para alternar entre orbital 3D, corte 2D, distribuição radial e energia/transições.
5. Consulte dados, regras, preenchimento manual e fontes científicas no painel lateral.

Na aba Energia, o controle `Campo Bz` aplica um campo magnético paralelo ao eixo `z`. O resultado informa os deslocamentos dos estados, da frequência e do comprimento de onda, além de identificar componentes `π` e `σ`.

Na visualização 3D, marque `Superposição temporal`. O orbital definido por `n`, `l` e `m` será o estado A; escolha o estado B e ajuste seu peso. A interface informa `ΔE`, a frequência de batimento e o período físico. O ritmo da animação é deliberadamente desacelerado para que a evolução seja visível. Essa visualização prepara um estado espacial de um elétron e não modifica a ocupação eletrônica do átomo.

O visualizador 3D pode ser rotacionado com o mouse e ampliado com a roda de rolagem.

## Estrutura do projeto

```text
atom/       Átomos e configurações eletrônicas
data/       Dados da tabela periódica
nucleus/    Representação do núcleo
orbitals/   Orbitais e funções de onda
particles/  Partículas fundamentais
physics/    Constantes, blindagem e cálculos físicos
simulator/  Renderização e gerenciamento da cena
demos/      Demonstrações gráficas interativas
tests/      Testes automatizados
ui/         Interface gráfica e tema
utils/      Grids, amostragem e cortes 2D
main.py     Ponto de entrada da aplicação
```

## Testes

Os testes automatizados de função de onda, distribuição radial, preenchimento eletrônico, íons e energias podem ser executados com:

```bash
python -m pytest tests -q
```

O arquivo `demos/orbital_visual_demo.py` é uma demonstração gráfica interativa e requer uma tela com suporte a OpenGL.

## Escopo científico

O simulador utiliza funções de onda hidrogenoides com carga nuclear efetiva estimada pelas regras de Slater. Os gráficos radiais e as energias exibidas usam esse mesmo modelo aproximado. A evolução temporal combina dois estados como `Ψ(t)=cₐψₐe⁻ⁱᴱᵃᵗ/ℏ+cᵦψᵦe⁻ⁱᴱᵇᵗ/ℏ` e renderiza `|Ψ(t)|²`; estados degenerados permanecem estacionários neste modelo. A velocidade visível é uma desaceleração didática, enquanto a frequência e o período mostrados são os valores físicos aproximados. O diagnóstico E1 considera os estados individuais `n`, `l`, `mₗ` e `mₛ`; ele não calcula estrutura fina, acoplamento spin–órbita ou `J`. O campo magnético usa o modelo Zeeman linear na base desacoplada `|n,l,mₗ,mₛ⟩`; ele desloca as energias, mas não deforma os orbitais 3D. A comparação NIST é limitada ao hidrogênio neutro até `n=5` e usa valores sem campo. Esses modelos são adequados à exploração didática, mas não substituem métodos de química quântica multieletrônica, como Hartree–Fock ou DFT.

## Referências científicas

- [OpenStax Chemistry 2e — Desenvolvimento da teoria quântica](https://openstax.org/books/chemistry-2e/pages/6-3-development-of-quantum-theory): função de onda, probabilidade e números quânticos.
- [OpenStax Chemistry 2e — Configurações eletrônicas](https://openstax.org/books/chemistry-2e/pages/6-4-electronic-structure-of-atoms-electron-configurations): Aufbau, Hund, Pauli, diagramas e exceções.
- [MIT OpenCourseWare — Quantum Physics I](https://ocw.mit.edu/courses/8-04-quantum-physics-i-spring-2013/pages/lecture-notes/): superposição, estados estacionários e evolução temporal pela equação de Schrödinger.
- [IUPAC Gold Book — Princípio da Exclusão de Pauli](https://goldbook.iupac.org/terms/view/PT07089): definição terminológica.
- [J. C. Slater — Atomic Shielding Constants](https://doi.org/10.1103/PhysRev.36.57): artigo original do modelo de blindagem usado em `Z_eff`.
- [NIST Atomic Spectra Database](https://physics.nist.gov/asd): níveis e linhas espectrais avaliados para átomos e íons.
- [NIST — Níveis de energia de H I](https://physics.nist.gov/PhysRefData/Handbook/Tables/hydrogentable5.htm): níveis avaliados usados na comparação incorporada.
- [NIST Atomic Spectroscopy — Regras de seleção](https://www.nist.gov/pml/atomic-spectroscopy-compendium-basic-ideas-notation-data-and-formulas/atomic-spectroscopy): regras para transições E1, M1 e E2.
- [NIST Atomic Spectroscopy — Efeito Zeeman](https://physics.nist.gov/Pubs/AtSpec/node12.html): desdobramento de níveis em campo magnético.
- [NIST/CODATA — Magnetão de Bohr em eV/T](https://physics.nist.gov/cgi-bin/cuu/Value?mubev=): constante usada no deslocamento Zeeman.
- [NIST/CODATA — Constantes físicas fundamentais](https://physics.nist.gov/cuu/Constants/index.html): valores recomendados das constantes usadas nos cálculos.

## Próximos passos

- Site estático com documentação matemática, técnica e pedagógica completa.
- Módulo avançado opcional de estrutura fina e acoplamento spin–órbita.

## Status

Projeto acadêmico em desenvolvimento.
