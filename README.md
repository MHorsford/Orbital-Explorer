# Orbital Explorer

Simulador interativo de orbitais atômicos desenvolvido em Python para apoiar o ensino de física quântica e configuração eletrônica.

O programa permite explorar formas orbitais em 3D, analisar cortes da função de onda em 2D e construir configurações eletrônicas verificando o Princípio de Aufbau, a Regra de Hund e o Princípio da Exclusão de Pauli.

## Recursos principais

- Seleção de elementos químicos de `Z = 1` a `Z = 118`.
- Exploração dos números quânticos `n`, `l` e `m`.
- Visualização 3D por isosuperfícies ou nuvem de pontos.
- Cortes 2D da amplitude `ψ` e da probabilidade `|ψ|²`.
- Representação das fases positiva e negativa da função de onda.
- Configuração eletrônica fundamental dos átomos neutros.
- Aplicação de exceções conhecidas da ordem simples de Aufbau.
- Diagramas de orbitais com spins `↑` e `↓`.
- Verificação de Aufbau, Hund e Pauli.
- Construção manual de configurações a partir de orbitais vazios.
- Identificação de estados fundamentais e estados excitados permitidos.
- Promoção de elétrons entre orbitais no modo manual.

## Modos de interação

### Explorar orbitais

Exibe a forma matemática do orbital selecionado, mesmo quando ele está vazio no elemento escolhido. É o modo indicado para estudar os efeitos de `n`, `l` e `m` sobre a geometria orbital.

### Átomo real

Mostra a ocupação eletrônica fundamental do elemento selecionado. Orbitais vazios não representam densidade eletrônica física nesse modo.

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
4. Use a aba central para alternar entre a visualização 3D e o corte 2D.
5. Consulte os dados, regras e controles de preenchimento no painel lateral.

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
tests/      Testes automatizados e demonstração visual
ui/         Interface gráfica e tema
utils/      Grids, amostragem e cortes 2D
main.py     Ponto de entrada da aplicação
```

## Testes

Os testes automatizados de função de onda e preenchimento eletrônico podem ser executados com:

```bash
python -m pytest tests/test_electron_filling.py tests/teste_wavefunction.py -q
```

O arquivo `tests/test_orbital_visual.py` é uma demonstração gráfica interativa e requer uma tela com suporte a OpenGL.

## Escopo científico

O simulador utiliza funções de onda hidrogenoides com carga nuclear efetiva estimada pelas regras de Slater. Essa aproximação produz representações didáticas coerentes, mas não substitui métodos de química quântica multieletrônica, como Hartree–Fock ou DFT.

Atualmente, o modo de átomo real trabalha com átomos neutros. O suporte a íons faz parte das extensões planejadas.

## Próximos passos

- Suporte a cátions e ânions.
- Atividades didáticas guiadas.
- Exportação de imagens e relatórios.
- Site estático com documentação matemática, técnica e pedagógica completa.

## Status

Projeto acadêmico em desenvolvimento.
