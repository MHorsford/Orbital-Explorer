"""Referências científicas consultáveis pela interface."""

from PyQt5.QtWidgets import QLabel, QTextBrowser, QVBoxLayout, QWidget


SCIENTIFIC_REFERENCES = (
    {
        "title": "OpenStax Chemistry 2e — Desenvolvimento da teoria quântica",
        "scope": "Função de onda, densidade de probabilidade e números quânticos.",
        "url": "https://openstax.org/books/chemistry-2e/pages/6-3-development-of-quantum-theory",
    },
    {
        "title": "OpenStax Chemistry 2e — Configurações eletrônicas",
        "scope": "Aufbau, Regra de Hund, Pauli, diagramas e exceções.",
        "url": "https://openstax.org/books/chemistry-2e/pages/6-4-electronic-structure-of-atoms-electron-configurations",
    },
    {
        "title": "IUPAC Gold Book — Princípio da Exclusão de Pauli",
        "scope": "Definição terminológica recomendada pela IUPAC.",
        "url": "https://goldbook.iupac.org/terms/view/PT07089",
    },
    {
        "title": "J. C. Slater — Atomic Shielding Constants (1930)",
        "scope": "Artigo original das constantes de blindagem usadas em Z_eff.",
        "url": "https://doi.org/10.1103/PhysRev.36.57",
    },
    {
        "title": "NIST Atomic Spectra Database",
        "scope": "Níveis de energia, linhas espectrais e transições avaliadas.",
        "url": "https://physics.nist.gov/asd",
    },
    {
        "title": "NIST — Níveis de energia do hidrogênio neutro (H I)",
        "scope": "Níveis avaliados usados na comparação espectroscópica até n=5.",
        "url": "https://physics.nist.gov/PhysRefData/Handbook/Tables/hydrogentable5.htm",
    },
    {
        "title": "NIST Atomic Spectroscopy — Regras de seleção",
        "scope": "Transições E1, M1 e E2, mudança de paridade e regras angulares.",
        "url": "https://www.nist.gov/pml/atomic-spectroscopy-compendium-basic-ideas-notation-data-and-formulas/atomic-spectroscopy",
    },
    {
        "title": "NIST Atomic Spectroscopy — Efeito Zeeman",
        "scope": "Desdobramento magnético de níveis atômicos e magnetão de Bohr.",
        "url": "https://physics.nist.gov/Pubs/AtSpec/node12.html",
    },
    {
        "title": "NIST/CODATA — Magnetão de Bohr em eV/T",
        "scope": "Constante usada no cálculo do deslocamento Zeeman linear.",
        "url": "https://physics.nist.gov/cgi-bin/cuu/Value?mubev=",
    },
    {
        "title": "NIST/CODATA — Constantes físicas fundamentais",
        "scope": "Constantes de Planck, Rydberg, velocidade da luz e carga elementar.",
        "url": "https://physics.nist.gov/cuu/Constants/index.html",
    },
)


class ScientificReferencesWidget(QWidget):
    """Lista fontes científicas e explicita o alcance aproximado do modelo."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(9, 9, 9, 9)
        layout.setSpacing(8)

        intro = QLabel(
            "Referências para consultar a base científica do simulador. "
            "Os resultados quantitativos do programa são aproximações "
            "hidrogenoides com blindagem de Slater."
        )
        intro.setObjectName("helpBanner")
        intro.setWordWrap(True)
        intro.setToolTip(
            "Para valores espectroscópicos experimentais, consulte o NIST ASD."
        )
        layout.addWidget(intro)

        browser = QTextBrowser()
        browser.setObjectName("scientificReferences")
        browser.setOpenExternalLinks(True)
        browser.setToolTip("Clique no título de uma fonte para abri-la no navegador.")
        cards = []
        for reference in SCIENTIFIC_REFERENCES:
            cards.append(
                "<p>"
                f"<a href='{reference['url']}'><b>{reference['title']}</b></a><br>"
                f"<span>{reference['scope']}</span>"
                "</p>"
            )
        browser.setHtml(
            "<style>"
            "body { color: #dceeff; font-family: 'Segoe UI'; }"
            "a { color: #55d7f2; text-decoration: none; }"
            "p { margin: 8px 2px 13px 2px; }"
            "span { color: #aac0d4; }"
            "</style>"
            "<h3>Fontes científicas</h3>"
            + "".join(cards)
            + "<hr><p><b>Importante:</b> Aufbau, Hund e Pauli verificam a "
              "organização eletrônica; eles não garantem, sozinhos, a "
              "estabilidade física de um íon.</p>"
        )
        layout.addWidget(browser, stretch=1)
