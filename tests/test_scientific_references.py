from ui.scientific_references import SCIENTIFIC_REFERENCES


def test_scientific_references_cover_the_models_and_experimental_data():
    titles = " ".join(reference["title"] for reference in SCIENTIFIC_REFERENCES)

    assert "OpenStax" in titles
    assert "Slater" in titles
    assert "NIST Atomic Spectra" in titles
    assert "hidrogênio neutro" in titles
    assert "Regras de seleção" in titles
    assert "Efeito Zeeman" in titles
    assert "Magnetão de Bohr" in titles
    assert "CODATA" in titles


def test_scientific_reference_links_are_secure():
    assert all(
        reference["url"].startswith("https://")
        for reference in SCIENTIFIC_REFERENCES
    )
