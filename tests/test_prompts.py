import pytest

from src.prompts import (
    EXPLAIN_PROMPT,
    REQUIRED_SECTIONS,
    PromptError,
    load_prompt,
    missing_sections,
    parse_sections,
    render_system_prompt,
)


def test_explain_prompt_has_the_five_sections():
    sections = parse_sections(load_prompt(EXPLAIN_PROMPT))
    for name in REQUIRED_SECTIONS:
        assert sections.get(name), f"falta o está vacía la sección '{name}'"


def test_missing_sections_detects_gaps():
    text = "## Rol\nAnalista\n\n## Contexto\nAlgo\n\n## Objetivo\n\n## Formato\nBreve"
    assert missing_sections(text) == ["Objetivo", "Restricciones"]


def test_load_prompt_rejects_incomplete_prompt(tmp_path, monkeypatch):
    (tmp_path / "malo.md").write_text("## Rol\nSolo un rol", encoding="utf-8")
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    with pytest.raises(PromptError, match="faltan secciones"):
        load_prompt("malo.md")


def test_load_prompt_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    with pytest.raises(PromptError, match="No encontré"):
        load_prompt("no_existe.md")


def test_render_fills_placeholders_and_strips_comments():
    text = render_system_prompt("Regresión logística", 0.34)
    assert "{model_name}" not in text and "{threshold}" not in text
    assert "Regresión logística" in text and "0.34" in text
    assert "<!--" not in text


def test_prompt_forbids_inventing_numbers():
    restrictions = parse_sections(load_prompt(EXPLAIN_PROMPT))["Restricciones"].lower()
    assert "no inventes" in restrictions
