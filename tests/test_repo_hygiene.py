"""Evita subir credenciales por accidente."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _lines(name: str) -> list[str]:
    return [ln.strip() for ln in (ROOT / name).read_text(encoding="utf-8").splitlines()]


def test_env_files_are_git_ignored():
    assert ".env" in _lines(".gitignore")


def test_env_files_are_not_sent_to_docker():
    assert ".env" in _lines(".dockerignore")


def test_env_example_has_no_real_key():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "ANTHROPIC_API_KEY=" in text
    for line in text.splitlines():
        if line.startswith("ANTHROPIC_API_KEY="):
            assert line.split("=", 1)[1].strip() in ("", "tu_clave_aqui")


def test_no_secret_like_strings_in_source():
    for path in list((ROOT / "src").rglob("*.py")) + list((ROOT / "prompts").glob("*.md")):
        assert "sk-ant-" not in path.read_text(encoding="utf-8"), path
