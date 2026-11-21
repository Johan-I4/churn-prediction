"""Carga y validación de los prompts versionados en `prompts/`.

El prompt de sistema sigue 5 secciones: Rol, Contexto, Objetivo, Formato y Restricciones.
La ruta se puede cambiar con la variable de entorno PROMPTS_DIR.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

DEFAULT_PROMPTS_DIR = "prompts"
EXPLAIN_PROMPT = "explicar_churn.md"
REQUIRED_SECTIONS = ("Rol", "Contexto", "Objetivo", "Formato", "Restricciones")


class PromptError(ValueError):
    """El prompt no existe o no cumple la estructura esperada."""


def prompts_dir() -> Path:
    return Path(os.getenv("PROMPTS_DIR", DEFAULT_PROMPTS_DIR))


def parse_sections(text: str) -> dict[str, str]:
    """Devuelve {título de sección '## ...': contenido}. Ignora comentarios HTML."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    sections: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        match = re.match(r"^##\s+(.+?)\s*$", line)
        if match:
            current = match.group(1)
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def missing_sections(text: str) -> list[str]:
    """Secciones obligatorias que faltan o están vacías."""
    found = parse_sections(text)
    return [name for name in REQUIRED_SECTIONS if not found.get(name)]


def load_prompt(name: str = EXPLAIN_PROMPT) -> str:
    path = prompts_dir() / name
    if not path.exists():
        raise PromptError(f"No encontré el prompt '{path}'.")
    text = path.read_text(encoding="utf-8")
    missing = missing_sections(text)
    if missing:
        raise PromptError(f"Al prompt '{path}' le faltan secciones: {', '.join(missing)}")
    return text


def render_system_prompt(model_name: str, threshold: float) -> str:
    """Prompt de sistema listo para enviar: quita comentarios y rellena los marcadores."""
    text = load_prompt(EXPLAIN_PROMPT)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = text.replace("{model_name}", str(model_name)).replace("{threshold}", f"{threshold:.2f}")
    return text.strip()
