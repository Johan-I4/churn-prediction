"""Generadores de texto para `/explain`. Funciona SIN API key.

Backends:
- `template` (por defecto): texto determinista armado con las contribuciones del modelo.
  No usa red, no cuesta nada y es el que se prueba en CI.
- `anthropic` (opcional): usa la API de Anthropic. Requiere ANTHROPIC_API_KEY y LLM_MODEL
  y el paquete `anthropic` (requirements-llm.txt). La clave solo se lee del entorno.

Selección con la variable LLM_BACKEND: `template` (defecto), `anthropic` o `auto`
(`anthropic` solo si hay ANTHROPIC_API_KEY y LLM_MODEL; si no, `template`).
"""
from __future__ import annotations

import os
from typing import Protocol

from .explain import ExplanationCase, Factor
from .prompts import render_system_prompt


class LLMConfigError(RuntimeError):
    """Configuración incompleta del backend (sin mostrar nunca la clave)."""


class Explainer(Protocol):
    name: str

    def explain(self, case: ExplanationCase) -> str: ...


def _fmt_factors(factors: list[Factor]) -> str:
    return "; ".join(f"{f.label} = {f.value} ({f.contribution:+.2f})" for f in factors)


def build_user_message(case: ExplanationCase) -> str:
    """Mensaje de usuario con SOLO los datos que el modelo entregó."""
    lines = [
        f"Probabilidad de churn (calculada por el modelo): {case.probability:.1%}",
        f"Umbral de decisión: {case.threshold:.2f}",
        f"Marcado como riesgo de abandono: {'sí' if case.predicted_churn else 'no'}",
        "Factores que más aumentan el riesgo (contribución en log-odds): "
        + (_fmt_factors(case.factors_up) or "ninguno"),
        "Factores que más reducen el riesgo (contribución en log-odds): "
        + (_fmt_factors(case.factors_down) or "ninguno"),
    ]
    if case.context:
        lines.append("Documentación de referencia:")
        lines += [f"- {c}" for c in case.context]
    return "\n".join(lines)


class TemplateBackend:
    """Redacción determinista, sin LLM."""

    name = "template"

    def explain(self, case: ExplanationCase) -> str:
        verdict = (
            f"supera el umbral de decisión ({case.threshold:.2f}): se marca como riesgo de abandono"
            if case.predicted_churn
            else f"no supera el umbral de decisión ({case.threshold:.2f}): no se marca como riesgo"
        )
        parts = [f"La probabilidad estimada de churn es {case.probability:.1%} y {verdict}."]
        if case.factors_up:
            parts.append(f"Lo que más aumenta el riesgo: {_fmt_factors(case.factors_up)}.")
        if case.factors_down:
            parts.append(f"Lo que más lo reduce: {_fmt_factors(case.factors_down)}.")
        parts.append(
            "Son asociaciones del modelo (contribuciones al log-odds), no causas: "
            "cambiar un factor no garantiza que el cliente se quede."
        )
        return " ".join(parts)


class AnthropicBackend:
    """Redacción con la API de Anthropic. El cliente se puede inyectar (pruebas con mock)."""

    name = "anthropic"

    def __init__(self, model: str | None = None, client=None, max_tokens: int = 400):
        self.model = model or os.getenv("LLM_MODEL")
        if not self.model:
            raise LLMConfigError("Define LLM_MODEL con el identificador del modelo a usar.")
        self.max_tokens = max_tokens
        if client is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise LLMConfigError("Falta la variable de entorno ANTHROPIC_API_KEY.")
            try:
                import anthropic
            except ImportError as exc:
                raise LLMConfigError("Instala el paquete: pip install -r requirements-llm.txt") from exc
            client = anthropic.Anthropic(api_key=api_key)
        self._client = client

    def explain(self, case: ExplanationCase) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=render_system_prompt(case.model_name, case.threshold),
            messages=[{"role": "user", "content": build_user_message(case)}],
        )
        text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text").strip()
        if not text:
            raise RuntimeError("El modelo devolvió una respuesta vacía.")
        return text


def get_explainer() -> Explainer:
    """Elige el backend según LLM_BACKEND (defecto: template)."""
    choice = os.getenv("LLM_BACKEND", "template").strip().lower()
    if choice == "template":
        return TemplateBackend()
    if choice == "anthropic":
        return AnthropicBackend()
    if choice == "auto":
        if os.getenv("ANTHROPIC_API_KEY") and os.getenv("LLM_MODEL"):
            return AnthropicBackend()
        return TemplateBackend()
    raise LLMConfigError(f"LLM_BACKEND='{choice}' no es válido (usa template, anthropic o auto).")
