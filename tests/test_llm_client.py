"""Pruebas del cliente de texto. No usan red ni API key: el cliente de Anthropic es un doble."""
from types import SimpleNamespace

import pytest

from src.explain import ExplanationCase, Factor
from src.llm_client import (
    AnthropicBackend,
    LLMConfigError,
    TemplateBackend,
    build_user_message,
    get_explainer,
)

ENV_VARS = ("ANTHROPIC_API_KEY", "LLM_MODEL", "LLM_BACKEND")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def case():
    return ExplanationCase(
        probability=0.623,
        threshold=0.34,
        predicted_churn=True,
        model_name="Regresión logística",
        factors_up=[Factor("Contract", "Contrato", "Month-to-month", 0.83)],
        factors_down=[Factor("tenure", "Antigüedad (meses)", "60", -0.9)],
        context=["Umbral de decisión por costos\nEl umbral es 0.34."],
    )


class FakeClient:
    def __init__(self, text="Texto del modelo.", error=None):
        self.calls, self._text, self._error = [], text, error
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self._text)])


def test_template_is_deterministic_and_uses_only_given_numbers(case):
    text = TemplateBackend().explain(case)
    assert text == TemplateBackend().explain(case)
    assert "62.3 %" in text.replace("%", " %") or "62.3%" in text
    assert "0.34" in text and "Month-to-month" in text and "+0.83" in text and "-0.90" in text
    assert "no causas" in text


def test_template_below_threshold(case):
    case.predicted_churn, case.probability = False, 0.10
    assert "no se marca como riesgo" in TemplateBackend().explain(case)


def test_user_message_contains_model_numbers_and_context(case):
    msg = build_user_message(case)
    assert "62.3%" in msg and "0.34" in msg and "Month-to-month" in msg
    assert "Documentación de referencia" in msg and "El umbral es 0.34" in msg


def test_anthropic_backend_sends_system_prompt_and_user_message(case):
    fake = FakeClient()
    backend = AnthropicBackend(model="modelo-de-prueba", client=fake)
    assert backend.explain(case) == "Texto del modelo."
    call = fake.calls[0]
    assert call["model"] == "modelo-de-prueba"
    assert "## Rol" in call["system"] and "Regresión logística" in call["system"]
    assert "{threshold}" not in call["system"]
    assert call["messages"][0]["role"] == "user"
    assert "62.3%" in call["messages"][0]["content"]


def test_anthropic_backend_requires_model_and_key(monkeypatch):
    with pytest.raises(LLMConfigError, match="LLM_MODEL"):
        AnthropicBackend(client=FakeClient())
    monkeypatch.setenv("LLM_MODEL", "x")
    with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
        AnthropicBackend()


def test_empty_response_raises(case):
    backend = AnthropicBackend(model="x", client=FakeClient(text="   "))
    with pytest.raises(RuntimeError, match="vacía"):
        backend.explain(case)


def test_get_explainer_defaults_to_template():
    assert get_explainer().name == "template"


def test_get_explainer_auto_needs_key_and_model(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "auto")
    assert get_explainer().name == "template"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "clave-falsa")
    assert get_explainer().name == "template"  # falta LLM_MODEL


def test_get_explainer_anthropic_without_config_fails_clearly(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "anthropic")
    with pytest.raises(LLMConfigError):
        get_explainer()


def test_invalid_backend_name(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "otro")
    with pytest.raises(LLMConfigError, match="no es válido"):
        get_explainer()
