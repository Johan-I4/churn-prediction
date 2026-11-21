"""Explicación local exacta de una predicción de regresión logística.

En una regresión logística el log-odds de un cliente es:

    logit = intercepto + suma(coeficiente_i * valor_transformado_i)

así que cada término es la contribución exacta de esa variable. No es SHAP ni una
aproximación: la identidad se comprueba en las pruebas (la suma reproduce `predict_proba`).

Interpretación de las contribuciones:
- Variables numéricas: están estandarizadas, la contribución es respecto al promedio del train.
- Variables categóricas: solo aporta la categoría que tiene el cliente (las demás valen 0).
- Son asociaciones del modelo, no causas.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd
from sklearn.linear_model import LogisticRegression

from .features import add_features

FEATURE_LABELS = {
    "SeniorCitizen": "Adulto mayor",
    "tenure": "Antigüedad (meses)",
    "MonthlyCharges": "Cargo mensual",
    "TotalCharges": "Cargo total",
    "n_services": "Número de servicios",
    "monthly_per_tenure": "Cargo mensual relativo a la antigüedad",
    "is_new_customer": "Cliente nuevo (6 meses o menos)",
    "gender": "Género",
    "Partner": "Tiene pareja",
    "Dependents": "Tiene dependientes",
    "PhoneService": "Servicio telefónico",
    "MultipleLines": "Líneas múltiples",
    "InternetService": "Servicio de internet",
    "OnlineSecurity": "Seguridad en línea",
    "OnlineBackup": "Respaldo en línea",
    "DeviceProtection": "Protección de dispositivo",
    "TechSupport": "Soporte técnico",
    "StreamingTV": "Streaming de TV",
    "StreamingMovies": "Streaming de películas",
    "Contract": "Contrato",
    "PaperlessBilling": "Factura electrónica",
    "PaymentMethod": "Método de pago",
}


class UnsupportedModelError(TypeError):
    """El modelo guardado no es una regresión logística: no hay explicación exacta."""


@dataclass(frozen=True)
class Factor:
    feature: str  # nombre original de la variable, p. ej. "Contract"
    label: str  # nombre legible, p. ej. "Contrato"
    value: str  # valor del cliente, p. ej. "Month-to-month"
    contribution: float  # contribución al log-odds (positiva = aumenta el riesgo)

    @property
    def effect(self) -> str:
        return "aumenta" if self.contribution > 0 else "reduce"


@dataclass(frozen=True)
class Explanation:
    intercept: float
    factors: list[Factor]  # ordenados por |contribución| descendente

    @property
    def logit(self) -> float:
        return self.intercept + sum(f.contribution for f in self.factors)

    @property
    def probability(self) -> float:
        return 1.0 / (1.0 + math.exp(-self.logit))


@dataclass
class ExplanationCase:
    """Todo lo que un generador de texto necesita para redactar la explicación."""

    probability: float
    threshold: float
    predicted_churn: bool
    model_name: str
    factors_up: list[Factor]
    factors_down: list[Factor]
    context: list[str] = field(default_factory=list)  # fragmentos recuperados (RAG)


def _format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def explain_customer(pipeline, customer: pd.DataFrame) -> Explanation:
    """Contribuciones exactas por variable para UN cliente (DataFrame de una fila, datos crudos)."""
    model = pipeline.named_steps["model"]
    if not isinstance(model, LogisticRegression):
        raise UnsupportedModelError(
            f"La explicación exacta solo existe para regresión logística; el modelo es {type(model).__name__}."
        )
    preprocess = pipeline.named_steps["preprocess"]
    columns = preprocess.named_steps["columns"]
    transformed = preprocess.transform(customer)[0]
    coef = model.coef_[0]
    raw = add_features(customer).iloc[0]

    factors: list[Factor] = []
    position = 0
    for name, transformer, cols in columns.transformers_:
        if name == "num":
            for col in cols:
                factors.append(
                    Factor(col, FEATURE_LABELS.get(col, col), _format_value(raw[col]),
                           float(coef[position] * transformed[position]))
                )
                position += 1
        elif name == "cat":
            for col, categories in zip(cols, transformer.categories_):
                for category in categories:
                    active = transformed[position]
                    if active:  # solo la categoría activa del cliente aporta
                        factors.append(
                            Factor(col, FEATURE_LABELS.get(col, col), str(category),
                                   float(coef[position] * active))
                        )
                    position += 1
    factors.sort(key=lambda f: abs(f.contribution), reverse=True)
    return Explanation(intercept=float(model.intercept_[0]), factors=factors)


def top_factors(explanation: Explanation, k: int = 3) -> tuple[list[Factor], list[Factor]]:
    """(k factores que más aumentan el riesgo, k factores que más lo reducen)."""
    up = [f for f in explanation.factors if f.contribution > 0][:k]
    down = [f for f in explanation.factors if f.contribution < 0][:k]
    return up, down


def retrieval_query(factors: list[Factor]) -> str:
    """Texto de consulta para el recuperador a partir de los factores más relevantes."""
    return " ".join(f"{f.label} {f.feature} {f.value}" for f in factors)
