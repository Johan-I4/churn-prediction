"""Esquemas Pydantic de entrada/salida de la API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]


class CustomerFeatures(BaseModel):
    """Un cliente con las mismas columnas del dataset original (sin customerID ni Churn)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 5,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 95.5,
                "TotalCharges": 470.2,
            }
        },
    )

    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Meses como cliente")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoNoInternet
    OnlineBackup: YesNoNoInternet
    DeviceProtection: YesNoNoInternet
    TechSupport: YesNoNoInternet
    StreamingTV: YesNoNoInternet
    StreamingMovies: YesNoNoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class PredictionResponse(BaseModel):
    churn_probability: float = Field(ge=0, le=1)
    churn_prediction: bool
    threshold: float


class FactorOut(BaseModel):
    feature: str
    label: str
    value: str
    contribution: float = Field(description="Contribución al log-odds (positiva = aumenta el riesgo)")
    effect: Literal["aumenta", "reduce"]


class ExplainResponse(PredictionResponse):
    factors_up: list[FactorOut]
    factors_down: list[FactorOut]
    context_sources: list[str] = Field(description="Fragmentos recuperados por el RAG (archivo#título)")
    explanation: str
    backend: str = Field(description="Backend que redactó el texto: template o anthropic")
    fallback_reason: str | None = Field(
        default=None, description="Si el backend pedido falló, por qué se usó template"
    )
