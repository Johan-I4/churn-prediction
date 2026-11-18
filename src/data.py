"""Carga y limpieza del dataset Telco Customer Churn."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
TARGET = "Churn"

NUMERIC_BASE = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
RAW_FEATURES = NUMERIC_BASE + CATEGORICAL


def load_clean(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Lee el CSV, corrige TotalCharges y convierte Churn a 0/1.

    TotalCharges trae 11 filas vacías (texto " "). Todas son clientes con
    tenure = 0 (aún no han facturado), así que se rellenan con 0.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No encontré el dataset en '{path}'. Descárgalo de Kaggle "
            "(Telco Customer Churn) y colócalo ahí; mira la sección "
            "'Datos' del README."
        )
    df = pd.read_csv(path)

    missing = [c for c in RAW_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df[TARGET] = (df[TARGET].astype(str).str.strip() == "Yes").astype(int)
    return df.drop(columns=["customerID"], errors="ignore")


def load_xy(path: str | Path = DEFAULT_DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Devuelve (X, y) con X en crudo (sin codificar)."""
    df = load_clean(path)
    return df[RAW_FEATURES].copy(), df[TARGET].copy()
