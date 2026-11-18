"""Variables derivadas. Se aplican dentro del Pipeline, así la API recibe datos crudos."""
from __future__ import annotations

import pandas as pd

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
NEW_NUMERIC = ["n_services", "monthly_per_tenure", "is_new_customer"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 3 variables sin ajustar nada con los datos (no hay riesgo de leakage).

    - n_services: cantidad de servicios contratados (incluye tener internet).
    - monthly_per_tenure: cargo mensual relativo a la antigüedad.
    - is_new_customer: 1 si lleva 6 meses o menos.
    """
    out = df.copy()
    out["n_services"] = (out[SERVICE_COLUMNS] == "Yes").sum(axis=1) + (
        out["InternetService"] != "No"
    ).astype(int)
    out["monthly_per_tenure"] = out["MonthlyCharges"] / (out["tenure"] + 1)
    out["is_new_customer"] = (out["tenure"] <= 6).astype(int)
    return out
