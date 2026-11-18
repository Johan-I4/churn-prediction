"""Datos sintéticos con el mismo esquema que Telco Customer Churn.

Solo sirven para probar el código sin depender del dataset real.
Las métricas obtenidas con ellos NO significan nada.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_telco(n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    yes_no = lambda: rng.choice(["Yes", "No"], n)  # noqa: E731

    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.2])
    add_on = lambda: np.where(  # noqa: E731
        internet == "No", "No internet service", rng.choice(["Yes", "No"], n)
    )
    phone = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
    multiple = np.where(phone == "No", "No phone service", rng.choice(["Yes", "No"], n))
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2])
    tenure = rng.integers(0, 73, n)
    monthly = np.round(rng.uniform(18, 118, n), 2)
    total = np.round(monthly * tenure, 2)

    logit = (
        -0.5
        + 1.2 * (contract == "Month-to-month")
        - 0.03 * tenure
        + 0.8 * (internet == "Fiber optic")
        + 0.01 * (monthly - 65)
    )
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))

    df = pd.DataFrame(
        {
            "customerID": [f"ID-{i:05d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Partner": yes_no(),
            "Dependents": yes_no(),
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": add_on(),
            "OnlineBackup": add_on(),
            "DeviceProtection": add_on(),
            "TechSupport": add_on(),
            "StreamingTV": add_on(),
            "StreamingMovies": add_on(),
            "Contract": contract,
            "PaperlessBilling": yes_no(),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ),
            "MonthlyCharges": monthly,
            "TotalCharges": total.astype(str),
            "Churn": np.where(churn, "Yes", "No"),
        }
    )
    # Como en el dataset real: clientes con tenure 0 traen TotalCharges vacío (" ")
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "
    return df
