"""Construye la base SQLite (3 tablas) a partir del CSV de Telco Customer Churn.

Uso:
    python -m src.build_db
    python -m src.build_db --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --db data/churn.db

Se puede ejecutar las veces que quieras: recrea las tablas desde cero.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from .data import DEFAULT_DATA_PATH

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"
DEFAULT_DB_PATH = Path("data/churn.db")

# tabla -> {columna del CSV: columna en SQL}
TABLE_COLUMNS = {
    "clientes": {
        "customerID": "customer_id",
        "gender": "gender",
        "SeniorCitizen": "senior_citizen",
        "Partner": "partner",
        "Dependents": "dependents",
        "tenure": "tenure",
        "Churn": "churn",
    },
    "servicios": {
        "customerID": "customer_id",
        "PhoneService": "phone_service",
        "MultipleLines": "multiple_lines",
        "InternetService": "internet_service",
        "OnlineSecurity": "online_security",
        "OnlineBackup": "online_backup",
        "DeviceProtection": "device_protection",
        "TechSupport": "tech_support",
        "StreamingTV": "streaming_tv",
        "StreamingMovies": "streaming_movies",
    },
    "facturacion": {
        "customerID": "customer_id",
        "Contract": "contract",
        "PaperlessBilling": "paperless_billing",
        "PaymentMethod": "payment_method",
        "MonthlyCharges": "monthly_charges",
        "TotalCharges": "total_charges",
    },
}


def load_raw(path: str | Path) -> pd.DataFrame:
    """Lee el CSV conservando customerID y aplica la misma limpieza que src/data.py."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No encontré el dataset en '{path}'. Descárgalo de Kaggle y colócalo ahí "
            "(mira la sección 'Datos' del README)."
        )
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["Churn"] = (df["Churn"].astype(str).str.strip() == "Yes").astype(int)
    return df


def build_db(csv_path: str | Path = DEFAULT_DATA_PATH, db_path: str | Path = DEFAULT_DB_PATH) -> dict:
    """Crea la BD y devuelve cuántas filas quedaron en cada tabla."""
    df = load_raw(csv_path)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        conn.execute("PRAGMA foreign_keys = ON")
        # clientes primero: las otras dos tablas dependen de ella (llave foránea)
        for table in ("clientes", "servicios", "facturacion"):
            mapping = TABLE_COLUMNS[table]
            rows = df[list(mapping)].astype(object).values.tolist()
            placeholders = ", ".join("?" * len(mapping))
            conn.executemany(
                f"INSERT INTO {table} ({', '.join(mapping.values())}) VALUES ({placeholders})", rows
            )
        conn.commit()

        problems = conn.execute("PRAGMA foreign_key_check").fetchall()
        if problems:
            raise RuntimeError(f"Hay filas huérfanas en la base: {problems[:3]}")
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLE_COLUMNS}
    finally:
        conn.close()


def main(argv=None):
    p = argparse.ArgumentParser(description="Crea la base SQLite del proyecto de churn")
    p.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    p.add_argument("--db", default=str(DEFAULT_DB_PATH))
    args = p.parse_args(argv)

    counts = build_db(args.data, args.db)
    print(f"Base creada en {args.db}")
    for table, n in counts.items():
        print(f"  {table:12s} {n:>6,} filas")


if __name__ == "__main__":
    main()
