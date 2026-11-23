"""Exporta los datos del dashboard de Power BI (CSV en UTF-8).

Uso (desde la raíz del repo, con el modelo ya entrenado con `python -m src.train`):
    python -m src.export_dashboard_data

Genera en powerbi/data/:
    clientes.csv          1 fila por cliente (7,043): atributos, churn real, `split`,
                          probabilidad, predicción con el umbral del modelo, nivel de
                          riesgo y factor principal que sube el riesgo.
    resumen_modelo.csv    1 fila: umbral y métricas del TEST leídas de reports/metrics.json.

Cuidado con el leakage
----------------------
El modelo se entrenó con el 80 % de los clientes. Aquí se puntúan los 7,043 (para poder
priorizar a quién contactar), así que cada fila lleva `split` = train | test, reproduciendo
el mismo `train_test_split` de `src/train.py`. Las métricas de desempeño (AUC, precision,
recall) NUNCA se calculan con filas de train: en `resumen_modelo.csv` vienen de
`reports/metrics.json` y, antes de escribir nada, se recalculan con las filas `split = test`
para comprobar que la partición reproducida coincide con la del entrenamiento.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import DEFAULT_DATA_PATH, RAW_FEATURES, TARGET, load_clean
from .evaluate import classification_summary
from .explain import UnsupportedModelError, explain_customer, top_factors

DEFAULT_MODEL_PATH = Path("models/churn_pipeline.joblib")
DEFAULT_METRICS_PATH = Path("reports/metrics.json")
DEFAULT_OUT_DIR = Path("powerbi/data")

# Mismos rangos que sql/queries/03_churn_por_antiguedad.sql (tenure en meses).
TENURE_BINS = [-1, 6, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-6 meses", "7-12 meses", "13-24 meses", "25-48 meses", "49-72 meses"]

# Niveles de riesgo anclados al umbral del modelo (decisión de diseño, no una verdad estadística):
#   Bajo      p <  0.5 * umbral
#   Medio     0.5 * umbral <= p < umbral
#   Alto      umbral <= p < 2 * umbral      (el modelo predice churn)
#   Muy alto  p >= 2 * umbral               (el modelo predice churn)
RISK_LABELS = ["Bajo", "Medio", "Alto", "Muy alto"]
LOW_FACTOR = 0.5
VERY_HIGH_FACTOR = 2.0

CLIENT_COLUMNS = [
    "customer_id",
    "Contract",
    "InternetService",
    "PaymentMethod",
    "tenure",
    "tenure_range",
    "tenure_range_order",
    "MonthlyCharges",
    "TotalCharges",
    "churn",
    "split",
    "churn_probability",
    "predicted_churn",
    "risk_level",
    "risk_order",
    "top_factor_feature",
    "top_factor_label",
    "top_factor_value",
    "top_factor_text",
    "top_factor_contribution",
]

SUMMARY_COLUMNS = [
    "model_name",
    "threshold",
    "n_train",
    "n_test",
    "test_roc_auc",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_accuracy",
    "test_tn",
    "test_fp",
    "test_fn",
    "test_tp",
    "sklearn_version",
]


class ExportConsistencyError(ValueError):
    """La partición train/test reproducida no coincide con la del entrenamiento."""


def tenure_range(tenure: pd.Series) -> pd.Series:
    """Rango de antigüedad con los mismos cortes que la consulta SQL 03."""
    return pd.cut(tenure, bins=TENURE_BINS, labels=TENURE_LABELS, right=True).astype(str)


def risk_level(proba: pd.Series | np.ndarray, threshold: float) -> pd.Series:
    """Nivel de riesgo (Bajo/Medio/Alto/Muy alto) a partir de la probabilidad y el umbral."""
    bins = [-np.inf, LOW_FACTOR * threshold, threshold, VERY_HIGH_FACTOR * threshold, np.inf]
    return pd.cut(pd.Series(np.asarray(proba)), bins=bins, labels=RISK_LABELS, right=False).astype(str)


def assign_split(y: pd.Series, test_size: float = 0.2, seed: int = 42) -> pd.Series:
    """'train' / 'test' por cliente, con la MISMA llamada que usa src/train.py."""
    idx_train, idx_test = train_test_split(
        y.index, test_size=test_size, stratify=y, random_state=seed
    )
    split = pd.Series("train", index=y.index, name="split")
    split.loc[idx_test] = "test"
    return split


def top_risk_factors(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Factor que más sube el riesgo de cada cliente, con `src/explain.py` (contribución exacta).

    Si el modelo no es una regresión logística no hay explicación exacta: las columnas quedan vacías.
    """
    empty = {
        "top_factor_feature": "",
        "top_factor_label": "",
        "top_factor_value": "",
        "top_factor_text": "",
        "top_factor_contribution": np.nan,
    }
    rows = []
    for i in range(len(X)):
        try:
            explanation = explain_customer(pipeline, X.iloc[[i]])
        except UnsupportedModelError:
            return pd.DataFrame([empty] * len(X), index=X.index)
        up, _ = top_factors(explanation, k=1)
        if not up:  # ningún factor sube el riesgo de este cliente
            rows.append(dict(empty))
            continue
        f = up[0]
        rows.append(
            {
                "top_factor_feature": f.feature,
                "top_factor_label": f.label,
                "top_factor_value": f.value,
                "top_factor_text": f"{f.label} = {f.value}",
                "top_factor_contribution": round(f.contribution, 6),
            }
        )
    return pd.DataFrame(rows, index=X.index)


def build_clients_table(
    data_path: str | Path, bundle: dict, test_size: float = 0.2, seed: int = 42
) -> pd.DataFrame:
    """Tabla de clientes lista para Power BI (una fila por cliente del CSV original)."""
    data_path = Path(data_path)
    df = load_clean(data_path)  # TotalCharges limpio y churn 0/1; sin customerID
    ids = pd.read_csv(data_path, usecols=["customerID"])["customerID"]
    if len(ids) != len(df) or not ids.is_unique:
        raise ValueError("customerID debe ser único y alinearse fila a fila con el CSV.")

    X, y = df[RAW_FEATURES], df[TARGET]
    pipeline, threshold = bundle["pipeline"], float(bundle["threshold"])

    proba = pipeline.predict_proba(X)[:, 1]
    out = pd.DataFrame(
        {
            "customer_id": ids.to_numpy(),
            "Contract": df["Contract"].to_numpy(),
            "InternetService": df["InternetService"].to_numpy(),
            "PaymentMethod": df["PaymentMethod"].to_numpy(),
            "tenure": df["tenure"].to_numpy(),
            "MonthlyCharges": df["MonthlyCharges"].to_numpy(),
            "TotalCharges": df["TotalCharges"].to_numpy(),
            "churn": y.to_numpy(),
        }
    )
    out["tenure_range"] = tenure_range(out["tenure"])
    out["tenure_range_order"] = out["tenure_range"].map({l: i + 1 for i, l in enumerate(TENURE_LABELS)})
    out["split"] = assign_split(y, test_size=test_size, seed=seed).to_numpy()
    out["churn_probability"] = proba  # sin redondear: la decisión coincide con la de la API
    out["predicted_churn"] = (proba >= threshold).astype(int)
    out["risk_level"] = risk_level(proba, threshold)
    out["risk_order"] = out["risk_level"].map({l: i + 1 for i, l in enumerate(RISK_LABELS)})
    factors = top_risk_factors(pipeline, X).reset_index(drop=True)
    out = pd.concat([out, factors], axis=1)
    return out[CLIENT_COLUMNS]


def verify_split_against_metrics(clients: pd.DataFrame, metrics: dict, threshold: float, tol: float = 1e-3):
    """Recalcula AUC, precision y recall SOLO con split = test y los compara con metrics.json.

    Si no coinciden, la partición reproducida no es la del entrenamiento (otra semilla,
    otro --test-size o un modelo reentrenado sin actualizar metrics.json) y habría riesgo
    de mezclar filas de train en las métricas. En ese caso no se exporta nada.
    """
    test = clients[clients["split"] == "test"]
    recomputed = classification_summary(test["churn"], test["churn_probability"], threshold)
    saved = metrics["test"]
    if abs(float(saved["threshold"]) - threshold) > 1e-9:
        raise ExportConsistencyError(
            f"El umbral del modelo ({threshold}) no coincide con metrics.json ({saved['threshold']}). "
            "Vuelve a entrenar: python -m src.train"
        )
    for key in ("roc_auc", "precision", "recall"):
        if abs(recomputed[key] - float(saved[key])) > tol:
            raise ExportConsistencyError(
                f"{key}: con las filas split=test sale {recomputed[key]:.4f} pero metrics.json dice "
                f"{float(saved[key]):.4f}. La partición train/test reproducida no coincide con la del "
                "entrenamiento; revisa --seed y --test-size, o reentrena con python -m src.train."
            )
    return recomputed


def build_model_summary(clients: pd.DataFrame, bundle: dict, metrics: dict) -> pd.DataFrame:
    """Una fila con el umbral y las métricas del TEST tal como están en metrics.json."""
    test = metrics["test"]
    cm = test["confusion_matrix"]
    row = {
        "model_name": bundle.get("model_name", metrics.get("best_model", "")),
        "threshold": float(bundle["threshold"]),
        "n_train": int((clients["split"] == "train").sum()),
        "n_test": int((clients["split"] == "test").sum()),
        "test_roc_auc": float(test["roc_auc"]),
        "test_precision": float(test["precision"]),
        "test_recall": float(test["recall"]),
        "test_f1": float(test["f1"]),
        "test_accuracy": float(test["accuracy"]),
        "test_tn": int(cm["tn"]),
        "test_fp": int(cm["fp"]),
        "test_fn": int(cm["fn"]),
        "test_tp": int(cm["tp"]),
        "sklearn_version": bundle.get("sklearn_version", ""),
    }
    return pd.DataFrame([row])[SUMMARY_COLUMNS]


def export(
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, Path]:
    """Genera clientes.csv y resumen_modelo.csv (UTF-8, sin índice). Devuelve sus rutas."""
    model_path, metrics_path, out_dir = Path(model_path), Path(metrics_path), Path(out_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"No encontré el modelo en '{model_path}'. Entrénalo: python -m src.train")
    if not metrics_path.exists():
        raise FileNotFoundError(f"No encontré '{metrics_path}'. Genera los reportes: python -m src.train")

    bundle = joblib.load(model_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    clients = build_clients_table(data_path, bundle, test_size=test_size, seed=seed)
    verify_split_against_metrics(clients, metrics, float(bundle["threshold"]))
    summary = build_model_summary(clients, bundle, metrics)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"clientes": out_dir / "clientes.csv", "resumen_modelo": out_dir / "resumen_modelo.csv"}
    clients.to_csv(paths["clientes"], index=False, encoding="utf-8", lineterminator="\n")
    summary.to_csv(paths["resumen_modelo"], index=False, encoding="utf-8", lineterminator="\n")
    return paths


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Exporta los CSV para el dashboard de Power BI")
    p.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    p.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--metrics", default=str(DEFAULT_METRICS_PATH))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    # Deben ser los mismos valores con los que se corrió src/train.py
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    return p.parse_args(argv)


def main(argv=None) -> dict[str, Path]:
    args = parse_args(argv)
    print("Puntuando clientes y calculando el factor principal (tarda alrededor de un minuto)...")
    paths = export(args.data, args.model, args.metrics, args.out_dir, args.test_size, args.seed)
    clients = pd.read_csv(paths["clientes"], encoding="utf-8")
    print(f"Clientes: {len(clients):,}  |  churn global: {clients['churn'].mean():.2%}")
    print(f"Split: {clients['split'].value_counts().to_dict()}")
    print("Comprobación OK: AUC, precision y recall recalculados con split=test coinciden con metrics.json")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return paths


if __name__ == "__main__":
    main()
