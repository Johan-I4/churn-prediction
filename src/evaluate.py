"""Métricas y elección del umbral con un supuesto de costos."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def net_benefit(y_true, proba, threshold, value_lost=500.0, offer_cost=50.0, success_rate=0.3) -> float:
    """Beneficio neto total de contactar a quienes tienen proba >= umbral.

    - Cada cliente contactado cuesta `offer_cost`.
    - Si realmente se iba (verdadero positivo), la oferta lo retiene con
      probabilidad `success_rate` y se evita perder `value_lost`.
    """
    y_true = np.asarray(y_true)
    contacted = np.asarray(proba) >= threshold
    true_pos = int((contacted & (y_true == 1)).sum())
    return true_pos * success_rate * value_lost - int(contacted.sum()) * offer_cost


def pick_threshold(y_true, proba, **cost_kwargs) -> tuple[float, pd.DataFrame]:
    """Busca el umbral que maximiza el beneficio neto. Devuelve (umbral, tabla)."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    rows = [(t, net_benefit(y_true, proba, t, **cost_kwargs)) for t in grid]
    table = pd.DataFrame(rows, columns=["threshold", "net_benefit"])
    best = table.loc[table["net_benefit"].idxmax(), "threshold"]
    return float(best), table


def classification_summary(y_true, proba, threshold) -> dict:
    """Métricas en test a un umbral dado."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
