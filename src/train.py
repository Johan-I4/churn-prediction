"""Entrena, evalúa y guarda el modelo de churn.

Uso:
    python -m src.train --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv

Genera:
    models/churn_pipeline.joblib   pipeline + umbral (lo usa la API)
    reports/cv_results.csv         tabla de validación cruzada
    reports/results.md             resultados listos para pegar en el README
    reports/metrics.json           métricas en test
    reports/precision_recall.png   curva precision-recall con el umbral elegido
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (  # noqa: E402
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)

from .data import DEFAULT_DATA_PATH, load_xy  # noqa: E402
from .evaluate import classification_summary, net_benefit, pick_threshold  # noqa: E402
from .modeling import get_models  # noqa: E402

SCORING = {
    "AUC": "roc_auc",
    # zero_division=0: el baseline nunca predice churn y no debe lanzar warnings
    "Precision": make_scorer(precision_score, zero_division=0),
    "Recall": make_scorer(recall_score, zero_division=0),
    "F1": make_scorer(f1_score, zero_division=0),
    "Accuracy": "accuracy",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Entrena el modelo de churn")
    p.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    p.add_argument("--models-dir", default="models")
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--cv-folds", type=int, default=5)
    # Supuestos de costos (edítalos según tu caso de negocio)
    p.add_argument("--value-lost", type=float, default=500.0, help="valor de un cliente perdido")
    p.add_argument("--offer-cost", type=float, default=50.0, help="costo de una oferta de retención")
    p.add_argument("--success-rate", type=float, default=0.3, help="prob. de que la oferta retenga")
    return p.parse_args(argv)


def df_to_markdown(df: pd.DataFrame, index_name: str = "Modelo") -> str:
    cols = [index_name] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for name, row in df.iterrows():
        lines.append("| " + " | ".join([str(name)] + [str(v) for v in row.values]) + " |")
    return "\n".join(lines)


def run_cv(models: dict, X, y, cv) -> pd.DataFrame:
    rows, means = {}, {}
    for name, pipe in models.items():
        res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=None)
        rows[name] = {
            m: f"{res[f'test_{m}'].mean():.4f} ± {res[f'test_{m}'].std():.4f}" for m in SCORING
        }
        means[name] = res["test_AUC"].mean()
        print(f"  {name:32s} AUC={means[name]:.4f}")
    table = pd.DataFrame(rows).T
    order = sorted(means, key=means.get, reverse=True)
    return table.loc[order], means


def plot_pr(y, proba, threshold, path: Path):
    precision, recall, _ = precision_recall_curve(y, proba)
    summary = classification_summary(y, proba, threshold)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recall, precision, lw=2, label="Curva precision-recall (CV)")
    ax.scatter(
        [summary["recall"]],
        [summary["precision"]],
        color="crimson",
        zorder=3,
        label=f"Umbral elegido = {threshold:.2f}",
    )
    ax.axhline(float(np.mean(y)), ls="--", color="gray", label="Tasa base de churn")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall (predicciones out-of-fold en train)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    models_dir, reports_dir = Path(args.models_dir), Path(args.reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    cost = dict(value_lost=args.value_lost, offer_cost=args.offer_cost, success_rate=args.success_rate)

    X, y = load_xy(args.data)
    print(f"Datos: {X.shape[0]} filas, tasa de churn global = {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    # 1) Comparación de modelos con los mismos folds (solo sobre train)
    print(f"Validación cruzada estratificada de {args.cv_folds} folds:")
    models = get_models(args.seed)
    cv_table, means = run_cv(models, X_train, y_train, cv)
    cv_table.to_csv(reports_dir / "cv_results.csv", index_label="Modelo")

    candidates = {k: v for k, v in means.items() if not k.startswith("Baseline")}
    best_name = max(candidates, key=candidates.get)
    print(f"Mejor modelo por AUC en CV: {best_name}")

    # 2) Umbral con supuesto de costos, usando predicciones out-of-fold
    oof = cross_val_predict(clone(models[best_name]), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold, _ = pick_threshold(y_train, oof, **cost)
    plot_pr(y_train, oof, threshold, reports_dir / "precision_recall.png")
    print(f"Umbral elegido: {threshold:.2f}")

    # 3) Entrenamiento final y evaluación en test (una sola vez)
    final = clone(models[best_name]).fit(X_train, y_train)
    proba_test = final.predict_proba(X_test)[:, 1]
    test = classification_summary(y_test, proba_test, threshold)
    baseline_acc = float(1 - y_test.mean()) if y_test.mean() < 0.5 else float(y_test.mean())
    test["baseline_accuracy"] = baseline_acc
    test["net_benefit_model"] = net_benefit(y_test, proba_test, threshold, **cost)
    test["net_benefit_contact_all"] = net_benefit(y_test, proba_test, 0.0, **cost)
    test["net_benefit_contact_none"] = 0.0
    metrics = {"best_model": best_name, "cost_assumptions": cost, "test": test}
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    cm = test["confusion_matrix"]
    md = [
        "## Resultados",
        "",
        f"Datos: {len(X)} clientes, churn global {y.mean():.1%}. "
        f"Split estratificado {int((1 - args.test_size) * 100)}/{int(args.test_size * 100)} "
        f"(semilla {args.seed}).",
        "",
        f"### Validación cruzada ({args.cv_folds} folds, solo train; media ± desviación)",
        "",
        df_to_markdown(cv_table),
        "",
        f"Modelo elegido por AUC en validación cruzada: **{best_name}**.",
        "",
        "### Umbral de decisión",
        "",
        f"Supuestos: un cliente perdido vale {args.value_lost:g}, una oferta de retención cuesta "
        f"{args.offer_cost:g} y retiene al cliente con probabilidad {args.success_rate:g}. "
        f"El umbral que maximiza el beneficio neto (sobre predicciones out-of-fold) es **{threshold:.2f}**.",
        "",
        "![Curva precision-recall](reports/precision_recall.png)",
        "",
        "### Test (usado una sola vez)",
        "",
        "| Métrica | Valor |",
        "|---|---|",
        f"| Accuracy del baseline (clase mayoritaria) | {baseline_acc:.4f} |",
        f"| AUC | {test['roc_auc']:.4f} |",
        f"| Precision (umbral {threshold:.2f}) | {test['precision']:.4f} |",
        f"| Recall (umbral {threshold:.2f}) | {test['recall']:.4f} |",
        f"| F1 (umbral {threshold:.2f}) | {test['f1']:.4f} |",
        f"| Matriz de confusión (TN/FP/FN/TP) | {cm['tn']} / {cm['fp']} / {cm['fn']} / {cm['tp']} |",
        f"| Beneficio neto: modelo vs. contactar a todos | {test['net_benefit_model']:.0f} vs. {test['net_benefit_contact_all']:.0f} |",
        "",
    ]
    (reports_dir / "results.md").write_text("\n".join(md))

    # 4) Guardar el pipeline completo + umbral
    bundle = {
        "pipeline": final,
        "threshold": threshold,
        "model_name": best_name,
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(bundle, models_dir / "churn_pipeline.joblib")

    print(f"\nTest: AUC={test['roc_auc']:.4f}  precision={test['precision']:.3f}  recall={test['recall']:.3f}")
    print(f"Modelo guardado en {models_dir / 'churn_pipeline.joblib'}")
    print(f"Resultados para el README en {reports_dir / 'results.md'}")
    return metrics


if __name__ == "__main__":
    main()
