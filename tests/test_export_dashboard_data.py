"""Pruebas del exportador de datos para Power BI (src/export_dashboard_data.py).

- Las pruebas con datos sintéticos corren siempre (como el resto de la suite).
- Las pruebas "con datos reales" comprueban las cifras del README y se omiten solas si
  falta el CSV real o el modelo entrenado (models/*.joblib no se sube a GitHub).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.data import DEFAULT_DATA_PATH, load_xy
from src.evaluate import classification_summary
from src.explain import explain_customer, top_factors
from src.export_dashboard_data import (
    CLIENT_COLUMNS,
    RISK_LABELS,
    SUMMARY_COLUMNS,
    TENURE_LABELS,
    ExportConsistencyError,
    assign_split,
    build_clients_table,
    export,
    risk_level,
    tenure_range,
)
from src.modeling import build_pipeline

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.4
SEED, TEST_SIZE = 42, 0.2


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def trained(synthetic_csv, tmp_path_factory):
    """Modelo entrenado SOLO con el 80 % (igual que train.py) + metrics.json coherente."""
    X, y = load_xy(synthetic_csv)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    pipeline = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    bundle = {"pipeline": pipeline, "threshold": THRESHOLD, "model_name": "test-model", "sklearn_version": "x"}
    test_metrics = classification_summary(y_te, pipeline.predict_proba(X_te)[:, 1], THRESHOLD)
    tmp = tmp_path_factory.mktemp("export")
    model_path, metrics_path = tmp / "model.joblib", tmp / "metrics.json"
    joblib.dump(bundle, model_path)
    metrics_path.write_text(json.dumps({"best_model": "test-model", "test": test_metrics}), encoding="utf-8")
    return {"bundle": bundle, "model_path": model_path, "metrics_path": metrics_path, "metrics": test_metrics}


@pytest.fixture(scope="module")
def exported(synthetic_csv, trained, tmp_path_factory):
    out = tmp_path_factory.mktemp("powerbi_data")
    paths = export(synthetic_csv, trained["model_path"], trained["metrics_path"], out)
    return {
        "paths": paths,
        "clientes": pd.read_csv(paths["clientes"], encoding="utf-8"),
        "resumen": pd.read_csv(paths["resumen_modelo"], encoding="utf-8"),
    }


# --------------------------------------------------------------------------- funciones puras
def test_tenure_range_uses_the_same_cuts_as_sql_query_03():
    t = pd.Series([0, 6, 7, 12, 13, 24, 25, 48, 49, 72])
    assert tenure_range(t).tolist() == [
        "0-6 meses", "0-6 meses", "7-12 meses", "7-12 meses", "13-24 meses",
        "13-24 meses", "25-48 meses", "25-48 meses", "49-72 meses", "49-72 meses",
    ]
    assert set(tenure_range(pd.Series(range(0, 73)))) == set(TENURE_LABELS)


def test_risk_level_boundaries_are_anchored_to_the_threshold():
    t = 0.34
    p = np.array([0.0, 0.169, 0.17, 0.339, 0.34, 0.679, 0.68, 1.0])
    assert risk_level(p, t).tolist() == [
        "Bajo", "Bajo", "Medio", "Medio", "Alto", "Alto", "Muy alto", "Muy alto",
    ]


def test_risk_level_alto_or_more_is_exactly_the_model_prediction():
    p = np.linspace(0, 1, 1001)
    t = 0.34
    flagged = risk_level(p, t).isin(["Alto", "Muy alto"]).to_numpy()
    assert (flagged == (p >= t)).all()


def test_assign_split_reproduces_the_train_py_split(xy):
    X, y = xy
    _, X_test, _, _ = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    split = assign_split(y, TEST_SIZE, SEED)
    assert set(split[split == "test"].index) == set(X_test.index)
    assert len(split) == len(y)


# --------------------------------------------------------------------------- CSV exportados
def test_files_are_utf8_without_bom_and_have_the_agreed_columns(exported):
    for path in exported["paths"].values():
        raw = Path(path).read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf")
        raw.decode("utf-8")
    assert list(exported["clientes"].columns) == CLIENT_COLUMNS
    assert list(exported["resumen"].columns) == SUMMARY_COLUMNS
    assert len(exported["resumen"]) == 1


def test_one_row_per_customer_and_ids_match_the_source(exported, synthetic_csv):
    src = pd.read_csv(synthetic_csv)
    c = exported["clientes"]
    assert len(c) == len(src)
    assert c["customer_id"].is_unique
    assert c["customer_id"].tolist() == src["customerID"].tolist()
    assert c["churn"].mean() == pytest.approx((src["Churn"] == "Yes").mean())


def test_key_columns_have_no_missing_values(exported):
    c = exported["clientes"]
    keys = [
        "customer_id", "Contract", "InternetService", "PaymentMethod", "tenure", "tenure_range",
        "MonthlyCharges", "TotalCharges", "churn", "split", "churn_probability",
        "predicted_churn", "risk_level", "risk_order",
    ]
    assert not c[keys].isna().any().any()
    assert set(c["split"]) == {"train", "test"}
    assert set(c["risk_level"]) <= set(RISK_LABELS)


def test_split_has_about_20_percent_test_and_is_stratified(exported):
    c = exported["clientes"]
    test = c[c["split"] == "test"]
    assert len(test) == pytest.approx(0.2 * len(c), abs=1)
    assert test["churn"].mean() == pytest.approx(c["churn"].mean(), abs=0.01)
    assert exported["resumen"].loc[0, "n_test"] == len(test)
    assert exported["resumen"].loc[0, "n_train"] == len(c) - len(test)


def test_prediction_uses_the_threshold_saved_in_the_model(exported, trained):
    c = exported["clientes"]
    assert exported["resumen"].loc[0, "threshold"] == THRESHOLD
    assert (c["predicted_churn"] == (c["churn_probability"] >= THRESHOLD).astype(int)).all()
    assert c["churn_probability"].between(0, 1).all()


def test_probabilities_match_the_pipeline_and_the_api_logic(exported, trained, xy):
    X, _ = xy
    expected = trained["bundle"]["pipeline"].predict_proba(X)[:, 1]
    assert exported["clientes"]["churn_probability"].to_numpy() == pytest.approx(expected, abs=1e-12)


def test_top_factor_is_the_one_from_src_explain(exported, trained, xy):
    X, _ = xy
    c = exported["clientes"]
    pipeline = trained["bundle"]["pipeline"]
    for i in (0, 5, 99, 400):
        up, _ = top_factors(explain_customer(pipeline, X.iloc[[i]]), k=1)
        assert c.loc[i, "top_factor_feature"] == up[0].feature
        assert c.loc[i, "top_factor_label"] == up[0].label
        assert c.loc[i, "top_factor_contribution"] == pytest.approx(up[0].contribution, abs=1e-6)
        assert c.loc[i, "top_factor_text"] == f"{up[0].label} = {up[0].value}"
    assert (c["top_factor_contribution"].dropna() > 0).all()  # solo factores que SUBEN el riesgo


def test_non_ascii_labels_survive_the_utf8_round_trip(exported):
    labels = set(exported["clientes"]["top_factor_label"].dropna())
    assert any("ü" in s or "ú" in s or "é" in s for s in labels)  # Antigüedad, Número, Género...


def test_orders_are_consistent_with_labels(exported):
    c = exported["clientes"]
    assert c.groupby("tenure_range")["tenure_range_order"].nunique().max() == 1
    assert c.groupby("risk_level")["risk_order"].nunique().max() == 1
    assert c.drop_duplicates("tenure_range").set_index("tenure_range")["tenure_range_order"].to_dict() == {
        label: i + 1 for i, label in enumerate(TENURE_LABELS) if label in set(c["tenure_range"])
    }


# --------------------------------------------------------------------------- leakage
def test_test_metrics_recomputed_from_split_test_match_metrics_json(exported, trained):
    c = exported["clientes"]
    test = c[c["split"] == "test"]
    got = classification_summary(test["churn"], test["churn_probability"], THRESHOLD)
    for key in ("roc_auc", "precision", "recall"):
        assert got[key] == pytest.approx(trained["metrics"][key], abs=1e-6)
        assert exported["resumen"].loc[0, f"test_{key}"] == pytest.approx(trained["metrics"][key], abs=1e-9)


def test_summary_metrics_come_from_metrics_json_not_from_all_rows(exported, trained):
    """Si se calcularan con las 800 filas el AUC cambiaría: el resumen debe traer el del test."""
    c = exported["clientes"]
    all_rows = classification_summary(c["churn"], c["churn_probability"], THRESHOLD)
    assert exported["resumen"].loc[0, "test_roc_auc"] != pytest.approx(all_rows["roc_auc"], abs=1e-9)
    cm = trained["metrics"]["confusion_matrix"]
    r = exported["resumen"].loc[0]
    assert (r["test_tn"], r["test_fp"], r["test_fn"], r["test_tp"]) == (cm["tn"], cm["fp"], cm["fn"], cm["tp"])
    assert r["test_tn"] + r["test_fp"] + r["test_fn"] + r["test_tp"] == r["n_test"]


def test_export_refuses_when_metrics_do_not_match_the_reproduced_split(synthetic_csv, trained, tmp_path):
    bad = json.loads(trained["metrics_path"].read_text(encoding="utf-8"))
    bad["test"]["roc_auc"] = bad["test"]["roc_auc"] - 0.05
    bad_path = tmp_path / "metrics.json"
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ExportConsistencyError, match="roc_auc"):
        export(synthetic_csv, trained["model_path"], bad_path, tmp_path / "out")
    assert not (tmp_path / "out" / "clientes.csv").exists()


def test_export_refuses_a_different_seed_than_the_training_one(synthetic_csv, trained, tmp_path):
    with pytest.raises(ExportConsistencyError):
        export(synthetic_csv, trained["model_path"], trained["metrics_path"], tmp_path / "out", seed=7)


def test_export_fails_clearly_without_model_or_metrics(synthetic_csv, trained, tmp_path):
    with pytest.raises(FileNotFoundError, match="src.train"):
        export(synthetic_csv, tmp_path / "no.joblib", trained["metrics_path"], tmp_path / "o")
    with pytest.raises(FileNotFoundError, match="src.train"):
        export(synthetic_csv, trained["model_path"], tmp_path / "no.json", tmp_path / "o")


def test_non_logistic_model_leaves_factor_columns_empty(synthetic_csv, tmp_path):
    X, y = load_xy(synthetic_csv)
    forest = build_pipeline(RandomForestClassifier(n_estimators=20, random_state=0)).fit(X, y)
    bundle = {"pipeline": forest, "threshold": THRESHOLD, "model_name": "rf"}
    table = build_clients_table(synthetic_csv, bundle)
    assert table["top_factor_text"].eq("").all()
    assert table["top_factor_contribution"].isna().all()
    assert table["churn_probability"].between(0, 1).all()


# --------------------------------------------------------------------------- higiene del repo
def test_generated_csvs_and_power_bi_cache_are_git_ignored():
    lines = [ln.strip() for ln in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()]
    assert "powerbi/data/*.csv" in lines
    assert "**/.pbi/cache.abf" in lines
    assert "**/.pbi/localSettings.json" in lines


# --------------------------------------------------------------------------- cifras del README (datos reales)
real_data = pytest.mark.skipif(not (ROOT / DEFAULT_DATA_PATH).exists(), reason="falta el CSV real")
real_model = pytest.mark.skipif(
    not (ROOT / "models" / "churn_pipeline.joblib").exists() or not (ROOT / "reports" / "metrics.json").exists(),
    reason="falta models/churn_pipeline.joblib (entrena con: python -m src.train)",
)


@pytest.fixture(scope="module")
def real_export(tmp_path_factory):
    out = tmp_path_factory.mktemp("real_powerbi")
    paths = export(
        ROOT / DEFAULT_DATA_PATH, ROOT / "models" / "churn_pipeline.joblib",
        ROOT / "reports" / "metrics.json", out,
    )
    return pd.read_csv(paths["clientes"]), pd.read_csv(paths["resumen_modelo"])


@real_data
@real_model
def test_real_headline_numbers_match_the_readme(real_export):
    c, _ = real_export
    assert len(c) == 7043
    assert round(100 * c["churn"].mean(), 2) == 26.54
    assert (c["split"] == "test").sum() == 1409


@real_data
@real_model
def test_real_segments_match_the_readme_and_sql_queries(real_export):
    c, _ = real_export
    pct = lambda col: (100 * c.groupby(col)["churn"].mean()).round(2).to_dict()  # noqa: E731
    assert pct("Contract") == {"Month-to-month": 42.71, "One year": 11.27, "Two year": 2.83}
    assert pct("tenure_range") == {
        "0-6 meses": 52.94, "7-12 meses": 35.89, "13-24 meses": 28.71,
        "25-48 meses": 20.39, "49-72 meses": 9.51,
    }
    mm = c[c["Contract"] == "Month-to-month"]
    assert round(100 * mm.loc[mm["churn"] == 1, "MonthlyCharges"].sum() / mm["MonthlyCharges"].sum(), 2) == 46.97
    mmf = mm[mm["InternetService"] == "Fiber optic"]
    assert round(100 * mmf["churn"].mean(), 2) == 54.61
    ec = c[(c["Contract"] == "Month-to-month") & (c["PaymentMethod"] == "Electronic check")]
    assert round(100 * ec["churn"].mean(), 2) == 53.73


@real_data
@real_model
def test_real_test_metrics_match_the_readme(real_export):
    c, summary = real_export
    t = c[c["split"] == "test"]
    got = classification_summary(t["churn"], t["churn_probability"], 0.34)
    assert round(got["roc_auc"], 4) == 0.8468
    assert round(100 * got["recall"], 2) == 72.73
    assert round(100 * got["precision"], 2) == 56.20
    assert (got["confusion_matrix"]["tp"], got["confusion_matrix"]["fp"]) == (272, 212)
    assert (got["confusion_matrix"]["fn"], got["confusion_matrix"]["tn"]) == (102, 823)
    r = summary.loc[0]
    assert r["threshold"] == 0.34 and r["n_test"] == 1409
    assert round(r["test_roc_auc"], 4) == 0.8468
