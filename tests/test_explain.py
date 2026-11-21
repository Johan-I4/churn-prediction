import math

import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.explain import (
    UnsupportedModelError,
    explain_customer,
    retrieval_query,
    top_factors,
)
from src.modeling import build_pipeline
from src.schemas import CustomerFeatures

VALID = CustomerFeatures.model_config["json_schema_extra"]["example"]


@pytest.fixture(scope="module")
def pipeline(model_path):
    return joblib.load(model_path)["pipeline"]


def test_contributions_reproduce_the_model_probability(pipeline, xy):
    """Identidad exacta: sigmoide(intercepto + suma de contribuciones) == predict_proba."""
    X, _ = xy
    for i in (0, 1, 7, 42, 300):
        row = X.iloc[[i]]
        explanation = explain_customer(pipeline, row)
        expected = float(pipeline.predict_proba(row)[0, 1])
        assert explanation.probability == pytest.approx(expected, abs=1e-9)


def test_identity_holds_for_the_api_example(pipeline):
    row = pd.DataFrame([VALID])
    explanation = explain_customer(pipeline, row)
    assert explanation.probability == pytest.approx(float(pipeline.predict_proba(row)[0, 1]), abs=1e-9)


def test_one_factor_per_variable(pipeline):
    explanation = explain_customer(pipeline, pd.DataFrame([VALID]))
    features = [f.feature for f in explanation.factors]
    assert len(features) == len(set(features)) == 22  # 7 numéricas + 15 categóricas


def test_factors_are_sorted_and_effect_matches_sign(pipeline):
    factors = explain_customer(pipeline, pd.DataFrame([VALID])).factors
    magnitudes = [abs(f.contribution) for f in factors]
    assert magnitudes == sorted(magnitudes, reverse=True)
    for f in factors:
        assert f.effect == ("aumenta" if f.contribution > 0 else "reduce")


def test_categorical_factor_carries_the_customer_value(pipeline):
    factors = {f.feature: f for f in explain_customer(pipeline, pd.DataFrame([VALID])).factors}
    assert factors["Contract"].value == "Month-to-month"
    assert factors["tenure"].value == "5"


def test_top_factors_split_by_sign(pipeline):
    explanation = explain_customer(pipeline, pd.DataFrame([VALID]))
    up, down = top_factors(explanation, k=3)
    assert len(up) <= 3 and len(down) <= 3
    assert all(f.contribution > 0 for f in up)
    assert all(f.contribution < 0 for f in down)


def test_non_logistic_model_is_rejected(xy):
    X, y = xy
    forest = build_pipeline(RandomForestClassifier(n_estimators=10, random_state=0)).fit(X, y)
    with pytest.raises(UnsupportedModelError, match="regresión logística"):
        explain_customer(forest, X.iloc[[0]])


def test_retrieval_query_uses_labels_and_values(pipeline):
    up, _ = top_factors(explain_customer(pipeline, pd.DataFrame([VALID])), k=2)
    query = retrieval_query(up)
    assert all(f.value in query for f in up)
    assert not math.isnan(len(query))
