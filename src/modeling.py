"""Construcción del Pipeline de scikit-learn."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC_BASE
from .features import NEW_NUMERIC, add_features

NUMERIC_FEATURES = NUMERIC_BASE + NEW_NUMERIC


def build_preprocessor() -> Pipeline:
    """Variables derivadas + one-hot + escalado. Todo se ajusta solo con el train."""
    columns = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ],
        sparse_threshold=0,  # salida densa (HistGradientBoosting no acepta sparse)
    )
    return Pipeline([("features", FunctionTransformer(add_features)), ("columns", columns)])


def build_pipeline(model) -> Pipeline:
    """Pipeline completo: recibe el DataFrame crudo y devuelve predicciones."""
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def get_models(seed: int = 42) -> dict:
    return {
        "Baseline (clase mayoritaria)": build_pipeline(DummyClassifier(strategy="most_frequent")),
        "Regresión logística": build_pipeline(LogisticRegression(max_iter=1000)),
        "Random Forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=seed
            )
        ),
        "Gradient Boosting (HistGB)": build_pipeline(
            HistGradientBoostingClassifier(random_state=seed)
        ),
    }
