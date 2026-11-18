import joblib
import pytest
from sklearn.linear_model import LogisticRegression

from src.data import load_xy
from src.modeling import build_pipeline

from .synthetic import make_synthetic_telco


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "telco.csv"
    make_synthetic_telco(n=800, seed=0).to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def xy(synthetic_csv):
    return load_xy(synthetic_csv)


@pytest.fixture(scope="session")
def model_path(tmp_path_factory, xy):
    """Entrena un modelo rápido con datos sintéticos y lo guarda como lo haría train.py."""
    X, y = xy
    pipeline = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    path = tmp_path_factory.mktemp("models") / "churn_pipeline.joblib"
    joblib.dump({"pipeline": pipeline, "threshold": 0.4, "model_name": "test-model"}, path)
    return path
