import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.data import RAW_FEATURES, load_clean
from src.evaluate import net_benefit, pick_threshold
from src.features import add_features
from src.logreg_scratch import LogisticRegressionScratch
from src.modeling import build_pipeline, build_preprocessor


def test_load_clean_fixes_total_charges(synthetic_csv):
    df = load_clean(synthetic_csv)
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])
    assert df["TotalCharges"].isna().sum() == 0
    assert set(df["Churn"].unique()) <= {0, 1}
    assert "customerID" not in df.columns


def test_add_features_values():
    row = pd.DataFrame(
        [
            {
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "MonthlyCharges": 90.0,
                "tenure": 4,
            }
        ]
    )
    out = add_features(row)
    assert out.loc[0, "n_services"] == 4  # Phone + OnlineSecurity + TechSupport + internet
    assert out.loc[0, "monthly_per_tenure"] == 18.0
    assert out.loc[0, "is_new_customer"] == 1


def test_pipeline_predicts_proba_with_expected_shape(xy):
    X, y = xy
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    proba = pipe.predict_proba(X.head(7))
    assert proba.shape == (7, 2)
    assert np.all((proba >= 0) & (proba <= 1))


def test_pipeline_handles_unseen_category(xy):
    X, y = xy
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    new = X.head(1).copy()
    new["PaymentMethod"] = "Cripto"
    assert pipe.predict_proba(new).shape == (1, 2)


def test_preprocessing_is_fit_only_on_train(xy):
    X, y = xy
    X_tr, X_te, _, _ = train_test_split(X, y, test_size=0.3, random_state=0)
    prep = build_preprocessor().fit(X_tr)
    scaler = prep.named_steps["columns"].named_transformers_["num"]
    names = prep.named_steps["columns"].transformers_[0][2]
    idx = names.index("MonthlyCharges")
    assert np.isclose(scaler.mean_[idx], X_tr["MonthlyCharges"].mean())
    assert not np.isclose(scaler.mean_[idx], pd.concat([X_tr, X_te])["MonthlyCharges"].mean())


def test_pick_threshold_prefers_perfect_separation():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    proba = np.array([0.1, 0.1, 0.2, 0.2, 0.8, 0.8, 0.9, 0.9])
    thr, table = pick_threshold(y, proba)
    assert 0.2 < thr <= 0.8
    assert net_benefit(y, proba, thr) > 0
    assert len(table) > 0


def test_logreg_scratch_close_to_sklearn(xy):
    X, y = xy
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=1)
    prep = build_preprocessor()
    Z_tr, Z_te = prep.fit_transform(X_tr), prep.transform(X_te)

    scratch = LogisticRegressionScratch(lr=0.1, n_iter=1500).fit(Z_tr, y_tr)
    sk = LogisticRegression(max_iter=1000).fit(Z_tr, y_tr)

    assert scratch.loss_history_[-1] < scratch.loss_history_[0]
    auc_scratch = roc_auc_score(y_te, scratch.predict_proba(Z_te))
    auc_sk = roc_auc_score(y_te, sk.predict_proba(Z_te)[:, 1])
    assert abs(auc_scratch - auc_sk) < 0.02


def test_raw_features_listed_once():
    assert len(RAW_FEATURES) == len(set(RAW_FEATURES)) == 19
