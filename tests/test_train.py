import json

import joblib

from src.train import main


def test_train_end_to_end_on_synthetic_data(synthetic_csv, tmp_path):
    models_dir, reports_dir = tmp_path / "models", tmp_path / "reports"
    metrics = main(
        [
            "--data", str(synthetic_csv),
            "--models-dir", str(models_dir),
            "--reports-dir", str(reports_dir),
        ]
    )
    assert (models_dir / "churn_pipeline.joblib").exists()
    for name in ["cv_results.csv", "results.md", "metrics.json", "precision_recall.png"]:
        assert (reports_dir / name).exists()

    bundle = joblib.load(models_dir / "churn_pipeline.joblib")
    assert 0.05 <= bundle["threshold"] <= 0.95
    assert 0.5 < metrics["test"]["roc_auc"] <= 1.0
    saved = json.loads((reports_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved["best_model"] == bundle["model_name"]
