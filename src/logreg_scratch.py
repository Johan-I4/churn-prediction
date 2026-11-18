"""Regresión logística desde cero con NumPy y descenso de gradiente.

Uso (compara contra scikit-learn con los mismos datos preprocesados):
    python -m src.logreg_scratch --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv

Gradiente de la pérdida log-loss:  dL/dw = Xᵀ(σ(Xw + b) − y) / n
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class LogisticRegressionScratch:
    def __init__(self, lr: float = 0.1, n_iter: int = 2000, l2: float = 0.0):
        self.lr, self.n_iter, self.l2 = lr, n_iter, l2
        self.w_ = None
        self.b_ = 0.0
        self.loss_history_: list[float] = []

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        n, d = X.shape
        self.w_, self.b_, self.loss_history_ = np.zeros(d), 0.0, []
        eps = 1e-12
        for _ in range(self.n_iter):
            p = sigmoid(X @ self.w_ + self.b_)
            error = p - y
            self.w_ -= self.lr * (X.T @ error / n + self.l2 * self.w_)
            self.b_ -= self.lr * error.mean()
            loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            self.loss_history_.append(float(loss))
        return self

    def predict_proba(self, X):
        return sigmoid(np.asarray(X, dtype=float) @ self.w_ + self.b_)


def main(argv=None):
    from .data import DEFAULT_DATA_PATH, load_xy
    from .modeling import build_preprocessor

    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    X, y = load_xy(args.data)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=args.seed)

    prep = build_preprocessor()
    Z_tr = prep.fit_transform(X_tr)  # se ajusta solo con train
    Z_te = prep.transform(X_te)

    scratch = LogisticRegressionScratch(lr=0.1, n_iter=2000).fit(Z_tr, y_tr)
    sk = LogisticRegression(max_iter=1000).fit(Z_tr, y_tr)

    auc_scratch = roc_auc_score(y_te, scratch.predict_proba(Z_te))
    auc_sk = roc_auc_score(y_te, sk.predict_proba(Z_te)[:, 1])
    print(f"Pérdida inicial: {scratch.loss_history_[0]:.4f} -> final: {scratch.loss_history_[-1]:.4f}")
    print(f"AUC NumPy: {auc_scratch:.4f} | AUC scikit-learn: {auc_sk:.4f} | diferencia: {abs(auc_scratch - auc_sk):.4f}")


if __name__ == "__main__":
    main()
