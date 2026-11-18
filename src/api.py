"""API REST con FastAPI.

    uvicorn src.api:app --reload
    Documentación interactiva: http://127.0.0.1:8000/docs

La ruta del modelo se puede cambiar con la variable de entorno MODEL_PATH.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request

from .schemas import CustomerFeatures, PredictionResponse

DEFAULT_MODEL_PATH = "models/churn_pipeline.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    if not path.exists():
        raise RuntimeError(
            f"No encontré el modelo en '{path}'. Entrénalo primero: python -m src.train"
        )
    app.state.bundle = joblib.load(path)
    yield


app = FastAPI(
    title="Telco Churn API",
    description="Predice la probabilidad de que un cliente abandone el servicio.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request):
    bundle = request.app.state.bundle
    return {"status": "ok", "model": bundle["model_name"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures, request: Request):
    bundle = request.app.state.bundle
    row = pd.DataFrame([customer.model_dump()])
    probability = float(bundle["pipeline"].predict_proba(row)[0, 1])
    threshold = float(bundle["threshold"])
    return PredictionResponse(
        churn_probability=round(probability, 6),
        churn_prediction=probability >= threshold,
        threshold=threshold,
    )
