"""API REST con FastAPI.

    uvicorn src.api:app --reload
    Documentación interactiva: http://127.0.0.1:8000/docs

Variables de entorno:
    MODEL_PATH      ruta del modelo (defecto models/churn_pipeline.joblib)
    KNOWLEDGE_DIR   carpeta de documentos para el RAG (defecto docs/knowledge)
    PROMPTS_DIR     carpeta de prompts (defecto prompts)
    LLM_BACKEND     template (defecto) | anthropic | auto; ver src/llm_client.py
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from .explain import (
    ExplanationCase,
    UnsupportedModelError,
    explain_customer,
    retrieval_query,
    top_factors,
)
from .llm_client import LLMConfigError, TemplateBackend, get_explainer
from .rag import Retriever
from .schemas import CustomerFeatures, ExplainResponse, FactorOut, PredictionResponse

logger = logging.getLogger("churn.api")

DEFAULT_MODEL_PATH = "models/churn_pipeline.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    if not path.exists():
        raise RuntimeError(
            f"No encontré el modelo en '{path}'. Entrénalo primero: python -m src.train"
        )
    app.state.bundle = joblib.load(path)

    # /explain es opcional: si falla algo de aquí, /health y /predict siguen funcionando.
    try:
        app.state.retriever = Retriever.from_directory()
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("RAG desactivado: %s", exc)
        app.state.retriever = None
    try:
        app.state.explainer = get_explainer()
    except LLMConfigError as exc:
        logger.warning("Backend LLM no disponible (%s); se usa template.", exc)
        app.state.explainer = TemplateBackend()
    yield


app = FastAPI(
    title="Telco Churn API",
    description="Predice la probabilidad de que un cliente abandone el servicio.",
    version="1.1.0",
    lifespan=lifespan,
)


def _score(bundle: dict, customer: CustomerFeatures) -> tuple[pd.DataFrame, float, float]:
    row = pd.DataFrame([customer.model_dump()])
    probability = float(bundle["pipeline"].predict_proba(row)[0, 1])
    return row, probability, float(bundle["threshold"])


@app.get("/health")
def health(request: Request):
    bundle = request.app.state.bundle
    return {"status": "ok", "model": bundle["model_name"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures, request: Request):
    _, probability, threshold = _score(request.app.state.bundle, customer)
    return PredictionResponse(
        churn_probability=round(probability, 6),
        churn_prediction=probability >= threshold,
        threshold=threshold,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(customer: CustomerFeatures, request: Request):
    """Predicción + factores exactos del modelo + contexto (RAG) + texto en lenguaje de negocio.

    La probabilidad y los factores los calcula siempre el modelo; el texto solo los redacta.
    """
    state = request.app.state
    bundle = state.bundle
    row, probability, threshold = _score(bundle, customer)

    try:
        explanation = explain_customer(bundle["pipeline"], row)
    except UnsupportedModelError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    up, down = top_factors(explanation, k=3)

    context, sources = [], []
    if state.retriever is not None:
        for chunk, _score_value in state.retriever.search(retrieval_query(up + down), k=3):
            context.append(chunk.text)
            sources.append(f"{chunk.source}#{chunk.title}")

    case = ExplanationCase(
        probability=round(probability, 6),
        threshold=threshold,
        predicted_churn=probability >= threshold,
        model_name=bundle["model_name"],
        factors_up=up,
        factors_down=down,
        context=context,
    )

    backend, fallback_reason = state.explainer.name, None
    try:
        text = state.explainer.explain(case)
    except Exception as exc:  # noqa: BLE001 - cualquier fallo externo cae a la plantilla
        logger.warning("Falló el backend '%s': %s", backend, type(exc).__name__)
        text = TemplateBackend().explain(case)
        backend, fallback_reason = "template", f"{state.explainer.name} falló: {type(exc).__name__}"

    to_out = lambda f: FactorOut(  # noqa: E731
        feature=f.feature, label=f.label, value=f.value,
        contribution=round(f.contribution, 6), effect=f.effect,
    )
    return ExplainResponse(
        churn_probability=case.probability,
        churn_prediction=case.predicted_churn,
        threshold=threshold,
        factors_up=[to_out(f) for f in up],
        factors_down=[to_out(f) for f in down],
        context_sources=sources,
        explanation=text,
        backend=backend,
        fallback_reason=fallback_reason,
    )
