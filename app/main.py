"""MakanPredict API — a FastAPI service over the Project 1 grocery price-tier model.

Given a grocery item (or category), a store type and a state, it predicts whether
the price is likely to be budget / fair / premium relative to that item's national
median — and returns the probability of each tier. It never takes a price as input
(the price defines the label). A separate /price-check applies the label rule
directly to a price you enter.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.catalog import get_artifact, get_metadata, price_check
from app.predict import predict_price_tier
from app.schemas import (
    HealthResponse,
    MetadataResponse,
    PredictRequest,
    PredictResponse,
    PriceCheckRequest,
    PriceCheckResponse,
)

DESCRIPTION = """
Predicts the **price tier** (budget / fair / premium) of a Malaysian grocery item
for a given **store type** and **state**, using an XGBoost model trained on DOSM
PriceCatcher data (Project 1).

* **POST `/predict`** — the ML prediction (tier + confidence + all three probabilities).
* **POST `/price-check`** — deterministic: compare a price you saw to the item's
  national median. This is the *label rule*, not the ML model.
* **GET `/metadata`** — valid premise types / states / categories / items + model info.
* **GET `/health`** — liveness + model-loaded check.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the 16 MB model + build metadata ONCE, at startup — so requests never
    # pay the load cost and we fail fast if the artifact is missing.
    get_artifact()
    get_metadata()
    yield


app = FastAPI(
    title="MakanPredict API",
    version="1.0.0",
    description=DESCRIPTION,
    lifespan=lifespan,
)


@app.get("/", tags=["meta"])
def root():
    md = get_metadata()
    return {
        "name": "MakanPredict API",
        "description": (
            "Predicts whether a grocery item's price is budget / fair / premium for a "
            "given store type and state, using a Project 1 XGBoost model on DOSM "
            "PriceCatcher data."
        ),
        "model": md["model_name"],
        "weighted_f1": md["weighted_f1"],
        "endpoints": {
            "POST /predict": "ML price-tier prediction (item/category + premise_type + state)",
            "POST /price-check": "Deterministic verdict: a price vs the item's national median",
            "GET /metadata": "Valid premise types / states / categories / items + model info",
            "GET /health": "Liveness + model-loaded check",
            "GET /docs": "Interactive API docs",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    try:
        art = get_artifact()
        return HealthResponse(status="ok", model_loaded=True, model_name=art.get("model_name"))
    except Exception:
        return HealthResponse(status="error", model_loaded=False, model_name=None)


@app.get("/metadata", response_model=MetadataResponse, tags=["meta"])
def metadata():
    return get_metadata()


@app.post("/predict", response_model=PredictResponse, tags=["predict"])
def predict(req: PredictRequest):
    """Predict the price tier for one item-at-a-store context."""
    payload = req.model_dump(exclude_none=True)
    try:
        return predict_price_tier(payload)
    except ValueError as exc:  # defensive — schema validation should catch these first
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/price-check", response_model=PriceCheckResponse, tags=["predict"])
def price_check_endpoint(req: PriceCheckRequest):
    """Deterministic verdict: compare a price to the item's national median (label rule)."""
    try:
        return price_check(req.item, req.price)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
