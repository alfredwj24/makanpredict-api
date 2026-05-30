"""Single-item prediction interface for MakanPredict.

Loads the trained model artifact and predicts the price tier of a food item from its context
(item, store type, state). This is the function the FastAPI follow-up project imports.

    from app.predict import predict_price_tier

    predict_price_tier({
        "item": "AYAM BERSIH - STANDARD",   # or "item_category": "AYAM"
        "premise_type": "Pasar Mini",
        "state": "Selangor",
    })
    # -> {"prediction": "premium", "confidence": 0.61,
    #     "probabilities": {"budget": 0.12, "fair": 0.27, "premium": 0.61}}

Note: no raw price is accepted — the price *defines* the label, so feeding it in would be
circular. The model answers "for this item, store type and state, is the price likely to be
budget/fair/premium?".
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

# Allow `from src.predict import ...` and `python src/predict.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from app.features import build_features  # noqa: E402

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "price_classifier.pkl"
CONTEXT_FIELDS = ("premise_type", "state")  # required, plus one of item / item_category
_UNKNOWN = "UNKNOWN"


@lru_cache(maxsize=2)
def _load_artifact(model_path: str):
    """Load and cache the trained model artifact."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Train it first with `python -m src.train`."
        )
    return joblib.load(path)


def _resolve_row(payload: dict, artifact: dict) -> dict:
    """Validate the payload and assemble the single feature row build_features expects."""
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict.")

    missing = [f for f in CONTEXT_FIELDS if not payload.get(f)]
    if not payload.get("item") and not payload.get("item_category"):
        missing.append("item (or item_category)")
    if missing:
        raise ValueError(
            "Missing required field(s): " + ", ".join(missing)
            + ". Provide 'premise_type', 'state', and one of 'item' / 'item_category'."
        )

    row = {
        "premise_type": payload["premise_type"],
        "state": payload["state"],
        "district": payload.get("district"),  # optional; falls back to a global average
        "item_code": -1,
        "item_category": payload.get("item_category", _UNKNOWN),
        "item_group": payload.get("item_group", _UNKNOWN),
        "unit": payload.get("unit", _UNKNOWN),
    }

    if payload.get("item"):
        catalog = artifact["item_catalog"]
        match = catalog[catalog["item"].str.casefold() == str(payload["item"]).casefold()]
        if len(match):
            best = match.iloc[0]
            row.update(
                item_code=int(best["item_code"]),
                item_category=best["item_category"],
                item_group=best["item_group"],
                unit=best["unit"],
            )
        # Unknown item name falls back to any provided item_category / global averages.
    return row


def predict_price_tier(payload: dict, model_path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    """Predict the price tier for one item-at-a-store context.

    Returns a dict with `prediction`, `confidence` and per-class `probabilities`. Raises
    ValueError if required fields are missing.
    """
    artifact = _load_artifact(str(model_path))
    row = _resolve_row(payload, artifact)

    features = build_features(pd.DataFrame([row]), artifact["reference"])
    proba = artifact["pipeline"].predict_proba(features)[0]
    classes = artifact["classes"]
    probabilities = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
    prediction = max(probabilities, key=probabilities.get)

    return {
        "prediction": prediction,
        "confidence": probabilities[prediction],
        "probabilities": probabilities,
    }


if __name__ == "__main__":
    examples = [
        {"item": "AYAM BERSIH - STANDARD", "premise_type": "Pasar Basah", "state": "Sabah"},
        {"item": "AYAM BERSIH - STANDARD", "premise_type": "Hypermarket", "state": "Kedah"},
        {"item_category": "BERAS", "premise_type": "Pasar Mini", "state": "W.P. Kuala Lumpur"},
    ]
    for ex in examples:
        print(ex, "->", predict_price_tier(ex))
