"""Contract tests for the Project 1 prediction interface (predict_price_tier)."""
import pytest

from app.predict import predict_price_tier
from tests.conftest import CANON


def test_canonical_prediction():
    r = predict_price_tier(CANON)
    assert r["prediction"] == "premium"
    assert r["confidence"] == pytest.approx(0.778, abs=0.01)


def test_response_shape_and_probabilities():
    r = predict_price_tier(CANON)
    assert set(r) == {"prediction", "confidence", "probabilities"}
    assert set(r["probabilities"]) == {"budget", "fair", "premium"}
    assert sum(r["probabilities"].values()) == pytest.approx(1.0, abs=0.02)
    # confidence is the top probability, and prediction is its class
    assert r["confidence"] == max(r["probabilities"].values())
    assert r["prediction"] == max(r["probabilities"], key=r["probabilities"].get)


def test_item_category_only_is_accepted():
    r = predict_price_tier(
        {"item_category": "BERAS", "premise_type": "Pasar Mini", "state": "W.P. Kuala Lumpur"}
    )
    assert r["prediction"] in {"budget", "fair", "premium"}


def test_unknown_item_falls_back_without_crashing():
    r = predict_price_tier(
        {"item": "NONEXISTENT ITEM XYZ", "premise_type": "Hypermarket", "state": "Johor"}
    )
    assert r["prediction"] in {"budget", "fair", "premium"}


@pytest.mark.parametrize(
    "payload",
    [
        {"premise_type": "Pasar Basah", "state": "Sabah"},                  # no item/category
        {"item": "AYAM BERSIH - STANDARD", "state": "Sabah"},              # no premise_type
        {"item": "AYAM BERSIH - STANDARD", "premise_type": "Pasar Basah"},  # no state
    ],
)
def test_missing_required_field_raises_valueerror(payload):
    with pytest.raises(ValueError):
        predict_price_tier(payload)
