"""Endpoint tests for the MakanPredict API (via FastAPI TestClient)."""
import pytest

from tests.conftest import CANON


def test_root_info(client):
    r = client.get("/")
    assert r.status_code == 200
    j = r.json()
    assert j["model"] == "xgboost"
    assert "weighted_f1" in j and "endpoints" in j


def test_health(client):
    j = client.get("/health").json()
    assert j["status"] == "ok"
    assert j["model_loaded"] is True
    assert j["model_name"] == "xgboost"


def test_metadata(client):
    j = client.get("/metadata").json()
    assert j["counts"] == {"items": 252, "item_categories": 33, "premise_types": 5, "states": 16}
    assert j["classes"] == ["budget", "fair", "premium"]
    assert j["weighted_f1"] == pytest.approx(0.739, abs=0.005)
    assert "AYAM BERSIH - STANDARD" in j["items_by_category"]["AYAM"]
    assert len(j["premise_types"]) == 5 and len(j["states"]) == 16


def test_predict_happy_path(client):
    r = client.post("/predict", json=CANON)
    assert r.status_code == 200
    j = r.json()
    assert j["prediction"] == "premium"
    assert j["confidence"] == pytest.approx(0.778, abs=0.01)
    assert set(j["probabilities"]) == {"budget", "fair", "premium"}
    assert sum(j["probabilities"].values()) == pytest.approx(1.0, abs=0.02)


def test_predict_category_only(client):
    r = client.post(
        "/predict",
        json={"item_category": "BERAS", "premise_type": "Pasar Mini", "state": "W.P. Kuala Lumpur"},
    )
    assert r.status_code == 200


@pytest.mark.parametrize(
    "payload,needle",
    [
        ({"item": "AYAM BERSIH - STANDARD", "premise_type": "Pasar Basah", "state": "Atlantis"}, "state"),
        ({"item": "AYAM BERSIH - STANDARD", "premise_type": "7-Eleven", "state": "Sabah"}, "premise_type"),
    ],
)
def test_predict_unsupported_value_returns_422(client, payload, needle):
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    assert needle in r.text


def test_predict_missing_item_and_category_422(client):
    r = client.post("/predict", json={"premise_type": "Pasar Basah", "state": "Sabah"})
    assert r.status_code == 422


def test_predict_missing_required_field_422(client):
    # premise_type omitted -> pydantic "field required"
    r = client.post("/predict", json={"item": "AYAM BERSIH - STANDARD", "state": "Sabah"})
    assert r.status_code == 422


@pytest.mark.parametrize("price,verdict", [(12.5, "premium"), (8.6, "fair"), (6.0, "budget")])
def test_price_check_verdicts(client, price, verdict):
    r = client.post("/price-check", json={"item": "AYAM BERSIH - STANDARD", "price": price})
    assert r.status_code == 200
    assert r.json()["verdict"] == verdict


def test_price_check_unknown_item_422(client):
    r = client.post("/price-check", json={"item": "UNICORN MEAT", "price": 5})
    assert r.status_code == 422


def test_price_check_nonpositive_price_422(client):
    r = client.post("/price-check", json={"item": "AYAM BERSIH - STANDARD", "price": 0})
    assert r.status_code == 422
