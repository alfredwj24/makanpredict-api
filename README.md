# 🛒 MakanPredict API

A **FastAPI** service + **Streamlit** frontend that serves the grocery price-tier model from
[MakanPredict](https://github.com/alfredwj24/makanpredict). Given a grocery **item** (or category), a **store type**, and a
**state**, it predicts whether the price is likely to be **budget**, **fair**, or **premium**
relative to that item's national median — and returns the probability of each tier.

> Project 2 of a 3-part AI portfolio. Model: XGBoost on Malaysia's DOSM PriceCatcher data
> (191,904 records · 273 items · 2,130 shops · 16 states), weighted F1 **0.739**, with SHAP
> explainability in Project 1.

---

## Why there's no `price` input

A store-item's tier is defined *relative to that item's national median price*, so the model
**never sees the raw price** — feeding it in would trivially determine the label. The model
answers *"for this item, store type and state, is the price likely to be over/under/fairly
priced?"* from context alone.

For the original *"is THIS price fair?"* question, the separate **`POST /price-check`** endpoint
applies the label rule directly (price vs the item's national median). It's deterministic — the
exact threshold rule, **not** the ML model — and is labelled as such.

---

## Screenshots

The Streamlit UI — dropdowns driven live from the API, a colour-coded result card, and the
three-tier probability bars:

![MakanPredict UI](docs/screenshot.png)

---

## Project structure

```
.
├── app/                      # FastAPI service + the Project 1 model code
│   ├── main.py               #   endpoints: / · /health · /metadata · /predict · /price-check
│   ├── schemas.py            #   Pydantic request/response models (strict 422 validation)
│   ├── catalog.py            #   artifact-driven metadata + the deterministic price-check
│   ├── predict.py            #   Project 1 prediction interface (predict_price_tier)
│   └── features.py           #   Project 1 feature engineering + the tier label rule
├── models/
│   └── price_classifier.pkl  # trained artifact (git-ignored — see models/README.md)
├── streamlit_app.py          # the frontend
├── tests/                    # pytest: model contract, endpoints/422s, perf + concurrency
├── requirements.txt
└── pytest.ini
```

---

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Make sure the model is in place (see models/README.md)
#    models/price_classifier.pkl

# 3. Run the API  ->  http://localhost:8000/docs
uvicorn app.main:app --reload --port 8000

# 4. In another terminal, run the UI  ->  http://localhost:8501
streamlit run streamlit_app.py
```

The Streamlit app reads the API base URL from `$MAKANPREDICT_API` (default
`http://localhost:8000`).

---

## API

| Method & path | Purpose |
|---|---|
| `POST /predict` | ML price-tier prediction (tier + confidence + all three probabilities) |
| `POST /price-check` | Deterministic verdict: a price vs the item's national median (label rule) |
| `GET /metadata` | Valid premise types / states / categories / items + model info |
| `GET /health` | Liveness + model-loaded check |
| `GET /` | Basic info |
| `GET /docs` | Interactive Swagger UI (free from FastAPI) |

### `POST /predict`

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' -d '{
  "item": "AYAM BERSIH - STANDARD",
  "premise_type": "Pasar Basah",
  "state": "Sabah"
}'
```
```json
{
  "prediction": "premium",
  "confidence": 0.778,
  "probabilities": { "budget": 0.034, "fair": 0.189, "premium": 0.778 }
}
```

**Request fields**

- `premise_type` *(required)* — one of 5 store types. Unsupported → **422**.
- `state` *(required)* — one of 16 states. Unsupported → **422**.
- `item` **or** `item_category` *(one required)* — free text; unknown values fall back
  gracefully (not rejected). Provide neither → **422**.

> Pass `item_category` (e.g. `"BERAS"`) instead of `item` to predict for a whole category.

### `POST /price-check`

```bash
curl -s localhost:8000/price-check -H 'content-type: application/json' -d '{
  "item": "AYAM BERSIH - STANDARD", "price": 12.50
}'
```
```json
{
  "item": "AYAM BERSIH - STANDARD", "item_category": "AYAM",
  "price": 12.5, "national_median": 8.6, "ratio": 1.4535, "verdict": "premium",
  "note": "Deterministic label rule (price vs the item's national median) — not the ML prediction."
}
```

Tier rule: `budget` if price < 0.90× median · `premium` if price > 1.10× median · else `fair`.

---

## Validation

`premise_type` and `state` are checked against the values the model actually knows (sourced
from the artifact) → a clear **422** listing the valid options. Missing required fields → **422**.
`item` / `item_category` are free text and fall back gracefully, so they are never rejected.

---

## Performance

Measured locally (Apple Silicon), model loaded once at startup via `lru_cache`:

- **Single warm request:** ~7 ms
- **200 requests at 10 concurrent:** 200/200 OK · p50 **54 ms** · p95 **71 ms** · ~181 req/s

Both well within the **< 200 ms** / **~10 concurrent** target.

---

## Testing

```bash
pytest
```
23 tests: the model contract (`predict_price_tier`), every endpoint, all the 422 cases, the
price-check verdicts, warm latency `< 200 ms`, and 10-way concurrency.

---

## Tech stack

FastAPI · Pydantic v2 · Uvicorn · Streamlit · pytest · httpx — over a scikit-learn pipeline +
XGBoost model (joblib). Python 3.10+.
