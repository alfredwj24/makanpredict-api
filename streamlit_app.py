"""MakanPredict — Streamlit frontend for the grocery price-tier model.

The backend is auto-detected:
  * if the FastAPI service is reachable (default http://localhost:8000, or whatever
    $MAKANPREDICT_API points to), the UI calls it over HTTP;
  * otherwise it loads the model **in-process** — so this runs standalone with no
    separate API server (e.g. on Streamlit Community Cloud).

Run:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import os

import httpx
import streamlit as st

API_BASE = os.environ.get("MAKANPREDICT_API", "http://localhost:8000").rstrip("/")

TIER_COLORS = {"budget": "#16a34a", "fair": "#d97706", "premium": "#e11d48"}
TIER_BLURB = {
    "budget": "cheaper than usual for this item",
    "fair": "about the normal price for this item",
    "premium": "pricier than usual for this item",
}
ANY = "— Any item in this category —"

st.set_page_config(page_title="MakanPredict", page_icon="🛒", layout="centered")
st.markdown(
    "<style>.block-container{padding-top:2.5rem;max-width:760px}</style>",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------- backend layer
@st.cache_resource(show_spinner=False)
def get_backend() -> tuple:
    """Decide once per process: use the HTTP API if reachable, else in-process model."""
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=2)
        if r.status_code == 200 and r.json().get("model_loaded"):
            return ("api", API_BASE)
    except Exception:
        pass
    return ("direct", None)


def error_text(resp: httpx.Response) -> str:
    try:
        detail = resp.json().get("detail")
        if isinstance(detail, list):  # pydantic validation errors
            return "; ".join(d.get("msg", "") for d in detail)
        return str(detail)
    except Exception:
        return f"HTTP {resp.status_code}"


@st.cache_data(ttl=300, show_spinner="Loading model…")
def load_metadata() -> dict:
    mode, base = get_backend()
    if mode == "api":
        r = httpx.get(f"{base}/metadata", timeout=15)
        r.raise_for_status()
        return r.json()
    from app.catalog import get_metadata
    return get_metadata()


def call_predict(payload: dict) -> tuple:
    """Return (ok, result_dict_or_error_message)."""
    mode, base = get_backend()
    if mode == "api":
        r = httpx.post(f"{base}/predict", json=payload, timeout=15)
        return (r.status_code == 200, r.json() if r.status_code == 200 else error_text(r))
    from app.predict import predict_price_tier
    try:
        return (True, predict_price_tier(payload))
    except ValueError as exc:
        return (False, str(exc))


def call_price_check(item: str, price: float) -> tuple:
    """Return (ok, result_dict_or_error_message)."""
    mode, base = get_backend()
    if mode == "api":
        r = httpx.post(f"{base}/price-check", json={"item": item, "price": price}, timeout=15)
        return (r.status_code == 200, r.json() if r.status_code == 200 else error_text(r))
    from app.catalog import price_check
    try:
        return (True, price_check(item, price))
    except ValueError as exc:
        return (False, str(exc))


# ------------------------------------------------------------------------ render
def render_prediction(res: dict, ctx: str) -> None:
    tier = res["prediction"]
    color = TIER_COLORS[tier]
    st.markdown(
        f"""
        <div style="border:1px solid {color}40;background:{color}14;border-radius:16px;
                    padding:22px;text-align:center;">
          <div style="font-size:13px;letter-spacing:.12em;color:#6b7280;
                      text-transform:uppercase;">Predicted price tier</div>
          <div style="font-size:46px;font-weight:800;color:{color};line-height:1.1;
                      margin:6px 0;">{tier.upper()}</div>
          <div style="font-size:18px;color:#374151;">{res['confidence'] * 100:.1f}% confidence</div>
          <div style="font-size:13px;color:#9ca3af;margin-top:4px;">{TIER_BLURB[tier]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"for {ctx}")
    st.write("")
    for t in ("budget", "fair", "premium"):
        pct = res["probabilities"][t] * 100
        c = TIER_COLORS[t]
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:12px;margin:7px 0;">
              <div style="width:74px;font-weight:600;color:{c};">{t.title()}</div>
              <div style="flex:1;background:#eef0f2;border-radius:8px;height:18px;overflow:hidden;">
                <div style="width:{pct:.1f}%;background:{c};height:100%;"></div>
              </div>
              <div style="width:52px;text-align:right;font-variant-numeric:tabular-nums;
                          color:#374151;">{pct:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_price_check(pc: dict) -> None:
    color = TIER_COLORS[pc["verdict"]]
    st.markdown(
        f"""
        <div style="border-left:5px solid {color};background:{color}10;border-radius:8px;
                    padding:14px 18px;">
          <span style="font-size:20px;font-weight:700;color:{color};">{pc['verdict'].upper()}</span>
          <span style="color:#4b5563;">&nbsp;— RM&nbsp;{pc['price']:.2f} vs national median
            RM&nbsp;{pc['national_median']:.2f} &nbsp;·&nbsp; ratio {pc['ratio']:.2f}×</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("⚖️ Deterministic label rule (price vs the item's national median) — not the ML prediction.")


# -------------------------------------------------------------------------- main
st.title("🛒 MakanPredict")
st.caption(
    "Is a grocery item **budget**, **fair**, or **premium** for its store type and state? "
    "An XGBoost model trained on Malaysia's DOSM PriceCatcher data."
)

try:
    md = load_metadata()
except Exception as exc:
    st.error(
        "Couldn't load the model.\n\n"
        "- **API mode:** start it with `uvicorn app.main:app --port 8000`.\n"
        "- **Direct mode:** ensure `models/price_classifier.pkl` is present.\n\n"
        f"Details: `{exc}`"
    )
    st.stop()

mode, base = get_backend()

with st.sidebar:
    st.subheader("Model")
    st.metric("Weighted F1", md["weighted_f1"])
    st.write(f"**Algorithm:** `{md['model_name']}`")
    st.write("**Tiers:** " + " · ".join(md["classes"]))
    st.divider()
    c = md["counts"]
    st.caption(
        f"{c['items']} items · {c['item_categories']} categories · "
        f"{c['premise_types']} store types · {c['states']} states"
    )
    st.caption("Serving: " + (f"API @ {base}" if mode == "api" else "direct (in-process model)"))

st.subheader("1 · Describe the item & place")
col1, col2 = st.columns(2)
premise_type = col1.selectbox("Store type", md["premise_types"], key="premise")
state = col2.selectbox("State", md["states"], key="state")

col3, col4 = st.columns(2)
category = col3.selectbox("Category", md["item_categories"], key="category")
# Item options depend on the chosen category. The category-specific key resets the
# item to "Any" when the category changes (avoids a stale-selection error).
item_options = [ANY] + md["items_by_category"].get(category, [])
item_choice = col4.selectbox("Item", item_options, key=f"item::{category}")
use_item = item_choice != ANY

if st.button("Predict price tier", type="primary", use_container_width=True, key="predict_btn"):
    payload = {"premise_type": premise_type, "state": state}
    payload["item" if use_item else "item_category"] = item_choice if use_item else category
    ok, data = call_predict(payload)
    if ok:
        ctx = (item_choice if use_item else f"any {category}") + f" · {premise_type} · {state}"
        st.session_state["prediction"] = data
        st.session_state["prediction_ctx"] = ctx
    else:
        st.session_state.pop("prediction", None)
        st.error(data)

if st.session_state.get("prediction"):
    render_prediction(st.session_state["prediction"], st.session_state.get("prediction_ctx", ""))

st.divider()
st.subheader("2 · Optional — is a price you saw fair?")
st.caption(
    "Compares a price directly to the item's national median: the exact label rule, "
    "computed deterministically (not the ML model). Needs a specific item."
)
if not use_item:
    st.info("Pick a specific item above (not “Any in this category”) to price-check.")
else:
    pc1, pc2 = st.columns([3, 1])
    price = pc1.number_input(
        f"Price you saw for “{item_choice}” (RM)",
        min_value=0.0, value=0.0, step=0.10, format="%.2f", key="price",
    )
    pc2.write("")
    pc2.write("")
    if pc2.button("Check price", use_container_width=True, key="check_btn"):
        if price <= 0:
            st.warning("Enter a price greater than 0.")
        else:
            ok, data = call_price_check(item_choice, price)
            if ok:
                render_price_check(data)
            else:
                st.error(data)

st.divider()
st.caption("MakanPredict · Project 2 of 3 · FastAPI + Streamlit over a Project 1 XGBoost model.")
