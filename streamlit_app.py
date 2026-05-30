"""MakanPredict — Streamlit frontend for the price-tier API.

Talks to the FastAPI service over HTTP (base URL from $MAKANPREDICT_API, default
http://localhost:8000). All dropdown values come from the API's /metadata, so the
UI stays in sync with the model automatically.

Run:  streamlit run streamlit_app.py     (with the API already running)
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


# --------------------------------------------------------------------------- API
@st.cache_data(ttl=300, show_spinner=False)
def fetch_metadata() -> dict:
    r = httpx.get(f"{API_BASE}/metadata", timeout=15)
    r.raise_for_status()
    return r.json()


def api_health() -> dict | None:
    try:
        return httpx.get(f"{API_BASE}/health", timeout=5).json()
    except Exception:
        return None


def error_text(resp: httpx.Response) -> str:
    try:
        detail = resp.json().get("detail")
        if isinstance(detail, list):  # pydantic validation errors
            return "; ".join(d.get("msg", "") for d in detail)
        return str(detail)
    except Exception:
        return f"HTTP {resp.status_code}"


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
    md = fetch_metadata()
except Exception:
    st.error(
        f"Can't reach the MakanPredict API at `{API_BASE}`.\n\n"
        "Start it first:\n\n```\nuvicorn app.main:app --port 8000\n```"
    )
    st.stop()

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
    health = api_health()
    if health and health.get("model_loaded"):
        st.success("API connected ✓")
    else:
        st.warning("API unreachable")
    st.caption(f"API: `{API_BASE}`")

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
    try:
        resp = httpx.post(f"{API_BASE}/predict", json=payload, timeout=15)
        if resp.status_code == 200:
            ctx = (item_choice if use_item else f"any {category}") + f" · {premise_type} · {state}"
            st.session_state["prediction"] = resp.json()
            st.session_state["prediction_ctx"] = ctx
        else:
            st.session_state.pop("prediction", None)
            st.error(error_text(resp))
    except Exception as exc:
        st.error(f"Request failed: {exc}")

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
            resp = httpx.post(f"{API_BASE}/price-check", json={"item": item_choice, "price": price}, timeout=15)
            if resp.status_code == 200:
                render_price_check(resp.json())
            else:
                st.error(error_text(resp))

st.divider()
st.caption("MakanPredict · Project 2 of 3 · FastAPI + Streamlit over a Project 1 XGBoost model.")
