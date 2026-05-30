"""Model-derived catalog, metadata, and the deterministic price-check.

Everything the API needs to describe itself — valid premise types, states, item
categories, the item list, the model name + metrics — is read straight from the
trained artifact, so it can never drift from the model. The 16 MB artifact is
loaded only ONCE: we reuse the lru_cache inside predict.py.
"""
from __future__ import annotations

from functools import lru_cache

from app.features import assign_tier
from app.predict import DEFAULT_MODEL_PATH, _load_artifact


@lru_cache(maxsize=1)
def get_artifact() -> dict:
    """Return the trained artifact, loaded once (shares predict.py's cache)."""
    return _load_artifact(str(DEFAULT_MODEL_PATH))


@lru_cache(maxsize=1)
def get_metadata() -> dict:
    """Valid values + model info for API clients and the Streamlit dropdowns."""
    art = get_artifact()
    ref = art["reference"]
    catalog = art["item_catalog"]

    items = sorted(catalog["item"].dropna().astype(str).unique().tolist())
    categories = sorted(catalog["item_category"].dropna().astype(str).unique().tolist())
    items_by_category = {
        str(cat): sorted(grp["item"].astype(str).unique().tolist())
        for cat, grp in catalog.dropna(subset=["item", "item_category"]).groupby("item_category")
    }
    premise_types = sorted(ref["premise_type_price_level"].keys())
    states = sorted(ref["state_price_level"].keys())
    metrics = {k: round(float(v), 4) for k, v in (art.get("metrics") or {}).items()}

    return {
        "model_name": art.get("model_name"),
        "classes": list(art["classes"]),
        "weighted_f1": metrics.get("test_f1"),
        "metrics": metrics,
        "premise_types": premise_types,
        "states": states,
        "item_categories": categories,
        "items": items,
        "items_by_category": items_by_category,
        "counts": {
            "items": len(items),
            "item_categories": len(categories),
            "premise_types": len(premise_types),
            "states": len(states),
        },
        "tier_rule": {
            "budget": "price < 0.90 × the item's national median",
            "fair": "0.90 × median ≤ price ≤ 1.10 × median",
            "premium": "price > 1.10 × the item's national median",
        },
    }


@lru_cache(maxsize=1)
def valid_premise_types() -> frozenset:
    return frozenset(get_metadata()["premise_types"])


@lru_cache(maxsize=1)
def valid_states() -> frozenset:
    return frozenset(get_metadata()["states"])


def resolve_item(name: str):
    """Return the catalog row for an item name (case-insensitive), or None."""
    catalog = get_artifact()["item_catalog"]
    match = catalog[catalog["item"].str.casefold() == str(name).casefold()]
    return match.iloc[0] if len(match) else None


def price_check(item: str, price: float) -> dict:
    """Deterministic tier verdict: a price vs the item's national median.

    This applies the same label *rule* that defined the training target — it is
    NOT the ML model. Requires a known item, because the national median is
    defined per item.
    """
    row = resolve_item(item)
    if row is None:
        raise ValueError(
            f"Unknown item '{item}'. The price check needs a known item so it can "
            f"look up that item's national median — see /metadata for the item list."
        )
    median = float(get_artifact()["reference"]["item_median_price"][int(row["item_code"])])
    ratio = price / median
    return {
        "item": str(row["item"]),
        "item_category": str(row["item_category"]),
        "price": round(float(price), 4),
        "national_median": round(median, 4),
        "ratio": round(ratio, 4),
        "verdict": assign_tier(ratio),
        "note": "Deterministic label rule (price vs the item's national median) — not the ML prediction.",
    }
