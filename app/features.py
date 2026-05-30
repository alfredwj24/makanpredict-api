"""Target labelling and feature engineering for the MakanPredict food-price-tier model.

A store-item's price tier is defined RELATIVE to that item's national median price, so the
model never sees the raw price (which would trivially determine the label). The features
describe only the *context* — what the item is, what kind of store sells it, and where — so
the task is a genuine prediction problem ("is this likely to be over/under/fairly priced?")
rather than arithmetic.

To avoid target leakage, the group-level price-level encodings are *fitted on the training
split only* (`fit_feature_reference`) and then mapped onto held-out / inference rows
(`build_features`).
"""
from __future__ import annotations

import pandas as pd

# --- Target definition: price as a ratio of the item's national median price ---
LOW_RATIO = 0.90    # below 0.90x of the item's national median -> budget (cheaper than usual)
HIGH_RATIO = 1.10   # above 1.10x -> premium (pricier than usual); in between -> fair
TARGET = "price_tier"
TIERS = ("budget", "fair", "premium")

# --- Feature schema (raw price and the ratio are deliberately excluded as they define the label) ---
CATEGORICAL_FEATURES = ["item_category", "item_group", "unit", "premise_type", "state"]
NUMERIC_FEATURES = ["item_median_price", "item_price_dispersion",
                    "premise_type_price_level", "state_price_level", "district_price_level"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Columns build_features needs on the input frame.
REQUIRED_INPUT = ["item_code", "item_category", "item_group", "unit", "premise_type", "state"]

# Bayesian shrinkage strength (pseudo-counts) for the group-level price-level encodings.
_SMOOTHING = 50.0


def assign_tier(ratio: float) -> str:
    """Map a price/national-median ratio to a price tier string."""
    if ratio < LOW_RATIO:
        return "budget"
    if ratio > HIGH_RATIO:
        return "premium"
    return "fair"


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add item_national_median, price_vs_item_median and the price_tier label.

    The national median is computed per item across the whole dataset — this *defines* the
    label (it is the reference "normal" price, analogous to a fixed threshold), not a model
    feature.
    """
    out = df.copy()
    out["item_national_median"] = out.groupby("item_code")["price"].transform("median")
    out["price_vs_item_median"] = out["price"] / out["item_national_median"]
    out[TARGET] = out["price_vs_item_median"].map(assign_tier)
    return out


def fit_feature_reference(df: pd.DataFrame) -> dict:
    """Learn the lookup tables used by build_features from a (training) DataFrame.

    Fitting these on the training split only — then mapping onto test/inference rows —
    keeps the group-level encodings free of target leakage.
    """
    global_ratio = float(df["price_vs_item_median"].mean())

    def smoothed_level(col: str) -> dict:
        grouped = df.groupby(col)["price_vs_item_median"].agg(["mean", "count"])
        # Shrink each group's mean ratio toward the global mean (stabilises small groups).
        level = (grouped["mean"] * grouped["count"] + global_ratio * _SMOOTHING) / (
            grouped["count"] + _SMOOTHING
        )
        return level.to_dict()

    by_item = df.groupby("item_code")
    item_dispersion = by_item["price_vs_item_median"].std()
    return {
        "global_ratio": global_ratio,
        "item_median_price": by_item["item_national_median"].first().to_dict(),
        "item_median_global": float(df["item_national_median"].median()),
        # How spread-out the item's prices are nationally (heteroskedasticity signal).
        "item_price_dispersion": item_dispersion.to_dict(),
        "item_dispersion_global": float(item_dispersion.median()),
        "item_category": by_item["item_category"].first().to_dict(),
        "item_group": by_item["item_group"].first().to_dict(),
        "unit": by_item["unit"].first().to_dict(),
        "premise_type_price_level": smoothed_level("premise_type"),
        "state_price_level": smoothed_level("state"),
        "district_price_level": smoothed_level("district"),
    }


def build_features(df: pd.DataFrame, reference: dict) -> pd.DataFrame:
    """Build the model feature matrix (FEATURE_COLUMNS) from clean rows using a fitted reference.

    `df` must contain the columns in REQUIRED_INPUT. Unknown items/premise-types/states fall
    back to global averages so inference never fails on unseen values.
    """
    out = pd.DataFrame(index=df.index)
    for col in CATEGORICAL_FEATURES:
        out[col] = df[col].astype("string")

    global_ratio = reference["global_ratio"]
    out["item_median_price"] = (
        df["item_code"].map(reference["item_median_price"]).fillna(reference["item_median_global"])
    )
    out["item_price_dispersion"] = (
        df["item_code"].map(reference["item_price_dispersion"]).fillna(reference["item_dispersion_global"])
    )
    out["premise_type_price_level"] = (
        df["premise_type"].map(reference["premise_type_price_level"]).fillna(global_ratio)
    )
    out["state_price_level"] = df["state"].map(reference["state_price_level"]).fillna(global_ratio)
    # district is optional at inference time; fall back to the global average when absent.
    if "district" in df.columns:
        out["district_price_level"] = (
            df["district"].map(reference["district_price_level"]).fillna(global_ratio)
        )
    else:
        out["district_price_level"] = global_ratio
    return out[FEATURE_COLUMNS]
