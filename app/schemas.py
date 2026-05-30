"""Pydantic request/response models for the MakanPredict API.

`premise_type` and `state` are validated strictly against the values the model
knows (unsupported -> 422). `item` / `item_category` are free text: unknown
values fall back gracefully inside the model, so they are not rejected.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.catalog import valid_premise_types, valid_states

# Allow `model_name` fields without pydantic's "model_" protected-namespace warning.
_ALLOW_MODEL_PREFIX = ConfigDict(protected_namespaces=())


class PredictRequest(BaseModel):
    premise_type: str = Field(..., description="Store type", examples=["Pasar Basah"])
    state: str = Field(..., description="Malaysian state", examples=["Sabah"])
    item: Optional[str] = Field(
        None,
        description="Product name (free text; unknown items fall back gracefully).",
        examples=["AYAM BERSIH - STANDARD"],
    )
    item_category: Optional[str] = Field(
        None,
        description="Category — used when no specific item is given.",
        examples=["BERAS"],
    )

    @model_validator(mode="after")
    def _validate(self):
        if not self.item and not self.item_category:
            raise ValueError("Provide one of 'item' or 'item_category'.")
        if self.premise_type not in valid_premise_types():
            raise ValueError(
                f"Unsupported premise_type '{self.premise_type}'. "
                f"Valid options: {sorted(valid_premise_types())}."
            )
        if self.state not in valid_states():
            raise ValueError(
                f"Unsupported state '{self.state}'. Valid options: {sorted(valid_states())}."
            )
        return self


class Probabilities(BaseModel):
    budget: float
    fair: float
    premium: float


class PredictResponse(BaseModel):
    prediction: str = Field(..., examples=["premium"])
    confidence: float = Field(..., examples=[0.778])
    probabilities: Probabilities


class PriceCheckRequest(BaseModel):
    item: str = Field(..., description="A known product name.", examples=["AYAM BERSIH - STANDARD"])
    price: float = Field(..., gt=0, description="The price you observed, in RM.", examples=[12.5])


class PriceCheckResponse(BaseModel):
    item: str
    item_category: str
    price: float
    national_median: float
    ratio: float
    verdict: str = Field(..., examples=["premium"])
    note: str


class HealthResponse(BaseModel):
    model_config = _ALLOW_MODEL_PREFIX
    status: str
    model_loaded: bool
    model_name: Optional[str] = None


class MetadataResponse(BaseModel):
    model_config = _ALLOW_MODEL_PREFIX
    model_name: Optional[str]
    classes: list[str]
    weighted_f1: Optional[float]
    metrics: dict
    premise_types: list[str]
    states: list[str]
    item_categories: list[str]
    items: list[str]
    items_by_category: dict
    counts: dict
    tier_rule: dict
