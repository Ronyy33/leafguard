"""Pydantic request/response models: the API contract documented in docs/LLD.md."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ClassScore(BaseModel):
    label: str = Field(..., examples=["Tomato___Late_blight"])
    display_name: str = Field(..., examples=["Tomato – Late blight"])
    probability: float = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    prediction_id: str = Field(..., description="Use this id when sending feedback")
    label: str
    display_name: str
    is_healthy: bool
    confidence: float = Field(..., ge=0, le=1)
    low_confidence: bool = Field(..., description="True if confidence is below the configured threshold")
    top3: list[ClassScore]
    heatmap_png_base64: Optional[str] = Field(None, description="Grad-CAM overlay (PNG, base64)")
    model_version: str
    inference_backend: Literal["mlflow_serve", "local"]
    latency_ms: float
    input_quality: dict[str, float] = Field(..., description="brightness / blur stats of the upload")
    warnings: list[str] = []


class FeedbackRequest(BaseModel):
    prediction_id: str
    true_label: str = Field(..., description="Correct class name (one of GET /classes)")


class FeedbackResponse(BaseModel):
    accepted: bool
    correct: bool
    rolling_accuracy: Optional[float]
    feedback_count: int


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadyResponse(BaseModel):
    ready: bool
    model_loaded: bool
    model_server_reachable: bool
    detail: str = ""


class ModelInfo(BaseModel):
    name: str
    alias: str
    version: str
    run_id: str
    backbone: str
    num_classes: int
    image_size: int
    test_macro_f1: Optional[float]
    classes: list[str]


class DriftStatus(BaseModel):
    window_size: int
    samples_in_window: int
    zscores: dict[str, float]
    drifting_features: list[str]
    threshold: float


class RetrainResponse(BaseModel):
    accepted: bool
    detail: str
    state: dict


class ModelActionResponse(BaseModel):
    ok: bool
    detail: str
    version: str


class ErrorResponse(BaseModel):
    error: str
    detail: str
