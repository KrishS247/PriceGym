"""FastAPI + ONNX Runtime Real-Time E-Commerce Dynamic Pricing Service."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from price_gym.envs.pricing_env import load_yaml_config
from price_gym.utils.demand import compute_cyclical_time_features


class PricingRequest(BaseModel):
    """Input payload representing current SKU inventory, market, and time state."""

    current_inventory: int = Field(..., ge=0, description="Remaining units in stock.")
    initial_inventory: int = Field(
        500, gt=0, description="Initial seasonal stock allocation."
    )
    current_day: int = Field(
        ..., ge=0, description="Current day index in the selling season."
    )
    episode_days: int = Field(
        90, gt=0, description="Total days in the selling season."
    )
    prev_price: float = Field(..., gt=0.0, description="Previous day's SKU price ($).")
    comp_price: float = Field(
        ..., gt=0.0, description="Competitor's current SKU price ($)."
    )
    rolling_7d_demand: float = Field(
        ..., ge=0.0, description="7-day rolling average daily units sold."
    )


class PricingResponse(BaseModel):
    """Real-time pricing recommendation and telemetry response."""

    recommended_price: float
    expected_margin_per_unit: float
    inference_latency_ms: float


class HealthResponse(BaseModel):
    """Service and ONNX Runtime model health status."""

    status: str
    model_loaded: bool
    model_path: str


class ONNXPricingService:
    """Manages ONNX Runtime inference session and observation normalization."""

    def __init__(
        self, model_path: str | Path, config_path: str | Path | None = None
    ) -> None:
        self.model_path = Path(model_path)
        cfg = load_yaml_config(config_path)
        env_cfg = cfg.get("env", {})
        dem_cfg = cfg.get("demand", {})

        self.min_price = float(env_cfg.get("min_price", 20.0))
        self.max_price = float(env_cfg.get("max_price", 100.0))
        self.unit_cost = float(env_cfg.get("unit_cost", 25.0))
        self.base_demand = float(dem_cfg.get("base_demand", 6.5))
        self.session: ort.InferenceSession | None = None

    def load(self) -> None:
        """Load the ONNX policy into an ONNX Runtime CPU InferenceSession."""
        if self.model_path.exists():
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
        else:
            self.session = None

    def build_observation(self, req: PricingRequest) -> np.ndarray:
        """Construct the 7-D float32 normalized observation vector matching DynamicPricingEnv."""
        price_span = max(self.max_price - self.min_price, 1e-6)

        inv_ratio = np.clip(
            req.current_inventory / max(req.initial_inventory, 1), 0.0, 1.0
        )
        time_rem_ratio = np.clip(
            (req.episode_days - req.current_day) / max(req.episode_days, 1), 0.0, 1.0
        )
        norm_prev = np.clip((req.prev_price - self.min_price) / price_span, 0.0, 1.0)
        norm_comp = np.clip((req.comp_price - self.min_price) / price_span, 0.0, 1.0)
        sin_t, cos_t = compute_cyclical_time_features(
            req.current_day, req.episode_days
        )
        norm_demand = np.clip(
            req.rolling_7d_demand / max(self.base_demand * 2.0, 1e-6), 0.0, 1.0
        )

        return np.array(
            [
                [
                    inv_ratio,
                    time_rem_ratio,
                    norm_prev,
                    norm_comp,
                    np.clip(sin_t, -1.0, 1.0),
                    np.clip(cos_t, -1.0, 1.0),
                    norm_demand,
                ]
            ],
            dtype=np.float32,
        )

    def predict_price(self, req: PricingRequest) -> PricingResponse:
        """Run sub-millisecond ONNX inference and return recommended price & unit margin."""
        if self.session is None:
            self.load()
        if self.session is None:
            raise HTTPException(
                status_code=503,
                detail=f"ONNX policy not found at {self.model_path}. Run scripts/export_onnx.py first.",
            )

        t0 = time.perf_counter()
        obs = self.build_observation(req)
        action_out = self.session.run(["action"], {"obs": obs})[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        raw_act = float(np.clip(action_out[0, 0], -1.0, 1.0))
        norm_act = 0.5 * (raw_act + 1.0)
        rec_price = float(
            np.clip(
                self.min_price + norm_act * (self.max_price - self.min_price),
                self.min_price,
                self.max_price,
            )
        )
        margin_per_unit = rec_price - self.unit_cost

        return PricingResponse(
            recommended_price=round(rec_price, 2),
            expected_margin_per_unit=round(margin_per_unit, 2),
            inference_latency_ms=round(latency_ms, 4),
        )


def _default_onnx_path() -> Path:
    env_override = os.environ.get("PRICEGYM_ONNX_PATH")
    if env_override:
        return Path(env_override)
    return Path(__file__).resolve().parents[3] / "artifacts" / "models" / "policy.onnx"


pricing_service = ONNXPricingService(model_path=_default_onnx_path())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ONNX Runtime session on FastAPI startup."""
    pricing_service.model_path = _default_onnx_path()
    pricing_service.load()
    yield


app = FastAPI(
    title="PriceGym Real-Time Dynamic Pricing API",
    description="Low-latency e-commerce dynamic pricing powered by ONNX Runtime.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> dict[str, Any]:
    """Return service and ONNX model health status."""
    if pricing_service.session is None:
        pricing_service.load()
    return {
        "status": "ok" if pricing_service.session is not None else "degraded",
        "model_loaded": pricing_service.session is not None,
        "model_path": str(pricing_service.model_path),
    }


@app.post("/predict-price", response_model=PricingResponse)
def predict_price(request: PricingRequest) -> PricingResponse:
    """Compute real-time price recommendation for a SKU state."""
    return pricing_service.predict_price(request)
