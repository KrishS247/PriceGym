"""Integration tests for FastAPI + ONNX Runtime real-time pricing endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from price_gym.api.app import app, pricing_service


class _TinyDummyPolicy(nn.Module):
    """Lightweight deterministic fallback policy if policy.onnx is not yet exported."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(7, 1)
        with torch.no_grad():
            self.fc.weight.fill_(0.05)
            self.fc.bias.fill_(0.1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc(obs))


@pytest.fixture(scope="module", autouse=True)
def ensure_onnx_model(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Ensure a valid ONNX model is loaded into pricing_service for testing."""
    default_onnx = (
        Path(__file__).resolve().parents[1] / "artifacts" / "models" / "policy.onnx"
    )
    if default_onnx.exists():
        pricing_service.model_path = default_onnx
        pricing_service.load()
        return

    tmp_onnx = tmp_path_factory.mktemp("onnx_test") / "policy.onnx"
    dummy = _TinyDummyPolicy().eval()
    torch.onnx.export(
        dummy,
        torch.zeros((1, 7), dtype=torch.float32),
        str(tmp_onnx),
        input_names=["obs"],
        output_names=["action"],
        opset_version=17,
        dynamo=False,
    )
    pricing_service.model_path = tmp_onnx
    pricing_service.load()


def test_health_endpoint() -> None:
    """Verify GET /health returns status ok and model_loaded True."""
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert "policy.onnx" in data["model_path"]


def test_predict_price_valid_payload() -> None:
    """Verify POST /predict-price returns valid price within bounds and low latency."""
    payload = {
        "current_inventory": 320,
        "initial_inventory": 500,
        "current_day": 35,
        "episode_days": 90,
        "prev_price": 58.50,
        "comp_price": 61.00,
        "rolling_7d_demand": 5.8,
    }
    with TestClient(app) as client:
        resp = client.post("/predict-price", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert 20.0 <= data["recommended_price"] <= 100.0
        assert data["expected_margin_per_unit"] == pytest.approx(
            data["recommended_price"] - 25.0, abs=1e-2
        )
        assert data["inference_latency_ms"] >= 0.0
        assert data["inference_latency_ms"] < 50.0


def test_predict_price_validation_error() -> None:
    """Verify POST /predict-price rejects negative inventory or invalid prices."""
    bad_payload = {
        "current_inventory": -10,
        "initial_inventory": 500,
        "current_day": 10,
        "episode_days": 90,
        "prev_price": 60.0,
        "comp_price": 60.0,
        "rolling_7d_demand": 5.0,
    }
    with TestClient(app) as client:
        resp = client.post("/predict-price", json=bad_payload)
        assert resp.status_code == 422
