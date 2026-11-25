"""Export trained SB3 policy to ONNX, verify numerical parity, and benchmark CPU latency."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from stable_baselines3 import PPO, SAC

from price_gym.utils.metrics import compute_latency_percentiles


class DeterministicPPOWrapper(nn.Module):
    """Wraps an SB3 PPO policy's deterministic actor (mlp_extractor.policy_net + action_net)."""

    def __init__(self, sb3_policy: nn.Module) -> None:
        super().__init__()
        self.features_extractor = sb3_policy.features_extractor
        self.policy_net = sb3_policy.mlp_extractor.policy_net
        self.action_net = sb3_policy.action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.features_extractor(obs)
        latent_pi = self.policy_net(features)
        mean_actions = self.action_net(latent_pi)
        return torch.clamp(mean_actions, -1.0, 1.0)


class DeterministicSACWrapper(nn.Module):
    """Wraps an SB3 SAC policy's deterministic actor (latent_pi + mu + tanh squashing)."""

    def __init__(self, sb3_policy: nn.Module) -> None:
        super().__init__()
        self.actor = sb3_policy.actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.actor.extract_features(obs, self.actor.features_extractor)
        latent_pi = self.actor.latent_pi(features)
        mean_actions = self.actor.mu(latent_pi)
        return torch.tanh(mean_actions)


def export_and_benchmark(
    model_path: str = "artifacts/models/ppo_best/best_model.zip",
    algo: str = "ppo",
    onnx_output_path: str = "artifacts/models/policy.onnx",
    n_benchmark_runs: int = 1000,
) -> dict[str, float]:
    """Export trained SB3 model to ONNX, assert parity, and benchmark latency."""
    algo_lower = algo.lower()
    ckpt_path = Path(model_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Trained SB3 checkpoint not found at: {ckpt_path}")

    if algo_lower == "ppo":
        sb3_model = PPO.load(str(ckpt_path), device="cpu")
        wrapper: nn.Module = DeterministicPPOWrapper(sb3_model.policy).eval()
    elif algo_lower == "sac":
        sb3_model = SAC.load(str(ckpt_path), device="cpu")
        wrapper = DeterministicSACWrapper(sb3_model.policy).eval()
    else:
        raise ValueError(f"Unsupported algorithm '{algo}'. Use 'ppo' or 'sac'.")

    out_path = Path(onnx_output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.zeros((1, 7), dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(out_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch_size"}, "action": {0: "batch_size"}},
        dynamo=False,
    )

    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = ort.InferenceSession(
        str(out_path), sess_options=sess_options, providers=["CPUExecutionProvider"]
    )

    # Parity test across 64 random valid observations
    rng = np.random.default_rng(42)
    low = np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0], dtype=np.float32)
    high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    test_obs = rng.uniform(low, high, size=(64, 7)).astype(np.float32)

    with torch.no_grad():
        torch_out = wrapper(torch.from_numpy(test_obs)).numpy()

    onnx_out = ort_session.run(["action"], {"obs": test_obs})[0]
    max_abs_diff = float(np.max(np.abs(torch_out - onnx_out)))

    if not np.allclose(torch_out, onnx_out, atol=1e-5):
        raise AssertionError(
            f"ONNX vs PyTorch parity check failed! Max abs diff: {max_abs_diff:.3e}"
        )

    print(
        f"[PriceGym] ONNX export succeeded: {out_path} "
        f"| Numerical parity verified (max |diff| = {max_abs_diff:.2e} <= 1e-5)"
    )

    # Warm-up
    single_obs = test_obs[0:1]
    for _ in range(50):
        ort_session.run(["action"], {"obs": single_obs})

    latencies_ms: list[float] = []
    for i in range(n_benchmark_runs):
        sample = test_obs[i % len(test_obs) : (i % len(test_obs)) + 1]
        t0 = time.perf_counter()
        ort_session.run(["action"], {"obs": sample})
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    stats = compute_latency_percentiles(latencies_ms)
    print(
        f"[PriceGym] ONNX Runtime CPU Latency ({n_benchmark_runs} calls, batch=1): "
        f"Mean={stats['mean_ms']:.4f} ms | "
        f"p50={stats['p50_ms']:.4f} ms | "
        f"p95={stats['p95_ms']:.4f} ms | "
        f"p99={stats['p99_ms']:.4f} ms"
    )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export trained SB3 policy to ONNX and benchmark CPU inference latency."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="artifacts/models/ppo_best/best_model.zip",
        help="Path to trained SB3 model zip file.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["ppo", "sac"],
        default="ppo",
        help="Algorithm type of the checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/models/policy.onnx",
        help="Output path for the exported ONNX model.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1000,
        help="Number of single-sample inference calls for latency benchmarking.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_and_benchmark(
        model_path=args.model_path,
        algo=args.algo,
        onnx_output_path=args.output,
        n_benchmark_runs=args.runs,
    )
