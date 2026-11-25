"""CLI Training Entrypoint for PriceGym RL Agents (PPO and SAC)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable

import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

from price_gym.envs.pricing_env import DynamicPricingEnv, load_yaml_config


def make_env_fn(config: dict[str, Any]) -> Callable[[], DynamicPricingEnv]:
    """Return a zero-arg callable that constructs DynamicPricingEnv."""

    def _init() -> DynamicPricingEnv:
        return DynamicPricingEnv(config=config)

    return _init


def train(
    config_path: str = "configs/default.yaml",
    algo: str = "ppo",
    timesteps: int = 200_000,
    n_envs: int = 4,
    seed: int = 42,
    output_dir: str = "artifacts",
) -> Path:
    """Train PPO or SAC on DynamicPricingEnv and save best checkpoint."""
    algo_lower = algo.lower()
    if algo_lower not in {"ppo", "sac"}:
        raise ValueError(f"Unsupported algorithm '{algo}'. Choose 'ppo' or 'sac'.")

    set_random_seed(seed)
    torch.manual_seed(seed)

    cfg = load_yaml_config(config_path)
    rl_cfg = cfg.get("rl", {}).get(algo_lower, {})

    out_root = Path(output_dir)
    tb_log_dir = out_root / "tb_logs"
    best_model_dir = out_root / "models" / f"{algo_lower}_best"
    eval_log_dir = out_root / "models" / f"{algo_lower}_eval_logs"

    tb_log_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_vec_env(
        make_env_fn(cfg),
        n_envs=n_envs,
        seed=seed,
    )
    eval_env = make_vec_env(
        make_env_fn(cfg),
        n_envs=1,
        seed=seed + 5000,
    )

    eval_freq = max(timesteps // (20 * max(n_envs, 1)), 500)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_log_dir),
        eval_freq=eval_freq,
        n_eval_episodes=25,
        deterministic=True,
        render=False,
        verbose=0,
    )

    net_arch = list(rl_cfg.get("net_arch", [128, 128]))
    policy_kwargs = {"net_arch": net_arch}

    if algo_lower == "ppo":
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=float(rl_cfg.get("learning_rate", 3e-4)),
            n_steps=int(rl_cfg.get("n_steps", 512)),
            batch_size=int(rl_cfg.get("batch_size", 128)),
            n_epochs=int(rl_cfg.get("n_epochs", 10)),
            gamma=float(rl_cfg.get("gamma", 0.995)),
            gae_lambda=float(rl_cfg.get("gae_lambda", 0.95)),
            clip_range=float(rl_cfg.get("clip_range", 0.2)),
            ent_coef=float(rl_cfg.get("ent_coef", 0.005)),
            vf_coef=float(rl_cfg.get("vf_coef", 0.5)),
            max_grad_norm=float(rl_cfg.get("max_grad_norm", 0.5)),
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(tb_log_dir),
            seed=seed,
            verbose=1,
            device="cpu",
        )
    else:
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=float(rl_cfg.get("learning_rate", 3e-4)),
            buffer_size=int(rl_cfg.get("buffer_size", 100_000)),
            learning_starts=int(rl_cfg.get("learning_starts", 1000)),
            batch_size=int(rl_cfg.get("batch_size", 256)),
            tau=float(rl_cfg.get("tau", 0.005)),
            gamma=float(rl_cfg.get("gamma", 0.995)),
            train_freq=int(rl_cfg.get("train_freq", 1)),
            gradient_steps=int(rl_cfg.get("gradient_steps", 1)),
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(tb_log_dir),
            seed=seed,
            verbose=1,
            device="cpu",
        )

    print(
        f"[PriceGym] Starting {algo_lower.upper()} training for {timesteps:,} timesteps "
        f"across {n_envs} vectorized environments (seed={seed})..."
    )
    start_time = time.perf_counter()
    model.learn(
        total_timesteps=timesteps,
        callback=eval_callback,
        tb_log_name=f"{algo_lower}_seed{seed}",
        progress_bar=False,
    )
    elapsed = max(time.perf_counter() - start_time, 1e-6)
    throughput = timesteps / elapsed

    best_zip = best_model_dir / "best_model.zip"
    if not best_zip.exists():
        model.save(str(best_zip))

    train_env.close()
    eval_env.close()

    print(
        f"[PriceGym] Completed {algo_lower.upper()} training in {elapsed:.2f}s "
        f"| Simulation Throughput: {throughput:,.1f} steps/sec"
    )
    print(f"[PriceGym] Best checkpoint saved to: {best_zip}")
    return best_zip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO or SAC dynamic pricing agent on PriceGym."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["ppo", "sac"],
        default="ppo",
        help="RL algorithm to train (ppo or sac).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Total training environment timesteps.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel vectorized environments.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible training.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        config_path=args.config,
        algo=args.algo,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
    )
