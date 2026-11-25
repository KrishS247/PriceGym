"""Holdout Evaluation Suite (500 Episodes) and Matplotlib Chart Generator."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, SAC

from price_gym.agents.baselines import (
    RandomPriceAgent,
    RuleBasedMarkdownAgent,
    StaticPriceAgent,
)
from price_gym.envs.pricing_env import DynamicPricingEnv, load_yaml_config
from price_gym.utils.metrics import (
    AgentBenchmarkResult,
    compute_margin_lift,
    format_markdown_table,
    run_policy_episode,
)

matplotlib.use("Agg")


def evaluate_policy(
    agent_name: str,
    policy_fn: Callable[[np.ndarray, dict[str, Any]], np.ndarray],
    cfg: dict[str, Any],
    seeds: list[int],
) -> AgentBenchmarkResult:
    """Evaluate a pricing policy across all holdout seeds."""
    env = DynamicPricingEnv(config=cfg)
    traces = [run_policy_episode(env, policy_fn, seed=s) for s in seeds]

    margins = np.array([t.total_margin for t in traces], dtype=np.float64)
    sell_throughs = np.array([t.sell_through_rate for t in traces], dtype=np.float64)
    stockouts = np.array(
        [100.0 if t.stockout else 0.0 for t in traces], dtype=np.float64
    )
    vols = np.array([t.mean_price_volatility for t in traces], dtype=np.float64)
    cum_curves = np.stack([t.cumulative_margins for t in traces], axis=0)

    return AgentBenchmarkResult(
        agent_name=agent_name,
        mean_margin=float(np.mean(margins)),
        std_margin=float(np.std(margins)),
        sell_through_pct=float(np.mean(sell_throughs)),
        stockout_pct=float(np.mean(stockouts)),
        mean_volatility=float(np.mean(vols)),
        mean_cumulative_margin_curve=np.mean(cum_curves, axis=0),
        std_cumulative_margin_curve=np.std(cum_curves, axis=0),
        sample_trace=traces[0],
    )


def generate_plots(
    results: dict[str, AgentBenchmarkResult],
    plots_dir: Path,
) -> None:
    """Generate and save 3 high-DPI evaluation charts to artifacts/plots/."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )

    palette = {
        "PPO": "#1f77b4",
        "SAC": "#2ca02c",
        "Rule-Based Markdown": "#ff7f0e",
        "Static Price": "#7f7f7f",
        "Random Price": "#d62728",
    }

    # 1. cumulative_margin.png
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    for name, res in results.items():
        days = np.arange(1, len(res.mean_cumulative_margin_curve) + 1)
        color = palette.get(name, "#333333")
        mean_c = res.mean_cumulative_margin_curve
        std_c = res.std_cumulative_margin_curve
        ax.plot(
            days,
            mean_c,
            label=f"{name} (${res.mean_margin:,.0f})",
            color=color,
            linewidth=2.2,
        )
        ax.fill_between(days, mean_c - std_c, mean_c + std_c, color=color, alpha=0.14)

    ax.set_title(
        "90-Day Cumulative Gross Margin Across Holdout Episodes (±1 Std Dev)",
        fontweight="bold",
    )
    ax.set_xlabel("Day of Season")
    ax.set_ylabel("Cumulative Gross Margin ($)")
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(plots_dir / "cumulative_margin.png", dpi=200)
    plt.close(fig)

    # 2. pricing_trajectory.png
    rl_key = (
        "PPO"
        if "PPO" in results
        else next((k for k in results if k in {"PPO", "SAC"}), "Rule-Based Markdown")
    )
    rl_trace = results[rl_key].sample_trace
    rule_trace = results["Rule-Based Markdown"].sample_trace

    if rl_trace is not None and rule_trace is not None:
        days = np.arange(1, len(rl_trace.daily_prices) + 1)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), dpi=200, sharex=True)

        ax1.plot(
            days,
            rl_trace.daily_prices,
            label=f"{rl_key} Agent Price",
            color="#1f77b4",
            linewidth=2.2,
        )
        ax1.plot(
            days,
            rl_trace.daily_comp_prices,
            label="Competitor Price",
            color="#2ca02c",
            linestyle="--",
            linewidth=1.8,
        )
        ax1.plot(
            days,
            rule_trace.daily_prices,
            label="Rule-Based Markdown Price",
            color="#ff7f0e",
            linestyle="-.",
            linewidth=1.8,
        )
        ax1.set_title(
            f"Single-Episode Dynamic Pricing Trajectory ({rl_key} vs. Competitor & Heuristic)",
            fontweight="bold",
        )
        ax1.set_ylabel("Daily Price ($)")
        ax1.legend(loc="upper right", frameon=True)

        ax2_twin = ax2.twinx()
        ax2.plot(
            days,
            rl_trace.daily_inventories,
            label=f"{rl_key} Remaining Inventory",
            color="#1f77b4",
            linewidth=2.2,
        )
        ax2.plot(
            days,
            rule_trace.daily_inventories,
            label="Rule-Based Remaining Inventory",
            color="#ff7f0e",
            linestyle="-.",
            linewidth=1.8,
        )
        ax2_twin.bar(
            days,
            rl_trace.daily_units_sold,
            label=f"{rl_key} Daily Units Sold",
            color="#1f77b4",
            alpha=0.25,
            width=0.7,
        )
        ax2.set_title("Inventory Drawdown & Daily Units Sold", fontweight="bold")
        ax2.set_xlabel("Day of Season")
        ax2.set_ylabel("Remaining Inventory (Units)")
        ax2_twin.set_ylabel("Daily Units Sold")
        ax2_twin.grid(False)

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)

        fig.tight_layout()
        fig.savefig(plots_dir / "pricing_trajectory.png", dpi=200)
        plt.close(fig)

    # 3. metrics_comparison.png
    agent_names = list(results.keys())
    lifts = [results[k].lift_vs_static_pct for k in agent_names]
    sell_throughs = [results[k].sell_through_pct for k in agent_names]

    x = np.arange(len(agent_names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=200)
    bars1 = ax.bar(
        x - width / 2,
        lifts,
        width,
        label="Margin Lift vs. Static (%)",
        color="#1f77b4",
    )
    bars2 = ax.bar(
        x + width / 2,
        sell_throughs,
        width,
        label="Sell-Through Rate (%)",
        color="#2ca02c",
    )

    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_title(
        "Holdout Benchmark Comparison: Margin Lift (%) & Sell-Through Rate (%)",
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(agent_names)
    ax.set_ylabel("Percentage (%)")
    ax.legend(loc="best", frameon=True)

    for b in bars1:
        h = b.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 1.2 if h >= 0 else -3.0
        ax.annotate(
            f"{h:+.1f}%",
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
        )
    for b in bars2:
        h = b.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(plots_dir / "metrics_comparison.png", dpi=200)
    plt.close(fig)


def run_evaluation(
    config_path: str = "configs/default.yaml",
    n_episodes: int | None = None,
    models_dir: str = "artifacts/models",
    plots_dir: str = "artifacts/plots",
) -> list[AgentBenchmarkResult]:
    """Run holdout evaluation across 500 unseen seeds and generate charts."""
    cfg = load_yaml_config(config_path)
    env_cfg = cfg.get("env", {})
    eval_cfg = cfg.get("evaluation", {})

    num_eps = int(
        n_episodes if n_episodes is not None else eval_cfg.get("n_episodes", 500)
    )
    seed_start = int(eval_cfg.get("holdout_seed_start", 10000))
    seeds = [seed_start + i for i in range(num_eps)]

    min_p = float(env_cfg.get("min_price", 20.0))
    max_p = float(env_cfg.get("max_price", 100.0))
    ref_p = float(env_cfg.get("ref_price", 60.0))

    policies: dict[str, Callable[[np.ndarray, dict[str, Any]], np.ndarray]] = {}

    models_root = Path(models_dir)
    ppo_path = models_root / "ppo_best" / "best_model.zip"
    if ppo_path.exists():
        ppo_model = PPO.load(str(ppo_path), device="cpu")
        policies["PPO"] = (
            lambda obs, info, m=ppo_model: m.predict(obs, deterministic=True)[0]
        )

    sac_path = models_root / "sac_best" / "best_model.zip"
    if sac_path.exists():
        sac_model = SAC.load(str(sac_path), device="cpu")
        policies["SAC"] = (
            lambda obs, info, m=sac_model: m.predict(obs, deterministic=True)[0]
        )

    rule_agent = RuleBasedMarkdownAgent(
        min_price=min_p, max_price=max_p, ref_price=ref_p
    )
    static_agent = StaticPriceAgent(min_price=min_p, max_price=max_p, ref_price=ref_p)
    random_agent = RandomPriceAgent(
        min_price=min_p, max_price=max_p, ref_price=ref_p, seed=seed_start
    )

    policies["Rule-Based Markdown"] = lambda obs, info: rule_agent.predict(obs, info)
    policies["Static Price"] = lambda obs, info: static_agent.predict(obs, info)
    policies["Random Price"] = lambda obs, info: random_agent.predict(obs, info)

    results: dict[str, AgentBenchmarkResult] = {}
    for name, fn in policies.items():
        results[name] = evaluate_policy(name, fn, cfg, seeds)

    static_margin = results["Static Price"].mean_margin
    rule_margin = results["Rule-Based Markdown"].mean_margin

    for res in results.values():
        res.lift_vs_static_pct = compute_margin_lift(res.mean_margin, static_margin)
        res.lift_vs_rule_pct = compute_margin_lift(res.mean_margin, rule_margin)

    ordered = list(results.values())
    table_md = format_markdown_table(ordered)
    print(f"\n### PriceGym Holdout Benchmark ({num_eps} Unseen Episodes)\n")
    print(table_md)
    print()

    generate_plots(results, Path(plots_dir))
    print(f"[PriceGym] Saved evaluation charts to: {Path(plots_dir).resolve()}")
    return ordered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained RL models and baseline policies on 500 holdout episodes."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of holdout episodes (defaults to evaluation.n_episodes in config).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(config_path=args.config, n_episodes=args.episodes)
