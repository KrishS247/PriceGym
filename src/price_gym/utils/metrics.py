"""Evaluation metrics and benchmarking utilities for PriceGym."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class EpisodeTrace:
    """Detailed daily trajectory and summary metrics for a single episode."""

    total_margin: float
    total_units_sold: int
    initial_inventory: int
    sell_through_rate: float
    stockout: bool
    mean_price_volatility: float
    daily_margins: np.ndarray
    cumulative_margins: np.ndarray
    daily_prices: np.ndarray
    daily_comp_prices: np.ndarray
    daily_inventories: np.ndarray
    daily_units_sold: np.ndarray


@dataclass
class AgentBenchmarkResult:
    """Aggregated benchmark statistics across N holdout evaluation episodes."""

    agent_name: str
    mean_margin: float
    std_margin: float
    sell_through_pct: float
    stockout_pct: float
    mean_volatility: float
    lift_vs_static_pct: float = 0.0
    lift_vs_rule_pct: float = 0.0
    mean_cumulative_margin_curve: np.ndarray = field(
        default_factory=lambda: np.zeros(90, dtype=np.float64)
    )
    std_cumulative_margin_curve: np.ndarray = field(
        default_factory=lambda: np.zeros(90, dtype=np.float64)
    )
    sample_trace: EpisodeTrace | None = None


def compute_margin_lift(agent_margin: float, baseline_margin: float) -> float:
    """Compute percentage margin lift over a baseline policy."""
    denom = abs(float(baseline_margin))
    if denom < 1e-9:
        return 0.0
    return float(((agent_margin - baseline_margin) / denom) * 100.0)


def compute_latency_percentiles(
    latencies_ms: list[float] | np.ndarray,
) -> dict[str, float]:
    """Return p50, p95, p99, and mean inference latency in milliseconds."""
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def run_policy_episode(
    env: Any,
    policy_fn: Callable[[np.ndarray, dict[str, Any]], np.ndarray],
    seed: int,
) -> EpisodeTrace:
    """Run one seeded evaluation episode and pad daily trajectories to episode_days."""
    obs, info = env.reset(seed=seed)
    horizon = int(env.episode_days)
    init_inv = int(env.initial_inventory)

    daily_margins = np.zeros(horizon, dtype=np.float64)
    daily_prices = np.zeros(horizon, dtype=np.float64)
    daily_comp_prices = np.zeros(horizon, dtype=np.float64)
    daily_inventories = np.zeros(horizon, dtype=np.float64)
    daily_units_sold = np.zeros(horizon, dtype=np.float64)
    volatilities: list[float] = []

    terminated = False
    truncated = False
    day_idx = 0
    last_price = float(env.ref_price)
    last_comp = float(info.get("comp_price", env.ref_price))
    stockout = False

    while not (terminated or truncated) and day_idx < horizon:
        action = policy_fn(obs, info)
        obs, _, terminated, truncated, info = env.step(action)

        last_price = float(info["price"])
        last_comp = float(info["comp_price"])
        daily_margins[day_idx] = float(info["daily_margin"])
        daily_prices[day_idx] = last_price
        daily_comp_prices[day_idx] = last_comp
        daily_inventories[day_idx] = float(info["inventory"])
        daily_units_sold[day_idx] = float(info["units_sold"])
        volatilities.append(float(info.get("price_volatility", 0.0)))
        stockout = bool(info["stockout"])
        day_idx += 1

    # If episode terminated early due to stockout, pad remaining days for aligned curves
    if day_idx < horizon:
        daily_prices[day_idx:] = last_price
        daily_comp_prices[day_idx:] = last_comp
        daily_inventories[day_idx:] = 0.0
        daily_units_sold[day_idx:] = 0.0
        daily_margins[day_idx:] = 0.0

    cumulative_margins = np.cumsum(daily_margins)
    total_margin = float(cumulative_margins[-1])
    total_sold = int(np.sum(daily_units_sold))
    sell_through_rate = (total_sold / max(init_inv, 1)) * 100.0
    mean_vol = float(np.mean(volatilities)) if volatilities else 0.0

    return EpisodeTrace(
        total_margin=total_margin,
        total_units_sold=total_sold,
        initial_inventory=init_inv,
        sell_through_rate=float(sell_through_rate),
        stockout=stockout,
        mean_price_volatility=mean_vol,
        daily_margins=daily_margins,
        cumulative_margins=cumulative_margins,
        daily_prices=daily_prices,
        daily_comp_prices=daily_comp_prices,
        daily_inventories=daily_inventories,
        daily_units_sold=daily_units_sold,
    )


def format_markdown_table(results: list[AgentBenchmarkResult]) -> str:
    """Format benchmark results into a clean Markdown summary table."""
    headers = [
        "Agent / Policy",
        "Mean Gross Margin ($)",
        "Lift vs. Static (%)",
        "Lift vs. Rule-Based (%)",
        "Sell-Through (%)",
        "Stockout Rate (%)",
        "Daily Price Volatility ($)",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in results:
        margin_str = f"${r.mean_margin:,.2f} ± ${r.std_margin:,.2f}"
        lift_static_str = f"{r.lift_vs_static_pct:+.2f}%"
        lift_rule_str = f"{r.lift_vs_rule_pct:+.2f}%"
        sell_str = f"{r.sell_through_pct:.2f}%"
        stockout_str = f"{r.stockout_pct:.2f}%"
        vol_str = f"${r.mean_volatility:.2f}"
        lines.append(
            f"| **{r.agent_name}** | {margin_str} | {lift_static_str} | "
            f"{lift_rule_str} | {sell_str} | {stockout_str} | {vol_str} |"
        )
    return "\n".join(lines)
