"""Market demand, price elasticity, seasonality, and competitor price dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MarketConfig:
    """Configuration parameters for the market and demand engine."""

    min_price: float = 20.0
    max_price: float = 100.0
    ref_price: float = 60.0
    unit_cost: float = 25.0
    base_demand: float = 6.5
    elasticity: float = 1.85
    cross_elasticity: float = 0.75
    weekly_amplitude: float = 0.18
    seasonal_amplitude: float = 0.25
    shock_std: float = 0.08
    shock_min: float = 0.75
    shock_max: float = 1.25
    comp_mean_reversion: float = 0.20
    comp_volatility: float = 2.0
    comp_promo_prob: float = 0.08
    comp_promo_discount: float = 0.20

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> MarketConfig:
        """Create a MarketConfig from the parsed YAML configuration dict."""
        env_cfg = config.get("env", {})
        dem_cfg = config.get("demand", {})
        return cls(
            min_price=float(env_cfg.get("min_price", 20.0)),
            max_price=float(env_cfg.get("max_price", 100.0)),
            ref_price=float(env_cfg.get("ref_price", 60.0)),
            unit_cost=float(env_cfg.get("unit_cost", 25.0)),
            base_demand=float(dem_cfg.get("base_demand", 6.5)),
            elasticity=float(dem_cfg.get("elasticity", 1.85)),
            cross_elasticity=float(dem_cfg.get("cross_elasticity", 0.75)),
            weekly_amplitude=float(dem_cfg.get("weekly_amplitude", 0.18)),
            seasonal_amplitude=float(dem_cfg.get("seasonal_amplitude", 0.25)),
            shock_std=float(dem_cfg.get("shock_std", 0.08)),
            shock_min=float(dem_cfg.get("shock_min", 0.75)),
            shock_max=float(dem_cfg.get("shock_max", 1.25)),
            comp_mean_reversion=float(dem_cfg.get("comp_mean_reversion", 0.20)),
            comp_volatility=float(dem_cfg.get("comp_volatility", 2.0)),
            comp_promo_prob=float(dem_cfg.get("comp_promo_prob", 0.08)),
            comp_promo_discount=float(dem_cfg.get("comp_promo_discount", 0.20)),
        )


def compute_seasonality(
    day: int,
    episode_days: int,
    weekly_amplitude: float = 0.18,
    seasonal_amplitude: float = 0.25,
) -> float:
    """Compute smooth weekly + mid-season sinusoidal demand multiplier.

    Weekend days (days 5 and 6 of each 7-day cycle) receive a positive boost,
    and a broader seasonal wave peaks around mid-season.
    """
    weekly_wave = weekly_amplitude * np.sin(2.0 * np.pi * (day % 7) / 7.0 - np.pi / 2.0)
    seasonal_wave = seasonal_amplitude * np.sin(np.pi * day / max(episode_days, 1))
    multiplier = 1.0 + weekly_wave + seasonal_wave
    return float(max(multiplier, 0.1))


def compute_cyclical_time_features(day: int, episode_days: int) -> tuple[float, float]:
    """Return (seasonality_sin, seasonality_cos) cyclical time features in [-1, 1]."""
    angle = 2.0 * np.pi * (day / max(episode_days, 1))
    return float(np.sin(angle)), float(np.cos(angle))


def step_competitor_price(
    prev_comp_price: float,
    cfg: MarketConfig,
    rng: np.random.Generator,
) -> float:
    """Advance competitor price using a mean-reverting random walk with promo drops."""
    drift = cfg.comp_mean_reversion * (cfg.ref_price - prev_comp_price)
    diffusion = cfg.comp_volatility * float(rng.normal(0.0, 1.0))
    candidate_price = prev_comp_price + drift + diffusion

    # Occasional promotional flash drop
    if float(rng.random()) < cfg.comp_promo_prob:
        candidate_price = cfg.ref_price * (1.0 - cfg.comp_promo_discount) + float(
            rng.normal(0.0, 1.0)
        )

    return float(np.clip(candidate_price, cfg.min_price, cfg.max_price))


def sample_demand_shock(cfg: MarketConfig, rng: np.random.Generator) -> float:
    """Sample a bounded multiplicative daily demand shock using the env RNG."""
    raw_shock = 1.0 + float(rng.normal(0.0, cfg.shock_std))
    return float(np.clip(raw_shock, cfg.shock_min, cfg.shock_max))


def compute_expected_demand(
    price: float,
    comp_price: float,
    day: int,
    episode_days: int,
    cfg: MarketConfig,
    demand_shock: float = 1.0,
) -> float:
    """Compute expected demand on day t using own-price and cross-price elasticity.

    Formula:
        expected_demand = base_demand
            * (price / ref_price)^(-elasticity)
            * (comp_price / price)^cross_elasticity
            * seasonality(t)
            * demand_shock
    """
    safe_price = max(float(price), 1e-3)
    safe_comp = max(float(comp_price), 1e-3)
    safe_ref = max(float(cfg.ref_price), 1e-3)

    own_effect = (safe_price / safe_ref) ** (-cfg.elasticity)
    cross_effect = (safe_comp / safe_price) ** cfg.cross_elasticity
    seasonality = compute_seasonality(
        day=day,
        episode_days=episode_days,
        weekly_amplitude=cfg.weekly_amplitude,
        seasonal_amplitude=cfg.seasonal_amplitude,
    )
    expected = cfg.base_demand * own_effect * cross_effect * seasonality * demand_shock
    return float(max(expected, 0.0))


def realize_demand(
    expected_demand: float,
    current_inventory: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Sample stochastic daily demand from Poisson and clip to available inventory.

    Returns:
        (units_sold, sampled_unconstrained_demand)
    """
    lam = max(float(expected_demand), 0.0)
    sampled_demand = int(rng.poisson(lam))
    units_sold = int(min(max(current_inventory, 0), sampled_demand))
    return units_sold, sampled_demand

