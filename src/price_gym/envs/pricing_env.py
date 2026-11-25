"""Custom Gymnasium Environment for E-Commerce Dynamic Pricing."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from price_gym.utils.demand import (
    MarketConfig,
    compute_cyclical_time_features,
    compute_expected_demand,
    realize_demand,
    sample_demand_shock,
    step_competitor_price,
)


def load_yaml_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration from path or default location."""
    if config_path is None:
        default_path = Path(__file__).resolve().parents[3] / "configs" / "default.yaml"
        if default_path.exists():
            with open(default_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}
    path = Path(config_path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class DynamicPricingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Single-SKU Seasonal Dynamic Pricing Gymnasium Environment.

    Observation Space (Box, shape=(7,), float32):
        0: inventory_ratio      in [0, 1]
        1: time_remaining_ratio in [0, 1]
        2: norm_prev_price      in [0, 1]
        3: norm_comp_price      in [0, 1]
        4: seasonality_sin      in [-1, 1]
        5: seasonality_cos      in [-1, 1]
        6: norm_rolling_demand  in [0, 1]

    Action Space (Box, shape=(1,), float32, range=[-1.0, 1.0]):
        Maps linearly to continuous price in [min_price, max_price].
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        raw_cfg = config if config is not None else load_yaml_config(config_path)
        env_cfg = raw_cfg.get("env", {})
        rew_cfg = raw_cfg.get("reward", {})

        self.episode_days: int = int(env_cfg.get("episode_days", 90))
        self.initial_inventory: int = int(env_cfg.get("initial_inventory", 500))
        self.market_cfg: MarketConfig = MarketConfig.from_dict(raw_cfg)

        self.min_price: float = self.market_cfg.min_price
        self.max_price: float = self.market_cfg.max_price
        self.ref_price: float = self.market_cfg.ref_price
        self.unit_cost: float = self.market_cfg.unit_cost

        self.volatility_penalty: float = float(rew_cfg.get("volatility_penalty", 12.0))
        self.holding_cost_per_unit: float = float(
            rew_cfg.get("holding_cost_per_unit", 0.03)
        )
        self.dead_stock_penalty: float = float(rew_cfg.get("dead_stock_penalty", 8.0))
        self.reward_scale: float = float(rew_cfg.get("reward_scale", 100.0))

        obs_low = np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0], dtype=np.float32)
        obs_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, shape=(7,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )

        # Internal episode state
        self.current_day: int = 0
        self.current_inventory: int = self.initial_inventory
        self.prev_price: float = self.ref_price
        self.comp_price: float = self.ref_price
        self.demand_history: deque[int] = deque(maxlen=7)

    def action_to_price(self, action: np.ndarray | float) -> float:
        """Map action in [-1.0, 1.0] linearly to continuous price in [min_price, max_price]."""
        act_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        clipped_act = float(np.clip(act_arr[0], -1.0, 1.0))
        norm = 0.5 * (clipped_act + 1.0)
        price = self.min_price + norm * (self.max_price - self.min_price)
        return float(np.clip(price, self.min_price, self.max_price))

    def price_to_action(self, price: float) -> np.ndarray:
        """Map continuous price in [min_price, max_price] to action in [-1.0, 1.0]."""
        price_span = max(self.max_price - self.min_price, 1e-6)
        norm = (float(price) - self.min_price) / price_span
        act = 2.0 * norm - 1.0
        return np.array([np.clip(act, -1.0, 1.0)], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        """Build a brand-new float32 normalized observation array."""
        price_span = max(self.max_price - self.min_price, 1e-6)

        inv_ratio = np.clip(
            self.current_inventory / max(self.initial_inventory, 1), 0.0, 1.0
        )
        time_rem_ratio = np.clip(
            (self.episode_days - self.current_day) / max(self.episode_days, 1),
            0.0,
            1.0,
        )
        norm_prev_price = np.clip(
            (self.prev_price - self.min_price) / price_span, 0.0, 1.0
        )
        norm_comp_price = np.clip(
            (self.comp_price - self.min_price) / price_span, 0.0, 1.0
        )
        sin_t, cos_t = compute_cyclical_time_features(
            self.current_day, self.episode_days
        )

        rolling_avg = (
            float(np.mean(self.demand_history))
            if len(self.demand_history) > 0
            else self.market_cfg.base_demand
        )
        norm_rolling_demand = np.clip(
            rolling_avg / max(self.market_cfg.base_demand * 2.0, 1e-6), 0.0, 1.0
        )

        obs = np.array(
            [
                inv_ratio,
                time_rem_ratio,
                norm_prev_price,
                norm_comp_price,
                np.clip(sin_t, -1.0, 1.0),
                np.clip(cos_t, -1.0, 1.0),
                norm_rolling_demand,
            ],
            dtype=np.float32,
        )
        return obs.copy()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to an initial seasonal state."""
        super().reset(seed=seed)

        self.current_day = 0
        self.current_inventory = self.initial_inventory
        self.prev_price = self.ref_price
        self.comp_price = step_competitor_price(
            self.ref_price, self.market_cfg, self.np_random
        )
        self.demand_history.clear()
        self.demand_history.append(int(round(self.market_cfg.base_demand)))

        obs = self._get_obs()
        info: dict[str, Any] = {
            "day": int(self.current_day),
            "price": float(self.prev_price),
            "comp_price": float(self.comp_price),
            "units_sold": 0,
            "daily_margin": 0.0,
            "inventory": int(self.current_inventory),
            "stockout": False,
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one daily pricing step in the market."""
        price = self.action_to_price(action)
        comp_price_today = float(self.comp_price)

        shock = sample_demand_shock(self.market_cfg, self.np_random)
        expected_demand = compute_expected_demand(
            price=price,
            comp_price=comp_price_today,
            day=self.current_day,
            episode_days=self.episode_days,
            cfg=self.market_cfg,
            demand_shock=shock,
        )
        units_sold, sampled_demand = realize_demand(
            expected_demand=expected_demand,
            current_inventory=self.current_inventory,
            rng=self.np_random,
        )

        self.current_inventory -= units_sold
        self.demand_history.append(units_sold)
        self.current_day += 1

        # Reward components
        daily_margin = (price - self.unit_cost) * units_sold
        price_change_ratio = (price - self.prev_price) / max(self.ref_price, 1e-6)
        vol_penalty = -self.volatility_penalty * (price_change_ratio**2)
        holding_penalty = -self.holding_cost_per_unit * self.current_inventory

        dead_stock_cost = 0.0
        if self.current_day >= self.episode_days and self.current_inventory > 0:
            dead_stock_cost = -self.dead_stock_penalty * self.current_inventory

        raw_reward = daily_margin + vol_penalty + holding_penalty + dead_stock_cost
        scaled_reward = float(raw_reward / max(self.reward_scale, 1e-6))

        # Update state for next step
        price_volatility_abs = abs(price - self.prev_price)
        self.prev_price = price
        self.comp_price = step_competitor_price(
            comp_price_today, self.market_cfg, self.np_random
        )

        stockout = self.current_inventory <= 0
        terminated = bool(stockout)
        truncated = bool(self.current_day >= self.episode_days and not terminated)

        obs = self._get_obs()
        info: dict[str, Any] = {
            "day": int(self.current_day),
            "price": float(price),
            "comp_price": float(comp_price_today),
            "expected_demand": float(expected_demand),
            "sampled_demand": int(sampled_demand),
            "units_sold": int(units_sold),
            "daily_margin": float(daily_margin),
            "price_volatility": float(price_volatility_abs),
            "inventory": int(self.current_inventory),
            "stockout": bool(stockout),
        }
        return obs, scaled_reward, terminated, truncated, info

