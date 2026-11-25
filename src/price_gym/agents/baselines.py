"""Baseline pricing policies for e-commerce dynamic pricing benchmarks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BasePricingAgent(ABC):
    """Shared interface (`predict(obs, info) -> action`) for baseline pricing agents."""

    def __init__(
        self,
        min_price: float = 20.0,
        max_price: float = 100.0,
        ref_price: float = 60.0,
    ) -> None:
        self.min_price = float(min_price)
        self.max_price = float(max_price)
        self.ref_price = float(ref_price)

    def price_to_action(self, price: float) -> np.ndarray:
        """Convert continuous price in [min_price, max_price] to action in [-1, 1]."""
        span = max(self.max_price - self.min_price, 1e-6)
        norm = (float(price) - self.min_price) / span
        act = 2.0 * norm - 1.0
        return np.array([np.clip(act, -1.0, 1.0)], dtype=np.float32)

    @abstractmethod
    def predict(
        self,
        obs: np.ndarray,
        info: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Predict normalized action in [-1.0, 1.0] given observation and info."""


class StaticPriceAgent(BasePricingAgent):
    """Always sets price equal to ref_price."""

    def predict(
        self,
        obs: np.ndarray,
        info: dict[str, Any] | None = None,
    ) -> np.ndarray:
        return self.price_to_action(self.ref_price)


class RandomPriceAgent(BasePricingAgent):
    """Samples uniformly from the continuous action space [-1.0, 1.0]."""

    def __init__(
        self,
        min_price: float = 20.0,
        max_price: float = 100.0,
        ref_price: float = 60.0,
        seed: int = 42,
    ) -> None:
        super().__init__(min_price=min_price, max_price=max_price, ref_price=ref_price)
        self.rng = np.random.default_rng(seed)

    def predict(
        self,
        obs: np.ndarray,
        info: dict[str, Any] | None = None,
    ) -> np.ndarray:
        val = float(self.rng.uniform(-1.0, 1.0))
        return np.array([val], dtype=np.float32)


class RuleBasedMarkdownAgent(BasePricingAgent):
    """Realistic merchant markdown & inventory-pacing heuristic.

    Tracks inventory burn rate relative to time remaining:
    - Anchors partially toward competitor price.
    - Applies markdown discounts when inventory ratio exceeds time remaining ratio
      (inventory lagging behind target sell-through schedule).
    - Raises price slightly when inventory is tight or recent rolling demand spikes.
    - Smooths daily price changes against previous price.
    """

    def predict(
        self,
        obs: np.ndarray,
        info: dict[str, Any] | None = None,
    ) -> np.ndarray:
        inv_ratio = float(obs[0])
        time_rem_ratio = float(obs[1])
        norm_prev_price = float(obs[2])
        norm_comp_price = float(obs[3])
        norm_rolling_demand = float(obs[6])

        span = self.max_price - self.min_price
        prev_price = self.min_price + norm_prev_price * span
        comp_price = self.min_price + norm_comp_price * span

        # Anchor 70% to reference price and 30% to competitor price
        anchor_price = 0.70 * self.ref_price + 0.30 * comp_price

        # Inventory excess relative to linear burn schedule: positive -> too much stock left
        inventory_excess = inv_ratio - time_rem_ratio

        if inventory_excess > 0.08:
            # Lagging behind schedule: markdown up to 30%
            markdown_factor = 1.0 - min(0.30, 0.55 * inventory_excess)
            target_price = anchor_price * markdown_factor
        elif inventory_excess < -0.08:
            # Selling faster than schedule: scarcity markup up to 20%
            markup_factor = 1.0 + min(0.20, 0.45 * abs(inventory_excess))
            target_price = anchor_price * markup_factor
        else:
            target_price = anchor_price

        # Demand surge or end-of-season slow demand adjustment
        if norm_rolling_demand > 0.60:
            target_price *= 1.05
        elif norm_rolling_demand < 0.25 and time_rem_ratio < 0.35:
            target_price *= 0.92

        # Smooth transition from previous price to limit price volatility
        smoothed_price = 0.65 * target_price + 0.35 * prev_price
        return self.price_to_action(smoothed_price)
