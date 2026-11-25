"""Unit and Gymnasium compliance tests for DynamicPricingEnv."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from price_gym.envs.pricing_env import DynamicPricingEnv


def test_gymnasium_check_env() -> None:
    """Verify DynamicPricingEnv passes Gymnasium's official check_env suite."""
    env = DynamicPricingEnv()
    check_env(env, skip_render_check=True)


def test_deterministic_seeding() -> None:
    """Verify identical seeds produce identical observations, rewards, and info."""
    env1 = DynamicPricingEnv()
    env2 = DynamicPricingEnv()

    obs1, info1 = env1.reset(seed=12345)
    obs2, info2 = env2.reset(seed=12345)
    np.testing.assert_allclose(obs1, obs2, atol=1e-7)
    assert info1 == info2

    actions = [
        np.array([-0.8], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([0.6], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
    ]
    for act in actions:
        o1, r1, term1, trunc1, inf1 = env1.step(act)
        o2, r2, term2, trunc2, inf2 = env2.step(act)
        np.testing.assert_allclose(o1, o2, atol=1e-7)
        assert r1 == pytest.approx(r2, abs=1e-7)
        assert term1 == term2
        assert trunc1 == trunc2
        assert inf1 == inf2


def test_price_bounds_and_mapping() -> None:
    """Verify action space maps cleanly and clips out-of-bound actions to [min_price, max_price]."""
    env = DynamicPricingEnv()
    env.reset(seed=42)

    # Minimum action -> min_price
    _, _, _, _, info_min = env.step(np.array([-1.0], dtype=np.float32))
    assert info_min["price"] == pytest.approx(env.min_price, abs=1e-5)

    # Maximum action -> max_price
    _, _, _, _, info_max = env.step(np.array([1.0], dtype=np.float32))
    assert info_max["price"] == pytest.approx(env.max_price, abs=1e-5)

    # Out-of-bounds actions clip to [min_price, max_price]
    _, _, _, _, info_under = env.step(np.array([-5.0], dtype=np.float32))
    assert info_under["price"] == pytest.approx(env.min_price, abs=1e-5)

    _, _, _, _, info_over = env.step(np.array([5.0], dtype=np.float32))
    assert info_over["price"] == pytest.approx(env.max_price, abs=1e-5)


def test_zero_inventory_early_termination() -> None:
    """Verify episode terminates early when inventory reaches 0."""
    custom_cfg = {
        "env": {
            "episode_days": 90,
            "initial_inventory": 5,
            "min_price": 20.0,
            "max_price": 100.0,
            "ref_price": 60.0,
            "unit_cost": 25.0,
        },
        "demand": {
            "base_demand": 25.0,
            "elasticity": 1.8,
            "cross_elasticity": 0.7,
        },
    }
    env = DynamicPricingEnv(config=custom_cfg)
    env.reset(seed=7)

    terminated = False
    truncated = False
    steps = 0
    last_info = {}
    while not (terminated or truncated):
        _, _, terminated, truncated, last_info = env.step(
            np.array([-1.0], dtype=np.float32)
        )
        steps += 1

    assert terminated is True
    assert truncated is False
    assert steps < 90
    assert last_info["inventory"] == 0
    assert last_info["stockout"] is True


def test_negative_margin_when_priced_below_cost() -> None:
    """Verify pricing below unit_cost (min_price=20 < unit_cost=25) yields negative daily margin."""
    env = DynamicPricingEnv()
    env.reset(seed=99)

    # Action -1.0 sets price to min_price (20.0), which is below unit_cost (25.0)
    _, reward, _, _, info = env.step(np.array([-1.0], dtype=np.float32))
    assert info["price"] == pytest.approx(20.0, abs=1e-5)
    if info["units_sold"] > 0:
        assert info["daily_margin"] < 0.0
        assert reward < 0.0


def test_distinct_obs_and_info_references() -> None:
    """Verify reset() and step() never reuse or mutate observation or info references."""
    env = DynamicPricingEnv()
    obs0, info0 = env.reset(seed=42)
    obs1, _, _, _, info1 = env.step(np.array([0.0], dtype=np.float32))

    assert obs0 is not obs1
    assert info0 is not info1

