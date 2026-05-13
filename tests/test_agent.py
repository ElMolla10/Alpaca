import numpy as np
import pandas as pd
import pytest

from app.agent.arl_agent import ARLAgent, BlockContext, default_style


def feature_frame(ret5, vol20, volume, close=100.0):
    rows = 30
    return pd.DataFrame(
        {
            "close": np.full(rows, close),
            "vol20": np.full(rows, vol20),
            "ret5": np.full(rows, ret5),
            "volume": np.full(rows, volume),
        }
    )


def selection_fixture():
    return {
        "MOMO": feature_frame(ret5=2.0, vol20=0.60, volume=1_000_000),
        "DOWN": feature_frame(ret5=-2.0, vol20=0.60, volume=900_000),
        "CALM": feature_frame(ret5=0.1, vol20=0.20, volume=800_000),
        "ILLIQ": feature_frame(ret5=1.0, vol20=0.50, volume=1),
        "FLAT": feature_frame(ret5=1.0, vol20=0.01, volume=1_000_000),
    }


def test_universe_filtering_and_profile_ranking():
    data = selection_fixture()

    high = ARLAgent(default_style("high_risk_short_term"))
    high_frame = high.build_selection_frame(data)
    assert "FLAT" not in high_frame.index
    assert "ILLIQ" not in high_frame.index
    assert list(high_frame.index) == sorted(
        high_frame.index, key=lambda sym: high_frame.loc[sym, "score"], reverse=True
    )
    assert high_frame.index[0] == "MOMO"

    low = ARLAgent(default_style("low_risk_long_term"))
    low_frame = low.build_selection_frame(data)
    assert low_frame.index[0] == "CALM"

    medium = ARLAgent(default_style("medium_risk_swing"))
    medium_frame = medium.build_selection_frame(data)
    assert set(medium_frame.index) == {"MOMO", "DOWN", "CALM"}
    assert medium_frame["score"].is_monotonic_decreasing


def test_epsilon_greedy_selection_exploit_and_middle_explore(monkeypatch):
    style = default_style("high_risk_short_term")
    style.max_symbols = 3
    agent = ARLAgent(style)
    feats = pd.DataFrame(
        {"score": range(12, 0, -1), "ret5": 0.0, "sigma_pct": 0.2, "liq": 1_000.0},
        index=[f"S{i}" for i in range(12)],
    )

    agent._eps = 0.0
    assert agent.select_universe(feats) == ["S0", "S1", "S2"]

    agent._eps = 1.0
    monkeypatch.setattr(np.random, "rand", lambda: 0.0)
    monkeypatch.setattr(np.random, "randint", lambda *_args, **_kwargs: 123)
    selected = agent.select_universe(feats)
    middle = set(feats.iloc[3:9].index)
    assert len(selected) == 3
    assert set(selected) <= middle


def test_epsilon_decay_and_floor():
    agent = ARLAgent(default_style("high_risk_short_term"))
    start = agent._eps

    agent.update_rewards({})
    assert agent._eps == pytest.approx(start * 0.995)

    for _ in range(2_000):
        agent.update_rewards({})
    assert agent._eps == pytest.approx(agent._eps_min)


def test_confidence_interpolation_midpoint_and_extremes():
    style = default_style("high_risk_short_term")
    agent = ARLAgent(style)
    ctx = BlockContext(is_friday=False, is_late=False, minutes_to_close=120.0, equity=100_000.0)

    agent._reward_ema["AAPL"] = 0.0
    mid = agent.decide_for_symbol("AAPL", ctx)
    assert mid.allow
    assert mid.band_R_override == pytest.approx((style.base_band_R * 1.20 + style.base_band_R * 0.85) / 2)
    assert mid.ema_hl_override == round((max(3, style.base_ema_hl + 3) + max(2, style.base_ema_hl - 2)) / 2)
    assert mid.size_mult == pytest.approx(style.base_size_mult)

    agent._reward_ema["AAPL"] = -100.0
    conservative = agent.decide_for_symbol("AAPL", ctx)
    assert conservative.band_R_override == pytest.approx(style.base_band_R * 1.20)
    assert conservative.size_mult == pytest.approx(style.base_size_mult * 0.70)

    agent._reward_ema["AAPL"] = 100.0
    aggressive = agent.decide_for_symbol("AAPL", ctx)
    assert aggressive.band_R_override == pytest.approx(style.base_band_R * 0.85)
    assert aggressive.size_mult == pytest.approx(style.base_size_mult * 1.30)


def test_temporal_gates():
    high = ARLAgent(default_style("high_risk_short_term"))
    medium = ARLAgent(default_style("medium_risk_swing"))
    low = ARLAgent(default_style("low_risk_long_term"))

    near_close = BlockContext(is_friday=False, is_late=False, minutes_to_close=30.0, equity=1.0)
    assert not high.decide_for_symbol("AAPL", near_close).allow

    friday_late = BlockContext(is_friday=True, is_late=True, minutes_to_close=120.0, equity=1.0)
    assert high.decide_for_symbol("AAPL", friday_late).allow
    assert not medium.decide_for_symbol("AAPL", friday_late).allow
    assert not low.decide_for_symbol("AAPL", friday_late).allow


def test_reward_ema_correctness():
    agent = ARLAgent(default_style("high_risk_short_term"))
    rewards = [1.0, -0.5, 2.0, 0.25]
    alpha = 0.25
    expected = 0.0

    for reward in rewards:
        expected = (1.0 - alpha) * expected + alpha * reward
        agent.update_rewards({"AAPL": reward})

    analytical = sum(alpha * ((1.0 - alpha) ** (len(rewards) - i - 1)) * r for i, r in enumerate(rewards))
    assert expected == pytest.approx(analytical)
    assert agent._reward_ema["AAPL"] == pytest.approx(analytical)
    assert agent._reward_count["AAPL"] == len(rewards)


def test_persistence_round_trip_decision_equivalence():
    style = default_style("medium_risk_swing")
    agent = ARLAgent(style)
    agent.update_rewards({"AAPL": 1.5, "MSFT": -0.75})
    restored = ARLAgent(style, persisted=agent.export_state())
    ctx = BlockContext(is_friday=False, is_late=False, minutes_to_close=90.0, equity=50_000.0)

    assert restored.export_state()["reward_ema"] == pytest.approx(agent.export_state()["reward_ema"])
    assert restored.decide_for_symbol("AAPL", ctx) == agent.decide_for_symbol("AAPL", ctx)
    assert restored.decide_for_symbol("MSFT", ctx) == agent.decide_for_symbol("MSFT", ctx)
