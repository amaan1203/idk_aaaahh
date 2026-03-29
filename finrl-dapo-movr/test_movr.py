"""
test_movr.py — Unit Tests for MOVRReward
=========================================
Run on any machine (no GPU, no ML frameworks, no FinRL):
    python test_movr.py

All 6 tests are assertion-based.
Exit code 0 = all pass. Exit code 1 = failure.
"""

import sys
from src.rewards.movr import MOVRReward


def test_positive_returns_positive_reward():
    """Consistent positive returns → positive accuracy-weighted reward."""
    m = MOVRReward(alpha=1.0, beta=0.0, gamma=0.0)
    m.reset()
    rewards = [m.compute(0.01, 1_000_000 + i * 10_000) for i in range(30)]
    assert all(r > 0 for r in rewards), f"Expected all positive, got: {rewards[:5]}"
    print("  ✓ positive returns → positive reward")


def test_drawdown_penalty_fires():
    """Large drawdown with gamma>0 should produce a negative reward."""
    m = MOVRReward(alpha=0.0, beta=0.0, gamma=1.0)
    m.reset()
    m.compute(0.1, 1_100_000)     # portfolio goes up
    r = m.compute(-0.2, 880_000)  # sharp drop below peak
    assert r < 0, f"Expected negative reward on drawdown, got: {r}"
    print("  ✓ drawdown penalty fires when gamma > 0")


def test_sharpe_component_returns_float():
    """delta_sharpe component must return a Python float."""
    m = MOVRReward(alpha=0.0, beta=1.0, gamma=0.0, sharpe_window=5)
    m.reset()
    r = None
    for i in range(6):
        r = m.compute(0.01, 1_000_000 + i * 5_000)
    assert isinstance(r, float), f"Expected float, got: {type(r)}"
    print("  ✓ delta_sharpe returns float")


def test_reset_clears_state():
    """reset() must clear all internal episode state."""
    m = MOVRReward(alpha=1.0, beta=0.5, gamma=0.3)
    m.reset()
    for i in range(25):
        m.compute(0.01, 1_000_000 + i * 5_000)
    assert m.state_dict()["n_returns"] > 0, "n_returns should be > 0 before reset"
    m.reset()
    s = m.state_dict()
    assert s["n_returns"] == 0, f"n_returns should be 0 after reset, got: {s['n_returns']}"
    assert s["prev_sharpe"] is None, f"prev_sharpe should be None after reset"
    print("  ✓ reset() clears all state")


def test_zero_weights_give_zero():
    """All zero weights must yield zero reward at every step."""
    m = MOVRReward(alpha=0.0, beta=0.0, gamma=0.0)
    m.reset()
    for _ in range(30):
        result = m.compute(0.01, 1_000_000)
        assert result == 0.0, f"Expected 0.0, got: {result}"
    print("  ✓ zero weights → zero reward")


def test_negative_weight_raises():
    """Negative weight must raise ValueError immediately."""
    raised = False
    try:
        MOVRReward(alpha=-1.0, beta=0.5, gamma=0.3)
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for negative alpha"
    print("  ✓ negative weight raises ValueError")


if __name__ == "__main__":
    print("Running MOVRReward unit tests...\n")
    test_positive_returns_positive_reward()
    test_drawdown_penalty_fires()
    test_sharpe_component_returns_float()
    test_reset_clears_state()
    test_zero_weights_give_zero()
    test_negative_weight_raises()
    print("\nAll 6 tests passed.")
    sys.exit(0)
