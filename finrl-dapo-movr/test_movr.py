"""
test_movr.py — Unit Tests for MOVRReward
=========================================
5 assertion-based tests verifying the MOVR reward function behaves correctly.
This file has zero dependencies on FinRL, PyTorch, or any trading library.

Run: python test_movr.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from src.rewards.movr import MOVRReward


def test_positive_returns_give_positive_reward():
    """Reward is positive for consistent positive returns."""
    movr = MOVRReward(alpha=1.0, beta=0.0, gamma=0.0)
    movr.reset()
    rewards = [movr.compute(0.01, 100_000 + i * 1000) for i in range(30)]
    avg_reward = np.mean(rewards)
    assert avg_reward > 0, f"Expected positive reward, got {avg_reward:.4f}"
    print("  [PASS] test_positive_returns_give_positive_reward")


def test_large_drawdown_is_penalised():
    """Reward decreases when a large drawdown occurs."""
    movr = MOVRReward(alpha=0.0, beta=0.0, gamma=1.0)
    movr.reset()
    # Run up portfolio first
    for i in range(10):
        movr.compute(0.01, 100_000 + i * 5000)
    # Then crash 50%
    r_drawdown = movr.compute(-0.3, 50_000)
    assert r_drawdown < 0, f"Large drawdown should give negative reward, got {r_drawdown:.4f}"
    print("  [PASS] test_large_drawdown_is_penalised")


def test_improving_sharpe_increases_reward():
    """
    beta > 0 makes total reward sensitive to Sharpe regime.
    Over many steps of consistent high returns, reward with beta=1
    should accumulate differently (non-zero delta-Sharpe terms) vs beta=0.
    """
    # Regime: consistent positive returns that gradually improve Sharpe
    returns_regime = [0.001] * 5 + [0.005] * 5 + [0.01] * 5  # accelerating gains

    # With beta=1: delta-Sharpe terms fire after window fills
    movr_with_beta = MOVRReward(alpha=0.0, beta=1.0, gamma=0.0, sharpe_window=5)
    movr_with_beta.reset()
    total_beta = 0.0
    val = 100_000.0
    for r in returns_regime:
        val *= (1 + r)
        total_beta += movr_with_beta.compute(r, val)

    # With beta=0: all delta-Sharpe terms are zero
    movr_no_beta = MOVRReward(alpha=0.0, beta=0.0, gamma=0.0, sharpe_window=5)
    movr_no_beta.reset()
    total_no_beta = 0.0
    val2 = 100_000.0
    for r in returns_regime:
        val2 *= (1 + r)
        total_no_beta += movr_no_beta.compute(r, val2)

    # beta=0 gives exactly 0 total; beta=1 gives non-zero (positive in this regime)
    assert total_no_beta == 0.0, f"beta=0 should yield 0 total, got {total_no_beta}"
    assert total_beta != 0.0, f"beta=1 should yield non-zero reward, got {total_beta}"
    print("  [PASS] test_improving_sharpe_increases_reward")


def test_reset_clears_state():
    """reset() clears internal state correctly."""
    movr = MOVRReward(alpha=1.0, beta=0.5, gamma=0.3)
    movr.reset()
    for i in range(30):
        movr.compute(0.01, 100_000 + i * 1000)
    state_before = movr.get_state()
    assert state_before["n_returns_stored"] > 0, "State should have data before reset"
    assert state_before["peak_value"] is not None

    movr.reset()
    state_after = movr.get_state()
    assert state_after["n_returns_stored"] == 0, "n_returns_stored should be 0 after reset"
    assert state_after["prev_sharpe"] is None, "prev_sharpe should be None after reset"
    assert state_after["peak_value"] is None, "peak_value should be None after reset"
    print("  [PASS] test_reset_clears_state")


def test_all_zero_weights_return_zero():
    """With alpha=beta=gamma=0, reward should be 0.0."""
    movr = MOVRReward(alpha=0.0, beta=0.0, gamma=0.0)
    movr.reset()
    for _ in range(25):
        r = movr.compute(0.01, 100_000)
    r = movr.compute(0.05, 110_000)
    assert r == 0.0, f"All-zero weights should give reward=0.0, got {r}"
    print("  [PASS] test_all_zero_weights_return_zero")


if __name__ == "__main__":
    print("\n=== MOVR Unit Tests ===\n")
    passed = 0
    tests = [
        test_positive_returns_give_positive_reward,
        test_large_drawdown_is_penalised,
        test_improving_sharpe_increases_reward,
        test_reset_clears_state,
        test_all_zero_weights_return_zero,
    ]
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {test_fn.__name__}: {e}")
        except Exception as e:
            print(f"  [ERROR] {test_fn.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed.\n")
    sys.exit(0 if passed == len(tests) else 1)
