import torch
import numpy as np
from src.rewards.movr import MOVRReward

def _get_action_reward(completion: str, future_return: float) -> float:
    """
    Map completion (BUY/SELL/HOLD) to a reward based on future return.
    """
    compl = completion.upper()
    if "BUY" in compl:
        return future_return
    elif "SELL" in compl:
        return -future_return
    else:
        return 0.0

def grpo_vanilla_reward(prompts, completions, future_returns, **kwargs) -> list:
    """
    Standard accuracy-based reward (Future Return).
    """
    rewards = []
    for comp, fr in zip(completions, future_returns):
        rewards.append(_get_action_reward(comp, fr))
    return rewards

def dapo_reward(prompts, completions, future_returns, **kwargs) -> list:
    """
    Direct Alignment from Portfolio Optimization (DAPO).
    Eq 3 (simplified): r_t * sign(r_t)
    """
    rewards = []
    for comp, fr in zip(completions, future_returns):
        r = _get_action_reward(comp, fr)
        # Weight the reward by the magnitude of the signal
        rewards.append(r * 10.0) # Scale for better training signal
    return rewards

def dapo_movr_reward(
    prompts, completions, future_returns, return_history, 
    alpha=1.0, beta=0.5, gamma=0.3, **kwargs
) -> list:
    """
    DAPO + Multi-Objective Verifiable Reward (MOVR).
    Combines the directional signal with Sharpe and Drawdown components.
    """
    rewards = []
    # Initialize a temporary stateful MOVR calculator
    # Note: In a real GRPO setup, each completion might have its own history,
    # but here we use the snapshot history provided in the record.
    movr = MOVRReward(alpha=alpha, beta=beta, gamma=gamma)
    
    for comp, fr, history in zip(completions, future_returns, return_history):
        # 1. Base accuracy signal (DAPO)
        base_r = _get_action_reward(comp, fr)
        
        # 2. MOVR components
        # We simulate one step of MOVR using the history and current return
        # History is used to 'prime' the Sharpe ratio
        movr.reset()
        for h_r in history:
            movr._returns.append(h_r)
        
        # Calculate MOVR reward for the predicted action's return
        # We treat base_r as the 'portfolio_return' for this step
        # and assume unit portfolio value for drawdown penalty since we don't have cumulative state here
        movr_r = movr.compute(base_r, 1.0)
        
        rewards.append(movr_r)
        
    return rewards
