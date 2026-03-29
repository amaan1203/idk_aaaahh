"""
src/algorithms/dapo_movr.py — DAPO with MOVR Reward Hook
=========================================================
Novel contribution: wraps the DAPO algorithm to work with the MOVR environment.
Corresponds to: Section 3 of the paper draft.

Identical to dapo.py except it is instantiated with StockTradingEnvMOVR
rather than StockTradingEnvLLMRisk. The reward computation happens in the
environment, so no algorithmic changes are needed here — the novel work is
entirely in src/rewards/movr.py and src/envs/env_movr.py.

Inputs: StockTradingEnvMOVR instance + DAPO hyperparameters
Outputs: trained policy using MOVR reward signal
"""

from src.algorithms.dapo import DAPOAlgorithm


class DAPOMOVRAlgorithm(DAPOAlgorithm):
    """
    DAPO algorithm configured for MOVR reward environments.

    Inherits all DAPO logic unchanged. The MOVR reward is computed
    inside StockTradingEnvMOVR.step() and is therefore transparent
    to the RL algorithm — it simply sees a scalar reward per step.

    This class exists as a named variant for traceability and checkpointing.
    """

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        assert hasattr(env, "_movr"), (
            "DAPOMOVRAlgorithm requires a StockTradingEnvMOVR environment "
            "(must have a _movr attribute). "
            "Use dapo.py for standard environments."
        )

    def train(self, total_epochs, steps_per_epoch, checkpoint_dir=None, verbose=True):
        """Training loop identical to DAPO — MOVR reward handled by the env."""
        print(
            f"[DAPO+MOVR] alpha={self.env.movr_alpha:.2f} "
            f"beta={self.env.movr_beta:.2f} "
            f"gamma={self.env.movr_gamma:.2f}"
        )
        return super().train(
            total_epochs=total_epochs,
            steps_per_epoch=steps_per_epoch,
            checkpoint_dir=checkpoint_dir,
            verbose=verbose,
        )
