"""
src/algorithms/grpo_vanilla.py — Vanilla GRPO Algorithm
=======================================================
Adapted from dapo.py with two surgical changes to produce standard GRPO behaviour.
Corresponds to: the DeepStock / pre-DAPO baseline.

Changes from DAPO:
  1. Symmetric epsilon clipping (epsilon_low == epsilon_high)
  2. dynamic_sampling=False by default (all samples used)

Inputs: gymnasium environment, config dict
Outputs: trained policy + value network (GRPO objective)
"""

from src.algorithms.dapo import DAPOAlgorithm


class GRPOVanillaAlgorithm(DAPOAlgorithm):
    """
    Vanilla GRPO: DAPO with symmetric epsilon clipping and no dynamic sampling.

    This is the pre-DAPO baseline. The key differences from DAPOAlgorithm:
    - symmetric_clip=True forces epsilon_high = epsilon_low in loss computation
    - dynamic_sampling=False keeps all samples regardless of reward homogeneity
    """

    def __init__(
        self,
        env,
        epsilon_low: float = 0.2,
        **kwargs,
    ):
        # Force vanilla GRPO behaviour:
        # 1. epsilon_high = epsilon_low (symmetric clipping)
        # 2. dynamic_sampling = False (no uninformative-group filtering)
        super().__init__(
            env,
            epsilon_low=epsilon_low,
            epsilon_high=epsilon_low,        # Change 1: symmetric clipping
            dynamic_sampling=False,          # Change 2: disabled
            symmetric_clip=True,             # redundant but explicit
            **kwargs,
        )

    def train(self, total_epochs, steps_per_epoch, checkpoint_dir=None, verbose=True):
        print("[GRPO Vanilla] symmetric_clip=True, dynamic_sampling=False")
        return super().train(
            total_epochs=total_epochs,
            steps_per_epoch=steps_per_epoch,
            checkpoint_dir=checkpoint_dir,
            verbose=verbose,
        )
