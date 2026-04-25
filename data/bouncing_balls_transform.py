"""Per-sample transform that prepares bouncing-balls samples for the LeWM
graph-encoder data dict contract.

The HDF5 dataset stores ``positions`` of shape ``(T, N=5, 2)`` in raw pixel
coordinates (origin at top-left, range ``[0, box_size]``). The graph encoder
expects:

  ``info["positions"]``: ``(T, N_max=8, 2)`` float, normalized to ``[-1, 1]``
  ``info["mask"]``:      ``(T, N_max=8)``    bool, ``True`` = object present
  ``info["action"]``:    ``(T, action_dim=1)`` float — zeros (autonomous physics)

This transform produces those keys in-place. ``positions`` is *normalized and
padded*; the original raw positions are overwritten (the only consumer is the
graph encoder, which always wants the contract form). ``velocities`` (if
loaded) is padded but **not** normalized — its scale is left to downstream
column-normalizers in ``train.py`` so it remains usable for the state-vector
ablation.

Although bouncing balls always has exactly N=5 balls, this is parameterized
via ``n_real`` so the same transform skeleton extends cleanly to PushT (where
N varies per episode and would be read from the data dict instead of being
hard-coded).
"""
from __future__ import annotations

import torch


class BouncingBallsGraphTransform:
    """Pad/mask/normalize positions and inject a zero action placeholder.

    Args:
        n_real: Number of real objects per timestep (5 for bouncing balls).
        n_max: Max object slots in the padded tensor (8, locked across PushT).
        box_size: Side length of the simulation box, used to normalize raw
            pixel coordinates to ``[-1, 1]``.
        action_dim: Width of the placeholder action vector. Bouncing balls
            has no action; we emit zeros so the JEPA action_encoder still has
            a tensor of the expected shape.
    """

    def __init__(
        self,
        n_real: int = 5,
        n_max: int = 8,
        box_size: int = 64,
        action_dim: int = 1,
    ) -> None:
        if n_real > n_max:
            raise ValueError(
                f"n_real ({n_real}) must be <= n_max ({n_max})"
            )
        self.n_real = int(n_real)
        self.n_max = int(n_max)
        self.box_size = float(box_size)
        self.action_dim = int(action_dim)

    def __call__(self, sample: dict) -> dict:
        # --- positions: (T, N=5, 2) raw px -> (T, N_max=8, 2) in [-1, 1] ---
        if "positions" in sample:
            pos = sample["positions"]
            if not torch.is_tensor(pos):
                pos = torch.as_tensor(pos)
            pos = pos.float()
            T, N, D = pos.shape
            if N != self.n_real or D != 2:
                raise ValueError(
                    f"Expected positions of shape (T, {self.n_real}, 2), "
                    f"got {tuple(pos.shape)}"
                )
            # Normalize: px in [0, box_size] -> [-1, 1]
            pos_norm = pos / self.box_size * 2.0 - 1.0
            padded = pos.new_zeros((T, self.n_max, 2))
            padded[:, : self.n_real] = pos_norm
            sample["positions"] = padded

            mask = torch.zeros((T, self.n_max), dtype=torch.bool)
            mask[:, : self.n_real] = True
            sample["mask"] = mask

        # --- velocities: pad only; let downstream normalizer handle scale ---
        if "velocities" in sample:
            vel = sample["velocities"]
            if not torch.is_tensor(vel):
                vel = torch.as_tensor(vel)
            vel = vel.float()
            T = vel.shape[0]
            padded_v = vel.new_zeros((T, self.n_max, 2))
            padded_v[:, : self.n_real] = vel
            sample["velocities"] = padded_v

        # --- action: bouncing balls is autonomous; emit zeros placeholder ---
        # Use positions T if available, else fall back to pixels T.
        if "action" not in sample:
            if "positions" in sample:
                T = sample["positions"].shape[0]
            elif "pixels" in sample:
                T = sample["pixels"].shape[0]
            else:
                raise KeyError(
                    "Cannot infer T to build action placeholder: sample has "
                    "no 'positions' or 'pixels' key."
                )
            sample["action"] = torch.zeros((T, self.action_dim), dtype=torch.float32)

        return sample
