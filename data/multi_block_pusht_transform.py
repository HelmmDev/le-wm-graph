"""Per-sample transform that prepares multi-block PushT samples for the LeWM
graph-encoder data dict contract.

Modeled after ``data/bouncing_balls_transform.py`` but for a robot-controlled
multi-object scene. The HDF5 dataset stores:

  ``positions``           (T, n_real, 2) float32 — raw px in [0, box_size]
  ``target_positions``    (T, n_real, 2) float32 — repeated per-step (constant)
  ``orientations``        (T, n_real)    float32 — radians
  ``target_orientations`` (T, n_real)    float32

After this transform, the sample contains:

  ``info["positions"]``           (T, n_max, 2) float, normalized to [-1, 1]
  ``info["mask"]``                (T, n_max)    bool, True = real block
  ``info["target_positions"]``    (T, n_max, 2) float, normalized to [-1, 1]
  ``info["target_mask"]``         (T, n_max)    bool, True = real target
  ``info["orientations"]``        (T, n_max)    float, padded with zeros
  ``info["target_orientations"]`` (T, n_max)    float, padded with zeros
  ``info["action"]``              (T, action_dim=2) float — passed through

Unlike bouncing balls (autonomous physics), action IS provided by the env, so
we do NOT inject a zero action — we leave any pre-existing ``action`` in the
sample untouched.
"""
from __future__ import annotations

import torch


class MultiBlockPushTGraphTransform:
    """Pad/mask/normalize positions, orientations, and goal info.

    Args:
        n_real: Number of real blocks per timestep (one of {2, 3, 5, 8}).
        n_max: Max object slots in the padded tensor (locked at 8 across
            the multi-block scaling scan).
        box_size: Side length of the simulation box, used to normalize raw
            pixel coordinates to ``[-1, 1]``.
        action_dim: Width of the action vector (2 for end-effector velocity).
            Provided for symmetry with the bouncing-balls transform; we do
            not synthesize zeros here.
    """

    def __init__(
        self,
        n_real: int,
        n_max: int = 8,
        box_size: int = 224,
        action_dim: int = 2,
    ) -> None:
        if n_real > n_max:
            raise ValueError(
                f"n_real ({n_real}) must be <= n_max ({n_max})"
            )
        self.n_real = int(n_real)
        self.n_max = int(n_max)
        self.box_size = float(box_size)
        self.action_dim = int(action_dim)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _to_float_tensor(self, x) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        return x.float()

    def _normalize_positions(self, pos: torch.Tensor) -> torch.Tensor:
        """[0, box_size] -> [-1, 1]."""
        return pos / self.box_size * 2.0 - 1.0

    def _pad_TxNxD(
        self, x: torch.Tensor, T: int, D: int
    ) -> torch.Tensor:
        out = x.new_zeros((T, self.n_max, D))
        out[:, : self.n_real, :] = x
        return out

    def _pad_TxN(self, x: torch.Tensor, T: int) -> torch.Tensor:
        out = x.new_zeros((T, self.n_max))
        out[:, : self.n_real] = x
        return out

    # ------------------------------------------------------------------
    # __call__
    # ------------------------------------------------------------------
    def __call__(self, sample: dict) -> dict:
        T_ref = None

        # --- positions: (T, n_real, 2) raw px -> (T, n_max, 2) in [-1, 1] ---
        if "positions" in sample:
            pos = self._to_float_tensor(sample["positions"])
            T, N, D = pos.shape
            if N != self.n_real or D != 2:
                raise ValueError(
                    f"Expected positions of shape (T, {self.n_real}, 2), "
                    f"got {tuple(pos.shape)}"
                )
            pos_norm = self._normalize_positions(pos)
            sample["positions"] = self._pad_TxNxD(pos_norm, T, 2)
            mask = torch.zeros((T, self.n_max), dtype=torch.bool)
            mask[:, : self.n_real] = True
            sample["mask"] = mask
            T_ref = T

        # --- orientations: (T, n_real) -> (T, n_max), padded with zeros ---
        if "orientations" in sample:
            ori = self._to_float_tensor(sample["orientations"])
            if ori.dim() != 2 or ori.shape[1] != self.n_real:
                raise ValueError(
                    f"Expected orientations of shape (T, {self.n_real}), "
                    f"got {tuple(ori.shape)}"
                )
            T = ori.shape[0]
            sample["orientations"] = self._pad_TxN(ori, T)
            T_ref = T_ref or T

        # --- target_positions: same handling, separate target_mask --------
        if "target_positions" in sample:
            tpos = self._to_float_tensor(sample["target_positions"])
            # Stored as (T, n_real, 2) in the HDF5; tolerate (n_real, 2) too.
            if tpos.dim() == 2:
                # (n_real, 2) — broadcast to (T, n_real, 2).
                if T_ref is None:
                    raise ValueError(
                        "target_positions has no T axis and no other key "
                        "established T — cannot broadcast."
                    )
                tpos = tpos.unsqueeze(0).expand(T_ref, -1, -1).contiguous()
            T, N, D = tpos.shape
            if N != self.n_real or D != 2:
                raise ValueError(
                    f"Expected target_positions (..., {self.n_real}, 2), "
                    f"got {tuple(tpos.shape)}"
                )
            tpos_norm = self._normalize_positions(tpos)
            sample["target_positions"] = self._pad_TxNxD(tpos_norm, T, 2)
            target_mask = torch.zeros((T, self.n_max), dtype=torch.bool)
            target_mask[:, : self.n_real] = True
            sample["target_mask"] = target_mask
            T_ref = T_ref or T

        # --- target_orientations -----------------------------------------
        if "target_orientations" in sample:
            tori = self._to_float_tensor(sample["target_orientations"])
            if tori.dim() == 1:
                if T_ref is None:
                    raise ValueError(
                        "target_orientations has no T axis and no T_ref."
                    )
                tori = tori.unsqueeze(0).expand(T_ref, -1).contiguous()
            T, N = tori.shape
            if N != self.n_real:
                raise ValueError(
                    f"Expected target_orientations (..., {self.n_real}), "
                    f"got {tuple(tori.shape)}"
                )
            sample["target_orientations"] = self._pad_TxN(tori, T)

        # --- action: passed through, no zeros injected -------------------
        # Multi-block PushT has a real 2D action; we leave it alone (this
        # matches how the pusht.yaml dataset config behaves).
        return sample
