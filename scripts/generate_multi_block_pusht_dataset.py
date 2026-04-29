"""Generate a multi-block PushT HDF5 dataset compatible with stable-worldmodel.

Schema (matching the conventions established by generate_bouncing_balls_dataset
and pusht_expert_train.h5):

  pixels              : (T_total, 224, 224, 3) uint8  Blosc-LZ4 chunks=(50, 224, 224, 3)
  state               : (T_total, n_blocks*6 + 4) float32  -- per-block (x, y, theta, vx, vy, omega)
                                                          + pusher (x, y, vx, vy)
  action              : (T_total, 2) float32   -- (vx, vy) in [-1, 1]
  positions           : (T_total, n_blocks, 2) float32  -- raw px in [0, box_size]
  orientations        : (T_total, n_blocks)    float32  -- radians
  target_positions    : (T_total, n_blocks, 2) float32  -- constant within an episode
  target_orientations : (T_total, n_blocks)    float32
  step_idx            : (T_total,)             int64
  ep_idx              : (T_total,)             int32
  ep_offset           : (n_episodes,)          int64
  ep_len              : (n_episodes,)          int32

Per-episode policy mix (drawn iid per episode):
  40%  pure random uniform actions
  40%  scripted single-block pushing with a PD policy
  20%  scripted multi-block: switch target block midway through

Single-process generation (CPU-only) so we don't compete with Phase 0
training on the GPU box.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  -- registers Blosc filter on import
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from envs.multi_block_pusht import MultiBlockPushT  # noqa: E402


def _default_output_path(n_blocks: int) -> str:
    return str(
        Path.home()
        / ".stable-wm"
        / f"multi_block_pusht_n{n_blocks}_train.h5"
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n_blocks", type=int, default=3, choices=[2, 3, 5, 8]
    )
    p.add_argument("--n_episodes", type=int, default=10_000)
    p.add_argument("--frames_per_episode", type=int, default=128)
    p.add_argument("--box_size", type=int, default=224)
    p.add_argument("--block_size", type=float, default=30.0)
    p.add_argument("--pusher_radius", type=float, default=10.0)
    p.add_argument("--dt", type=float, default=1.0 / 30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_path", type=str, default=None)
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args()


def _make_blosc_compression():
    return hdf5plugin.Blosc(
        cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE
    )


def _init_datasets(f: h5py.File, args: argparse.Namespace) -> None:
    blosc = _make_blosc_compression()
    H = args.box_size
    N = args.n_blocks
    state_dim = N * 6 + 4

    # Smaller chunk on T axis (50 vs 100) since 224x224 frames are 12x larger
    # than the bouncing-balls 64x64.
    f.create_dataset(
        "pixels",
        shape=(0, H, H, 3),
        maxshape=(None, H, H, 3),
        dtype=np.uint8,
        chunks=(50, H, H, 3),
        compression=blosc,
    )
    f.create_dataset(
        "state",
        shape=(0, state_dim),
        maxshape=(None, state_dim),
        dtype=np.float32,
        chunks=(1000, state_dim),
    )
    f.create_dataset(
        "action",
        shape=(0, 2),
        maxshape=(None, 2),
        dtype=np.float32,
        chunks=(1000, 2),
    )
    f.create_dataset(
        "positions",
        shape=(0, N, 2),
        maxshape=(None, N, 2),
        dtype=np.float32,
        chunks=(1000, N, 2),
    )
    f.create_dataset(
        "orientations",
        shape=(0, N),
        maxshape=(None, N),
        dtype=np.float32,
        chunks=(1000, N),
    )
    f.create_dataset(
        "target_positions",
        shape=(0, N, 2),
        maxshape=(None, N, 2),
        dtype=np.float32,
        chunks=(1000, N, 2),
    )
    f.create_dataset(
        "target_orientations",
        shape=(0, N),
        maxshape=(None, N),
        dtype=np.float32,
        chunks=(1000, N),
    )
    f.create_dataset(
        "step_idx", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(1000,)
    )
    f.create_dataset(
        "ep_idx", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=(1000,)
    )
    f.create_dataset(
        "ep_offset", shape=(0,), maxshape=(None,), dtype=np.int64
    )
    f.create_dataset(
        "ep_len", shape=(0,), maxshape=(None,), dtype=np.int32
    )


def _append(ds: h5py.Dataset, arr: np.ndarray) -> None:
    cur = ds.shape[0]
    ds.resize(cur + arr.shape[0], axis=0)
    ds[cur:] = arr


# ----------------------------------------------------------------------
# Scripted policies
# ----------------------------------------------------------------------
def _push_action(
    pusher_pos: np.ndarray,
    block_pos: np.ndarray,
    target_pos: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """A simple rule-based pushing controller.

    1. If pusher is far from the "behind block, opposite the target" approach
       point, drive there.
    2. Otherwise drive straight through the block toward the target.

    The action is unit-norm (clipped to [-1, 1]) with a small noise term so
    behavior cloning has a chance.
    """
    block_pos = np.asarray(block_pos, dtype=np.float32)
    target_pos = np.asarray(target_pos, dtype=np.float32)
    pusher_pos = np.asarray(pusher_pos, dtype=np.float32)

    to_target = target_pos - block_pos
    dist_to_target = np.linalg.norm(to_target)
    if dist_to_target < 1e-3:
        # Block is already at target — wiggle.
        return rng.uniform(-0.3, 0.3, size=2).astype(np.float32)
    push_dir = to_target / dist_to_target

    # Approach point: behind the block opposite the target, ~one block away.
    # To push block toward target, pusher must be on the far side of block
    # from the target — i.e., at block - push_dir * offset.
    approach = block_pos - push_dir * 25.0  # block_size ~ 30, so just behind

    to_approach = approach - pusher_pos
    d_app = np.linalg.norm(to_approach)
    to_block = block_pos - pusher_pos
    d_block = np.linalg.norm(to_block)

    # Are we behind the block already? Pusher is "behind" block (relative to
    # target) iff `to_block` is parallel to `+push_dir` — i.e., the block is
    # in the direction of the target from the pusher's perspective. Then
    # pushing toward the target via the block makes sense.
    behind_score = float(np.dot(push_dir, to_block / (d_block + 1e-6)))

    if behind_score > 0.6 and d_block < 35.0:
        # Push phase: drive through block toward target.
        action = push_dir
    else:
        # Approach phase: go to the approach point.
        action = to_approach / (d_app + 1e-6)

    # Add small noise so demos don't collapse to a deterministic mode.
    action = action + rng.normal(0.0, 0.1, size=2).astype(np.float32)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def _rollout_random(
    env: MultiBlockPushT,
    T: int,
    rng: np.random.Generator,
) -> dict:
    return _rollout(
        env, T, lambda step, st, info: rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
    )


def _rollout_scripted_single(
    env: MultiBlockPushT,
    T: int,
    rng: np.random.Generator,
) -> dict:
    target_block = int(rng.integers(0, env.n_blocks))

    def policy(step: int, st: dict, info: dict) -> np.ndarray:
        return _push_action(
            st["pusher_position"],
            st["positions"][target_block],
            st["target_positions"][target_block],
            rng,
        )

    return _rollout(env, T, policy)


def _rollout_scripted_multi(
    env: MultiBlockPushT,
    T: int,
    rng: np.random.Generator,
) -> dict:
    if env.n_blocks < 2:
        return _rollout_scripted_single(env, T, rng)
    a, b = rng.choice(env.n_blocks, size=2, replace=False)
    a, b = int(a), int(b)
    switch = max(15, T // 4)

    def policy(step: int, st: dict, info: dict) -> np.ndarray:
        target_block = a if step < switch else b
        return _push_action(
            st["pusher_position"],
            st["positions"][target_block],
            st["target_positions"][target_block],
            rng,
        )

    return _rollout(env, T, policy)


def _rollout(
    env: MultiBlockPushT,
    T: int,
    policy,
) -> dict:
    """Run a rollout under ``policy(step, state_dict, info_dict) -> action``.

    Records (frame_t, action_t, state_t) for t in [0, T). The first row is
    post-reset (action_0 is what we *just* commanded), matching the
    bouncing-balls / pusht conventions where row 0 is the initial obs.
    """
    H = env.box_size
    N = env.n_blocks
    pixels = np.empty((T, H, H, 3), dtype=np.uint8)
    actions = np.empty((T, 2), dtype=np.float32)
    positions = np.empty((T, N, 2), dtype=np.float32)
    orientations = np.empty((T, N), dtype=np.float32)
    velocities = np.empty((T, N, 2), dtype=np.float32)
    ang_vel = np.empty((T, N), dtype=np.float32)
    pusher_pos = np.empty((T, 2), dtype=np.float32)
    pusher_vel = np.empty((T, 2), dtype=np.float32)

    block_block_contact_total = 0
    block_pusher_contact_total = 0

    state = env.reset()
    info: dict = {}

    # Row 0: initial state, render, action chosen at t=0.
    pixels[0] = env.render()
    positions[0] = state["positions"]
    orientations[0] = state["orientations"]
    velocities[0] = state["velocities"]
    ang_vel[0] = state["angular_velocities"]
    pusher_pos[0] = state["pusher_position"]
    pusher_vel[0] = state["pusher_velocity"]
    target_positions = state["target_positions"].astype(np.float32)
    target_orientations = state["target_orientations"].astype(np.float32)
    actions[0] = policy(0, state, info)

    for t in range(1, T):
        frame, st, _r, _done, info = env.step(actions[t - 1])
        pixels[t] = frame
        positions[t] = st["positions"]
        orientations[t] = st["orientations"]
        velocities[t] = st["velocities"]
        ang_vel[t] = st["angular_velocities"]
        pusher_pos[t] = st["pusher_position"]
        pusher_vel[t] = st["pusher_velocity"]
        block_block_contact_total += int(info["block_block_contact_count"])
        block_pusher_contact_total += int(info["block_pusher_contact_count"])
        actions[t] = policy(t, st, info)

    # Final step using the last action so action_t aligns with state_t (the
    # convention used by the bouncing-balls/pusht datasets stores (s_t, a_t)
    # together; the consumer can ignore the trailing action of the episode).
    # Flat state: per-block (x, y, theta, vx, vy, omega) + pusher (x, y, vx, vy).
    state_flat = np.concatenate(
        [
            positions,                                  # (T, N, 2)
            orientations[..., None],                    # (T, N, 1)
            velocities,                                 # (T, N, 2)
            ang_vel[..., None],                         # (T, N, 1)
        ],
        axis=2,
    ).reshape(T, N * 6).astype(np.float32)
    pusher_flat = np.concatenate(
        [pusher_pos, pusher_vel], axis=1
    ).astype(np.float32)
    state_flat = np.concatenate([state_flat, pusher_flat], axis=1).astype(np.float32)

    target_pos_per_step = np.broadcast_to(
        target_positions[None, ...], (T, N, 2)
    ).copy()
    target_ori_per_step = np.broadcast_to(
        target_orientations[None, ...], (T, N)
    ).copy()

    return {
        "pixels": pixels,
        "state": state_flat,
        "action": actions,
        "positions": positions,
        "orientations": orientations,
        "target_positions": target_pos_per_step,
        "target_orientations": target_ori_per_step,
        "_block_block_contact_total": block_block_contact_total,
        "_block_pusher_contact_total": block_pusher_contact_total,
    }


def main() -> None:
    args = _parse_args()
    out_path = Path(
        args.output_path or _default_output_path(args.n_blocks)
    ).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    print(
        f"[gen] writing {args.n_episodes} eps x {args.frames_per_episode} "
        f"frames -> {out_path}"
    )
    print(
        f"[gen] n_blocks={args.n_blocks} box_size={args.box_size} "
        f"block_size={args.block_size} dt={args.dt}"
    )

    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    global_ptr = 0

    block_block_hist = np.zeros((args.n_episodes,), dtype=np.int64)
    block_pusher_hist = np.zeros((args.n_episodes,), dtype=np.int64)
    policy_choice_hist = {"random": 0, "single": 0, "multi": 0}

    with h5py.File(out_path, "w") as f:
        _init_datasets(f, args)

        for ep in range(args.n_episodes):
            ep_seed = args.seed * 1_000_003 + ep
            env = MultiBlockPushT(
                n_blocks=args.n_blocks,
                box_size=args.box_size,
                block_size=args.block_size,
                pusher_radius=args.pusher_radius,
                dt=args.dt,
                seed=ep_seed,
            )
            ep_rng = np.random.default_rng(ep_seed + 7919)

            # Sample policy type per the locked 40/40/20 mix.
            r = float(rng.uniform(0.0, 1.0))
            if r < 0.40:
                data = _rollout_random(env, args.frames_per_episode, ep_rng)
                policy_choice_hist["random"] += 1
            elif r < 0.80:
                data = _rollout_scripted_single(env, args.frames_per_episode, ep_rng)
                policy_choice_hist["single"] += 1
            else:
                data = _rollout_scripted_multi(env, args.frames_per_episode, ep_rng)
                policy_choice_hist["multi"] += 1

            T = args.frames_per_episode

            _append(f["pixels"], data["pixels"])
            _append(f["state"], data["state"])
            _append(f["action"], data["action"])
            _append(f["positions"], data["positions"])
            _append(f["orientations"], data["orientations"])
            _append(f["target_positions"], data["target_positions"])
            _append(f["target_orientations"], data["target_orientations"])
            _append(f["step_idx"], np.arange(T, dtype=np.int64))
            _append(f["ep_idx"], np.full((T,), ep, dtype=np.int32))

            ep_offset_ds = f["ep_offset"]
            ep_len_ds = f["ep_len"]
            ep_offset_ds.resize(ep + 1, axis=0)
            ep_len_ds.resize(ep + 1, axis=0)
            ep_offset_ds[ep] = global_ptr
            ep_len_ds[ep] = T
            global_ptr += T

            block_block_hist[ep] = data["_block_block_contact_total"]
            block_pusher_hist[ep] = data["_block_pusher_contact_total"]

            if (ep + 1) % args.log_every == 0 or ep + 1 == args.n_episodes:
                dt = time.perf_counter() - t0
                eps_per_s = (ep + 1) / dt
                eta = (args.n_episodes - ep - 1) / max(eps_per_s, 1e-9)
                print(
                    f"[gen] ep {ep + 1}/{args.n_episodes}  "
                    f"rate={eps_per_s:.2f} eps/s  "
                    f"eta={eta / 60:.1f} min  "
                    f"rows={global_ptr}"
                )

    elapsed = time.perf_counter() - t0
    size_mb = out_path.stat().st_size / 1e6
    print(
        f"[gen] done in {elapsed:.1f}s -- {global_ptr} rows, "
        f"{size_mb:.1f} MB at {out_path}"
    )

    # Contact-count histogram + sanity check.
    eps_with_bb = int(np.sum(block_block_hist > 0))
    pct_with_bb = 100.0 * eps_with_bb / max(args.n_episodes, 1)
    print(
        f"[gen] policy mix: random={policy_choice_hist['random']} "
        f"single={policy_choice_hist['single']} multi={policy_choice_hist['multi']}"
    )
    print(
        f"[gen] block-block contact: episodes with >=1 contact = "
        f"{eps_with_bb}/{args.n_episodes} ({pct_with_bb:.1f}%) "
        f"mean per ep = {block_block_hist.mean():.2f}  "
        f"max = {int(block_block_hist.max())}"
    )
    print(
        f"[gen] pusher-block contact: mean per ep = "
        f"{block_pusher_hist.mean():.2f}  max = {int(block_pusher_hist.max())}"
    )

    # Per-spec: <5% of episodes with block-block contact = WARNING.
    if pct_with_bb < 5.0:
        print(
            "[gen] WARNING: <5% of episodes contained block-block "
            "interaction. Per the Phase 2 spec this dataset should be "
            "regenerated."
        )


if __name__ == "__main__":
    main()
