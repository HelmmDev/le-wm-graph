"""Generate a bouncing-balls HDF5 dataset compatible with stable-worldmodel.

Schema matches stable_worldmodel.data.HDF5Dataset (see also
~/.stable-wm/pusht_expert_train.h5):

  pixels    : (T_total, H, W, 3) uint8  Blosc-LZ4 chunks=(100, H, W, 3)
  state     : (T_total, N*4)    float32  -- flattened [px, py, vx, vy] per ball
  positions : (T_total, N, 2)   float32  -- ground-truth positions
  velocities: (T_total, N, 2)   float32  -- ground-truth velocities
  radii     : (T_total, N)      float32  -- ball radii (constant per episode)
  step_idx  : (T_total,)        int64    -- step within episode
  ep_idx    : (T_total,)        int32    -- episode index per row
  ep_offset : (n_episodes,)     int64    -- starting row of each episode
  ep_len    : (n_episodes,)     int32    -- length of each episode

The `state` key is provided in flat form for compatibility with downstream
code that expects a 2D (T, D) state array (matching pusht's convention).
The structured `positions`/`velocities`/`radii` columns are the canonical
ground truth for graph encoders.

Single-process generation by design: Phase 0 dataloaders are CPU-bound and
we don't want to compete with them.
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

# Allow running this script from `scripts/` without installing the repo.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from envs.bouncing_balls import BouncingBalls  # noqa: E402


def _default_output_path() -> str:
    return str(
        Path.home() / '.stable-wm' / 'bouncing_balls_train.h5'
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n_episodes', type=int, default=10_000)
    p.add_argument('--frames_per_episode', type=int, default=64)
    p.add_argument('--n_balls', type=int, default=5)
    p.add_argument('--box_size', type=int, default=64)
    p.add_argument('--ball_radius', type=float, default=4.0)
    p.add_argument('--dt', type=float, default=1.0 / 30.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument(
        '--output_path',
        type=str,
        default=_default_output_path(),
        help='Output .h5 path (default: ~/.stable-wm/bouncing_balls_train.h5)',
    )
    p.add_argument(
        '--log_every',
        type=int,
        default=100,
        help='Print progress every N episodes.',
    )
    return p.parse_args()


def _make_blosc_compression():
    """Match the compressor used by stable_worldmodel for `pixels`."""
    import hdf5plugin

    return hdf5plugin.Blosc(
        cname='lz4', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE
    )


def _init_datasets(
    f: h5py.File,
    box_size: int,
    n_balls: int,
) -> None:
    """Create resizable datasets matching stable_worldmodel conventions."""
    blosc = _make_blosc_compression()

    f.create_dataset(
        'pixels',
        shape=(0, box_size, box_size, 3),
        maxshape=(None, box_size, box_size, 3),
        dtype=np.uint8,
        chunks=(100, box_size, box_size, 3),
        compression=blosc,
    )
    # Flat state: [px, py, vx, vy] per ball, concatenated.
    f.create_dataset(
        'state',
        shape=(0, n_balls * 4),
        maxshape=(None, n_balls * 4),
        dtype=np.float32,
        chunks=(1000, n_balls * 4),
    )
    f.create_dataset(
        'positions',
        shape=(0, n_balls, 2),
        maxshape=(None, n_balls, 2),
        dtype=np.float32,
        chunks=(1000, n_balls, 2),
    )
    f.create_dataset(
        'velocities',
        shape=(0, n_balls, 2),
        maxshape=(None, n_balls, 2),
        dtype=np.float32,
        chunks=(1000, n_balls, 2),
    )
    f.create_dataset(
        'radii',
        shape=(0, n_balls),
        maxshape=(None, n_balls),
        dtype=np.float32,
        chunks=(1000, n_balls),
    )
    f.create_dataset(
        'step_idx',
        shape=(0,),
        maxshape=(None,),
        dtype=np.int64,
        chunks=(1000,),
    )
    f.create_dataset(
        'ep_idx',
        shape=(0,),
        maxshape=(None,),
        dtype=np.int32,
        chunks=(1000,),
    )
    f.create_dataset(
        'ep_offset',
        shape=(0,),
        maxshape=(None,),
        dtype=np.int64,
    )
    f.create_dataset(
        'ep_len',
        shape=(0,),
        maxshape=(None,),
        dtype=np.int32,
    )


def _append(ds: h5py.Dataset, arr: np.ndarray) -> None:
    cur = ds.shape[0]
    ds.resize(cur + arr.shape[0], axis=0)
    ds[cur:] = arr


def _rollout(
    env: BouncingBalls, frames_per_episode: int
) -> dict[str, np.ndarray]:
    """Run one episode, capturing (frame, state) at each of T steps.

    The first row of each episode is the post-reset state (rendered),
    matching how stable-worldmodel records (o_t, a_t) tuples.
    """
    T = frames_per_episode
    box = env.box_size
    N = env.n_balls

    pixels = np.empty((T, box, box, 3), dtype=np.uint8)
    positions = np.empty((T, N, 2), dtype=np.float32)
    velocities = np.empty((T, N, 2), dtype=np.float32)
    radii = np.empty((T, N), dtype=np.float32)

    # t = 0: initial state from reset.
    state0 = env.reset()
    pixels[0] = env.render()
    positions[0] = state0['positions']
    velocities[0] = state0['velocities']
    radii[0] = state0['radii']

    # t = 1 .. T-1: advance physics.
    for t in range(1, T):
        frame, st = env.step()
        pixels[t] = frame
        positions[t] = st['positions']
        velocities[t] = st['velocities']
        radii[t] = st['radii']

    state_flat = np.concatenate(
        [positions.reshape(T, N * 2), velocities.reshape(T, N * 2)], axis=1
    ).astype(np.float32)
    return {
        'pixels': pixels,
        'state': state_flat,
        'positions': positions,
        'velocities': velocities,
        'radii': radii,
    }


def main() -> None:
    args = _parse_args()
    out_path = Path(args.output_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    print(
        f'[gen] writing {args.n_episodes} eps x {args.frames_per_episode} '
        f'frames -> {out_path}'
    )
    print(
        f'[gen] n_balls={args.n_balls} box_size={args.box_size} '
        f'ball_radius={args.ball_radius} dt={args.dt}'
    )

    t0 = time.perf_counter()
    global_ptr = 0

    with h5py.File(out_path, 'w') as f:
        _init_datasets(f, args.box_size, args.n_balls)

        for ep in range(args.n_episodes):
            # Per-episode seed so every episode is reproducible from
            # (--seed, ep_idx) alone.
            ep_seed = args.seed * 1_000_003 + ep
            env = BouncingBalls(
                n_balls=args.n_balls,
                box_size=args.box_size,
                ball_radius=args.ball_radius,
                dt=args.dt,
                seed=ep_seed,
            )
            data = _rollout(env, args.frames_per_episode)
            T = args.frames_per_episode

            _append(f['pixels'], data['pixels'])
            _append(f['state'], data['state'])
            _append(f['positions'], data['positions'])
            _append(f['velocities'], data['velocities'])
            _append(f['radii'], data['radii'])
            _append(
                f['step_idx'],
                np.arange(T, dtype=np.int64),
            )
            _append(
                f['ep_idx'],
                np.full((T,), ep, dtype=np.int32),
            )

            ep_offset_ds = f['ep_offset']
            ep_len_ds = f['ep_len']
            ep_offset_ds.resize(ep + 1, axis=0)
            ep_len_ds.resize(ep + 1, axis=0)
            ep_offset_ds[ep] = global_ptr
            ep_len_ds[ep] = T
            global_ptr += T

            if (ep + 1) % args.log_every == 0 or ep + 1 == args.n_episodes:
                dt = time.perf_counter() - t0
                eps_per_s = (ep + 1) / dt
                eta = (args.n_episodes - ep - 1) / max(eps_per_s, 1e-9)
                print(
                    f'[gen] ep {ep + 1}/{args.n_episodes}  '
                    f'rate={eps_per_s:.2f} eps/s  eta={eta / 60:.1f} min  '
                    f'rows={global_ptr}'
                )

    elapsed = time.perf_counter() - t0
    size_mb = out_path.stat().st_size / 1e6
    print(
        f'[gen] done in {elapsed:.1f}s -- {global_ptr} rows, '
        f'{size_mb:.1f} MB at {out_path}'
    )


if __name__ == '__main__':
    main()
