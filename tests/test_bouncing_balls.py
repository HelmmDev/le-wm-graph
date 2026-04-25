"""Sanity tests for envs/bouncing_balls.BouncingBalls.

Runnable either via `pytest tests/test_bouncing_balls.py` or as a plain
script (`python tests/test_bouncing_balls.py`) so it works in barebones
venvs without pytest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from envs.bouncing_balls import BouncingBalls


def test_constructs_with_defaults() -> None:
    env = BouncingBalls()
    state = env.reset()
    assert state['positions'].shape == (5, 2)
    assert state['velocities'].shape == (5, 2)
    assert state['radii'].shape == (5,)
    assert state['positions'].dtype == np.float32
    frame = env.render()
    assert frame.shape == (64, 64, 3)
    assert frame.dtype == np.uint8


def test_runs_100_steps() -> None:
    env = BouncingBalls(seed=0)
    env.reset()
    for _ in range(100):
        frame, state = env.step()
        assert frame.shape == (64, 64, 3)
        assert state['positions'].shape == (5, 2)
        # Balls should never escape the box (modulo float epsilon).
        ps = state['positions']
        assert np.all(ps >= -1.0) and np.all(ps <= 65.0), (
            f'ball escaped box: positions={ps}'
        )


def test_collision_changes_velocities() -> None:
    """Over 200 steps, at least one ball-ball collision must occur.

    We detect collisions by watching for a velocity change that is NOT
    explained by a wall bounce. A robust proxy: if all velocity magnitudes
    stay constant per ball over 200 steps, balls aren't interacting (in an
    elastic system with only walls, |v| per ball is preserved). With
    ball-ball collisions, the per-ball speeds will change.
    """
    env = BouncingBalls(n_balls=5, box_size=64, ball_radius=4, seed=0)
    state0 = env.reset()
    speeds0 = np.linalg.norm(state0['velocities'], axis=1)

    saw_speed_change = False
    for _ in range(200):
        _, st = env.step()
        speeds = np.linalg.norm(st['velocities'], axis=1)
        # Wall bounces preserve per-ball speed; only ball-ball collisions
        # redistribute speed between balls.
        if not np.allclose(np.sort(speeds), np.sort(speeds0), atol=1e-3):
            saw_speed_change = True
            break

    assert saw_speed_change, (
        'No ball-ball collision detected in 200 steps; balls are not '
        'interacting (initial speeds preserved per-ball under wall-only '
        'dynamics).'
    )


def test_frames_have_variance() -> None:
    env = BouncingBalls(seed=1)
    env.reset()
    frames = []
    for _ in range(20):
        f, _ = env.step()
        frames.append(f)
    stack = np.stack(frames, axis=0).astype(np.float32)
    # Per-pixel variance across time should be >0 somewhere.
    assert float(stack.var()) > 0.0, 'rendered frames are constant'
    # Single frames should also be non-degenerate (not all-zero / all-same).
    assert float(frames[0].var()) > 0.0, 'first frame is degenerate'


def _run_all() -> None:
    tests = [
        test_constructs_with_defaults,
        test_runs_100_steps,
        test_collision_changes_velocities,
        test_frames_have_variance,
    ]
    for t in tests:
        print(f'-- {t.__name__} ... ', end='', flush=True)
        t()
        print('ok')
    print('all tests passed')


if __name__ == '__main__':
    _run_all()
