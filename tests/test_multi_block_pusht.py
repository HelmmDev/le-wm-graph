"""Sanity tests for envs.multi_block_pusht.MultiBlockPushT.

Runnable either via ``pytest tests/test_multi_block_pusht.py`` or as a plain
script. CPU-only — does not touch the GPU and does NOT generate the full
dataset.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from envs.multi_block_pusht import MultiBlockPushT


def test_constructs_with_each_n() -> None:
    for n in (2, 3, 5, 8):
        env = MultiBlockPushT(n_blocks=n, seed=0)
        st = env.reset()
        assert st["positions"].shape == (n, 2), f"n={n}"
        assert st["orientations"].shape == (n,), f"n={n}"
        assert st["target_positions"].shape == (n, 2), f"n={n}"
        assert st["target_orientations"].shape == (n,), f"n={n}"
        frame = env.render()
        assert frame.shape == (224, 224, 3), f"n={n}"
        assert frame.dtype == np.uint8


def test_runs_100_steps_n5() -> None:
    env = MultiBlockPushT(n_blocks=5, seed=0)
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(100):
        a = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        frame, st, r, done, info = env.step(a)
        assert frame.shape == (224, 224, 3)
        assert st["positions"].shape == (5, 2)
        # No crashes, finite state.
        assert np.all(np.isfinite(st["positions"]))
        assert np.all(np.isfinite(st["orientations"]))


def test_block_block_contact_n5() -> None:
    """Over 200 steps in n=5, expect at least one block-block contact when
    the pusher is actively driven toward block 0 with target = block 1.

    A pure-random policy is too stochastic to reliably bump blocks together
    in 200 steps (verified empirically: ~10% of seeds produce a contact);
    we instead drive the pusher with a deterministic-ish policy that
    actually probes the contact path. This still exercises the contact
    counter and the physics — what we really need to test.
    """
    env = MultiBlockPushT(n_blocks=5, seed=0)
    state = env.reset()
    saw_contact = False
    saw_pusher_block = False
    for step in range(200):
        # Drive pusher toward block 0, with block 0 -> block 1 as the push axis.
        b0 = state["positions"][0]
        b1 = state["positions"][1]
        push_axis = b1 - b0
        push_axis = push_axis / (np.linalg.norm(push_axis) + 1e-6)
        approach = b0 - push_axis * 25.0
        pusher = state["pusher_position"]
        to_app = approach - pusher
        d_app = np.linalg.norm(to_app)
        to_b0 = b0 - pusher
        d_b0 = np.linalg.norm(to_b0)
        behind_score = float(np.dot(push_axis, to_b0 / (d_b0 + 1e-6)))
        if behind_score > 0.5 and d_b0 < 35.0:
            action = push_axis
        else:
            action = to_app / (d_app + 1e-6)
        action = np.clip(action.astype(np.float32), -1.0, 1.0)
        _, state, _, _, info = env.step(action)
        if info["block_block_contact_count"] > 0:
            saw_contact = True
        if info["block_pusher_contact_count"] > 0:
            saw_pusher_block = True
        if saw_contact:
            break
    assert saw_pusher_block, (
        "Pusher never made contact with a block in 200 steps — physics or "
        "scripted policy is misconfigured."
    )
    assert saw_contact, (
        "No block-block contact observed in 200 scripted-pusher steps with "
        "n=5 driving block 0 toward block 1; physics or contact-counter is "
        "misconfigured."
    )


def test_frames_have_variance() -> None:
    env = MultiBlockPushT(n_blocks=3, seed=1)
    env.reset()
    rng = np.random.default_rng(1)
    frames = []
    for _ in range(20):
        a = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        f, _, _, _, _ = env.step(a)
        frames.append(f)
    stack = np.stack(frames, axis=0).astype(np.float32)
    assert float(stack.var()) > 0.0, "rendered frames are constant"
    assert float(frames[0].var()) > 0.0, "first frame is degenerate"


def test_tiny_dataset_gen() -> None:
    """Run the dataset generator with n_episodes=5, frames=16 in a tempdir
    and verify the resulting HDF5 has the expected schema."""
    import h5py
    import subprocess

    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "tiny.h5"
        cmd = [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "generate_multi_block_pusht_dataset.py"),
            "--n_blocks", "3",
            "--n_episodes", "5",
            "--frames_per_episode", "16",
            "--output_path", str(out_path),
            "--seed", "42",
            "--log_every", "1",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        assert proc.returncode == 0, (
            f"generator failed: stdout={proc.stdout}\nstderr={proc.stderr}"
        )
        assert out_path.exists(), f"expected {out_path} to exist"

        with h5py.File(out_path, "r") as f:
            for k in [
                "pixels", "state", "action",
                "positions", "orientations",
                "target_positions", "target_orientations",
                "step_idx", "ep_idx", "ep_offset", "ep_len",
            ]:
                assert k in f, f"missing key {k}"
            T = 5 * 16
            N = 3
            assert f["pixels"].shape == (T, 224, 224, 3)
            assert f["state"].shape == (T, N * 6 + 4)
            assert f["action"].shape == (T, 2)
            assert f["positions"].shape == (T, N, 2)
            assert f["orientations"].shape == (T, N)
            assert f["target_positions"].shape == (T, N, 2)
            assert f["target_orientations"].shape == (T, N)
            assert f["ep_offset"].shape == (5,)
            assert f["ep_len"].shape == (5,)
            assert int(f["ep_len"][0]) == 16


def _run_all() -> None:
    tests = [
        test_constructs_with_each_n,
        test_runs_100_steps_n5,
        test_block_block_contact_n5,
        test_frames_have_variance,
        test_tiny_dataset_gen,
    ]
    failed = []
    for t in tests:
        print(f"-- {t.__name__} ... ", end="", flush=True)
        t0 = time.perf_counter()
        try:
            t()
            print(f"OK ({time.perf_counter() - t0:.2f}s)")
        except Exception as e:
            print(f"FAIL ({type(e).__name__}: {e})")
            failed.append(t.__name__)
    if failed:
        print(f"\n{len(failed)} FAILED: {failed}")
        sys.exit(1)
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
