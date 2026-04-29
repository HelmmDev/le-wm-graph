"""Multi-block PushT environment for Phase 2 of the LeWM graph-encoder study.

A custom Pymunk implementation (intentionally NOT dm_control's PushT, which is
hard-coded for a single T-shaped block). The agent is a small circular pusher
controlled by a 2D end-effector velocity command. The scene contains
``n_blocks`` rectangular blocks, each with its own (x, y, theta) target pose.
The episode succeeds when every block is within tolerance of its target.

State dict format (returned by ``reset()`` / ``step()``):
    {
        'positions':           (N, 2) float32  -- pixel-space block centers
        'orientations':        (N,)    float32  -- radians
        'velocities':          (N, 2) float32  -- linear, px/sec
        'angular_velocities':  (N,)   float32  -- rad/sec
        'pusher_position':     (2,)   float32
        'pusher_velocity':     (2,)   float32
        'target_positions':    (N, 2) float32
        'target_orientations': (N,)    float32
    }

Headless rendering is implemented in pure NumPy (filled rectangles + circles)
so the env runs on CPU-only or remote GPU boxes without any SDL/pygame
windowing layer.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pymunk

# Eight visually-distinct colors (R, G, B), uint8 — one per block at max N=8.
_BLOCK_COLORS = np.array(
    [
        (220, 50, 47),    # red
        (38, 139, 210),   # blue
        (133, 153, 0),    # green
        (181, 137, 0),    # amber
        (211, 54, 130),   # magenta
        (42, 161, 152),   # teal
        (203, 75, 22),    # orange
        (108, 113, 196),  # violet
    ],
    dtype=np.uint8,
)
_PUSHER_COLOR = np.array((40, 40, 40), dtype=np.uint8)
_BG_COLOR = np.array((255, 255, 255), dtype=np.uint8)

# Collision types (used to count block-block contacts in info).
_CT_BLOCK = 1
_CT_PUSHER = 2
_CT_WALL = 3


class MultiBlockPushT:
    """Pymunk-driven multi-block PushT environment.

    Args:
        n_blocks: Number of blocks (recommended 2, 3, 5, or 8).
        box_size: Side length of the (square) workspace, in world units == pixels.
        block_size: Side length of each square block in pixels (default 30).
        pusher_radius: Radius of the pusher circle in pixels (default 10).
        dt: Physics timestep (seconds). Default 1/30.
        max_steps: Default cap on episode length used by step()'s done flag.
        translation_tol: Per-block target translation tolerance (pixels).
            Default = 0.05 * block_size, per the spec.
        rotation_tol: Per-block target rotation tolerance (radians).
            Default = 15 degrees.
        seed: Optional seed for reproducible scenes.
        pusher_speed: World-units / sec applied per unit of action magnitude.
    """

    def __init__(
        self,
        n_blocks: int,
        box_size: int = 224,
        block_size: float = 30.0,
        pusher_radius: float = 10.0,
        dt: float = 1.0 / 30.0,
        max_steps: int = 256,
        translation_tol: Optional[float] = None,
        rotation_tol: float = float(np.deg2rad(15.0)),
        seed: Optional[int] = None,
        pusher_speed: float = 80.0,
    ) -> None:
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")
        self.n_blocks = int(n_blocks)
        self.box_size = int(box_size)
        self.block_size = float(block_size)
        self.pusher_radius = float(pusher_radius)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.translation_tol = (
            float(translation_tol)
            if translation_tol is not None
            else 0.05 * self.block_size
        )
        self.rotation_tol = float(rotation_tol)
        self.pusher_speed = float(pusher_speed)
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        self._space: Optional[pymunk.Space] = None
        self._block_bodies: list[pymunk.Body] = []
        self._block_shapes: list[pymunk.Poly] = []
        self._pusher_body: Optional[pymunk.Body] = None
        self._pusher_shape: Optional[pymunk.Circle] = None

        self._target_positions: np.ndarray = np.zeros((self.n_blocks, 2), dtype=np.float32)
        self._target_orientations: np.ndarray = np.zeros((self.n_blocks,), dtype=np.float32)

        # Per-step counters reset in step().
        self._block_block_contacts: int = 0
        self._block_pusher_contacts: int = 0
        self._step_count: int = 0

        # Color per block.
        self._colors = _BLOCK_COLORS[
            np.arange(self.n_blocks) % len(_BLOCK_COLORS)
        ]

        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> dict:
        """Reset the environment, sampling new initial and target poses."""
        if self._seed is not None:
            self._rng = np.random.default_rng(self._seed)

        self._space = pymunk.Space()
        self._space.gravity = (0.0, 0.0)
        # Damping makes blocks slide-and-stop rather than glide forever.
        # 0.85 chosen so a pushed block coasts ~10-20 px before stopping —
        # enough to plausibly bump into another block but not glide forever.
        self._space.damping = 0.85

        self._add_walls()
        self._block_bodies, self._block_shapes = [], []

        init_positions = self._sample_non_overlapping_positions()
        init_orientations = self._rng.uniform(
            -np.pi, np.pi, size=self.n_blocks
        ).astype(np.float32)

        for i in range(self.n_blocks):
            body, shape = self._make_block(
                init_positions[i], float(init_orientations[i])
            )
            self._space.add(body, shape)
            self._block_bodies.append(body)
            self._block_shapes.append(shape)

        # Pusher: small circle, spawned at a random spot also avoiding blocks.
        pusher_pos = self._sample_pusher_spawn(init_positions)
        self._pusher_body, self._pusher_shape = self._make_pusher(pusher_pos)
        self._space.add(self._pusher_body, self._pusher_shape)

        # Target poses (independent rejection-sample so targets don't overlap
        # each other; they CAN overlap initial blocks, that's fine).
        target_positions = self._sample_non_overlapping_positions()
        target_orientations = self._rng.uniform(
            -np.pi, np.pi, size=self.n_blocks
        ).astype(np.float32)
        self._target_positions = target_positions.astype(np.float32)
        self._target_orientations = target_orientations.astype(np.float32)

        # Collision-counting handlers (count contact begins per step).
        self._install_collision_handlers()

        self._block_block_contacts = 0
        self._block_pusher_contacts = 0
        self._step_count = 0

        return self._state_dict()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, dict, float, bool, dict]:
        """Advance physics one step.

        Args:
            action: (2,) array of (vx, vy) in [-1, 1]. Applied to the pusher
                as a velocity command (scaled by ``pusher_speed``).

        Returns:
            (frame, state_dict, reward, done, info)
        """
        assert self._space is not None, "call reset() first"
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] != 2:
            raise ValueError(f"action must be shape (2,), got {a.shape}")
        a = np.clip(a, -1.0, 1.0)

        # Velocity-command pusher: set velocity directly each step. Since the
        # pusher is a kinematic body, this propagates contacts cleanly.
        self._pusher_body.velocity = (
            float(a[0]) * self.pusher_speed,
            float(a[1]) * self.pusher_speed,
        )

        # Reset per-step contact counters before stepping.
        self._block_block_contacts = 0
        self._block_pusher_contacts = 0

        self._space.step(self.dt)
        self._step_count += 1

        frame = self.render()
        state = self._state_dict()

        # Per-block target distances (translation + rotation residuals).
        trans_dists, rot_dists = self._target_distances()
        all_within = bool(
            np.all(trans_dists <= self.translation_tol)
            and np.all(rot_dists <= self.rotation_tol)
        )
        done = all_within or (self._step_count >= self.max_steps)

        # Reward: dense, used only by scripted policies / debugging — sum of
        # negative translation distance (in block-size units) plus a small
        # rotation penalty. Training does not use this.
        reward = -float(
            np.sum(trans_dists) / self.block_size
            + 0.1 * np.sum(rot_dists)
        )

        info = {
            "translation_distances": trans_dists.astype(np.float32),
            "rotation_distances": rot_dists.astype(np.float32),
            "all_within_tol": all_within,
            "block_block_contact_count": int(self._block_block_contacts),
            "block_pusher_contact_count": int(self._block_pusher_contacts),
            "step": int(self._step_count),
        }
        return frame, state, reward, done, info

    def render(self) -> np.ndarray:
        """Headless rasterizer: white background, filled blocks, outline targets,
        and a dark pusher disk."""
        H = W = self.box_size
        frame = np.empty((H, W, 3), dtype=np.uint8)
        frame[:] = _BG_COLOR

        # Pre-compute pixel grid once.
        ys = np.arange(H, dtype=np.float32)[:, None]
        xs = np.arange(W, dtype=np.float32)[None, :]

        # 1) Outline targets first (drawn beneath blocks/pusher so they look
        #    like ghosts).
        for i in range(self.n_blocks):
            tcx, tcy = float(self._target_positions[i, 0]), float(self._target_positions[i, 1])
            ttheta = float(self._target_orientations[i])
            ghost = self._blend(_BG_COLOR, self._colors[i], 0.35)
            self._draw_rect_outline(
                frame, xs, ys, tcx, tcy, ttheta, ghost, stroke=1.5
            )

        # 2) Filled blocks.
        for i, body in enumerate(self._block_bodies):
            cx, cy = float(body.position.x), float(body.position.y)
            theta = float(body.angle)
            self._draw_rect_filled(frame, xs, ys, cx, cy, theta, self._colors[i])

        # 3) Pusher.
        pcx, pcy = float(self._pusher_body.position.x), float(self._pusher_body.position.y)
        pcy_img = (H - 1) - pcy
        pmask = (xs - pcx) ** 2 + (ys - pcy_img) ** 2 <= self.pusher_radius ** 2
        frame[pmask] = _PUSHER_COLOR

        return frame

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _blend(bg: np.ndarray, fg: np.ndarray, alpha: float) -> np.ndarray:
        """Alpha-blend fg over bg; returns uint8 (3,)."""
        out = (1.0 - alpha) * bg.astype(np.float32) + alpha * fg.astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _state_dict(self) -> dict:
        positions = np.array(
            [(b.position.x, b.position.y) for b in self._block_bodies],
            dtype=np.float32,
        )
        orientations = np.array(
            [b.angle for b in self._block_bodies], dtype=np.float32
        )
        velocities = np.array(
            [(b.velocity.x, b.velocity.y) for b in self._block_bodies],
            dtype=np.float32,
        )
        ang_velocities = np.array(
            [b.angular_velocity for b in self._block_bodies], dtype=np.float32
        )
        pusher_pos = np.array(
            (self._pusher_body.position.x, self._pusher_body.position.y),
            dtype=np.float32,
        )
        pusher_vel = np.array(
            (self._pusher_body.velocity.x, self._pusher_body.velocity.y),
            dtype=np.float32,
        )
        return {
            "positions": positions,
            "orientations": orientations,
            "velocities": velocities,
            "angular_velocities": ang_velocities,
            "pusher_position": pusher_pos,
            "pusher_velocity": pusher_vel,
            "target_positions": self._target_positions.copy(),
            "target_orientations": self._target_orientations.copy(),
        }

    def _target_distances(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-block (translation_dist, rotation_dist) arrays."""
        positions = np.array(
            [(b.position.x, b.position.y) for b in self._block_bodies],
            dtype=np.float32,
        )
        orientations = np.array(
            [b.angle for b in self._block_bodies], dtype=np.float32
        )
        trans = np.linalg.norm(
            positions - self._target_positions, axis=1
        ).astype(np.float32)
        # Wrap rotation residual to [-pi, pi] then take abs.
        delta = (orientations - self._target_orientations + np.pi) % (
            2.0 * np.pi
        ) - np.pi
        rot = np.abs(delta).astype(np.float32)
        return trans, rot

    def _add_walls(self) -> None:
        s = self.box_size
        static_body = self._space.static_body
        walls = [
            pymunk.Segment(static_body, (0, 0), (s, 0), 0.5),
            pymunk.Segment(static_body, (s, 0), (s, s), 0.5),
            pymunk.Segment(static_body, (s, s), (0, s), 0.5),
            pymunk.Segment(static_body, (0, s), (0, 0), 0.5),
        ]
        for w in walls:
            w.elasticity = 0.1
            w.friction = 0.5
            w.collision_type = _CT_WALL
            self._space.add(w)

    def _sample_non_overlapping_positions(self) -> np.ndarray:
        """Rejection-sample non-overlapping block centers (axis-aligned bbox
        inflated to be conservative; orientations are random so we just want
        some breathing room)."""
        # Use the diagonal of the (axis-aligned) block as a separation lower
        # bound — this avoids overlap regardless of orientation.
        margin = self.block_size * 0.75
        lo, hi = margin, self.box_size - margin
        if hi <= lo:
            raise RuntimeError(
                f"box_size={self.box_size} too small for block_size={self.block_size}"
            )
        min_sep = self.block_size * np.sqrt(2.0) + 2.0
        positions: list[np.ndarray] = []
        max_tries = 10_000
        for _ in range(max_tries):
            if len(positions) == self.n_blocks:
                break
            cand = self._rng.uniform(lo, hi, size=2).astype(np.float32)
            ok = True
            for p in positions:
                if np.linalg.norm(cand - p) < min_sep:
                    ok = False
                    break
            if ok:
                positions.append(cand)
        if len(positions) < self.n_blocks:
            raise RuntimeError(
                f"Could not place {self.n_blocks} blocks "
                f"(block_size={self.block_size}) in box of size {self.box_size}"
            )
        return np.stack(positions, axis=0)

    def _sample_pusher_spawn(self, block_positions: np.ndarray) -> np.ndarray:
        margin = self.pusher_radius + 2.0
        lo, hi = margin, self.box_size - margin
        sep_to_block = self.block_size * 0.75 + self.pusher_radius
        for _ in range(2000):
            cand = self._rng.uniform(lo, hi, size=2).astype(np.float32)
            d = np.linalg.norm(block_positions - cand, axis=1)
            if np.all(d > sep_to_block):
                return cand
        # If the workspace is dense, fall back to a corner.
        return np.array([margin, margin], dtype=np.float32)

    def _make_block(
        self, pos: np.ndarray, theta: float
    ) -> tuple[pymunk.Body, pymunk.Poly]:
        s = self.block_size
        mass = 1.0
        moment = pymunk.moment_for_box(mass, (s, s))
        body = pymunk.Body(mass, moment)
        body.position = (float(pos[0]), float(pos[1]))
        body.angle = float(theta)
        verts = [(-s / 2, -s / 2), (s / 2, -s / 2), (s / 2, s / 2), (-s / 2, s / 2)]
        shape = pymunk.Poly(body, verts)
        shape.elasticity = 0.1
        shape.friction = 0.6
        shape.collision_type = _CT_BLOCK
        return body, shape

    def _make_pusher(
        self, pos: np.ndarray
    ) -> tuple[pymunk.Body, pymunk.Circle]:
        # Kinematic body: we drive its velocity directly each step.
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = (float(pos[0]), float(pos[1]))
        body.velocity = (0.0, 0.0)
        shape = pymunk.Circle(body, self.pusher_radius)
        shape.elasticity = 0.1
        shape.friction = 0.7
        shape.collision_type = _CT_PUSHER
        return body, shape

    def _install_collision_handlers(self) -> None:
        """Register handlers that count contact begins.

        pymunk 6.x exposes ``Space.on_collision`` (callback API). pymunk
        5.x uses ``Space.add_collision_handler`` returning a handler. We
        support both by feature-detection.
        """
        space = self._space
        if hasattr(space, "on_collision"):
            # pymunk >= 6
            def _bb(arbiter, *_):
                self._block_block_contacts += 1
                return True

            def _bp(arbiter, *_):
                self._block_pusher_contacts += 1
                return True

            space.on_collision(_CT_BLOCK, _CT_BLOCK, begin=_bb)
            space.on_collision(_CT_BLOCK, _CT_PUSHER, begin=_bp)
        elif hasattr(space, "add_collision_handler"):
            # pymunk 5.x
            h_bb = space.add_collision_handler(_CT_BLOCK, _CT_BLOCK)

            def _bb(arbiter, space_, data):
                self._block_block_contacts += 1
                return True

            h_bb.begin = _bb

            h_bp = space.add_collision_handler(_CT_BLOCK, _CT_PUSHER)

            def _bp(arbiter, space_, data):
                self._block_pusher_contacts += 1
                return True

            h_bp.begin = _bp
        # else: fall through silently — counters stay at 0.

    # ------------------------------------------------------------------
    # Drawing helpers (NumPy)
    # ------------------------------------------------------------------
    def _rect_corners_world(
        self, cx: float, cy: float, theta: float
    ) -> np.ndarray:
        """Return (4, 2) array of rect corner positions in world (pre-flip) coords."""
        s = self.block_size / 2.0
        local = np.array(
            [(-s, -s), (s, -s), (s, s), (-s, s)], dtype=np.float32
        )
        c, sn = np.cos(theta), np.sin(theta)
        R = np.array([[c, -sn], [sn, c]], dtype=np.float32)
        return local @ R.T + np.array([cx, cy], dtype=np.float32)

    def _draw_rect_filled(
        self,
        frame: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        cx: float,
        cy: float,
        theta: float,
        color: np.ndarray,
    ) -> None:
        H = self.box_size
        # Rotate the pixel grid into the block's local frame and check the
        # axis-aligned half-extents. Frame y is flipped vs world y.
        cy_img = (H - 1) - cy
        # Rotation in image coords: world theta CCW becomes image theta CW
        # because of the y flip. Use -theta when rotating image-space points.
        c, sn = np.cos(-theta), np.sin(-theta)
        dx = xs - cx
        dy = ys - cy_img
        local_x = c * dx - sn * dy
        local_y = sn * dx + c * dy
        s = self.block_size / 2.0
        mask = (np.abs(local_x) <= s) & (np.abs(local_y) <= s)
        frame[mask] = color

    def _draw_rect_outline(
        self,
        frame: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        cx: float,
        cy: float,
        theta: float,
        color: np.ndarray,
        stroke: float = 1.5,
    ) -> None:
        H = self.box_size
        cy_img = (H - 1) - cy
        c, sn = np.cos(-theta), np.sin(-theta)
        dx = xs - cx
        dy = ys - cy_img
        local_x = c * dx - sn * dy
        local_y = sn * dx + c * dy
        s = self.block_size / 2.0
        # Outline = inside-outer-rect AND outside-inner-rect.
        outer = (np.abs(local_x) <= s) & (np.abs(local_y) <= s)
        inner = (
            (np.abs(local_x) <= (s - stroke))
            & (np.abs(local_y) <= (s - stroke))
        )
        ring = outer & ~inner
        frame[ring] = color
