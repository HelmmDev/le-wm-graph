"""2D bouncing-balls simulation environment.

Used as a Phase 1 MVP world-model benchmark: a domain where graph structure
is unambiguously load-bearing (predicting next state requires knowing
collisions). Pymunk handles physics; we render simple per-ball colored disks
into a fixed-size RGB frame.

The environment has:
  - elastic ball-ball and ball-wall collisions (restitution=1.0)
  - no friction, no gravity
  - rectangular box of size `box_size x box_size` pixels (= world units)
  - up to 10 visually-distinct ball colors (the spec asks for 5; we keep a
    longer table so n_balls > 5 still renders cleanly)

State dict format:
  {
    'positions':  (N, 2) float32  -- in pixel/world coords
    'velocities': (N, 2) float32  -- in pixel/world units per second
    'radii':      (N,)   float32  -- in pixels
  }
"""
from __future__ import annotations

import numpy as np
import pymunk

# Ten visually-distinct colors (R, G, B), uint8. First five are the
# "primary" set referenced by the spec; the rest cover larger n_balls.
_BALL_COLORS = np.array(
    [
        (220, 50, 47),    # red
        (38, 139, 210),   # blue
        (133, 153, 0),    # green
        (181, 137, 0),    # yellow/amber
        (211, 54, 130),   # magenta
        (42, 161, 152),   # cyan/teal
        (203, 75, 22),    # orange
        (108, 113, 196),  # violet
        (238, 232, 213),  # cream
        (147, 161, 161),  # gray
    ],
    dtype=np.uint8,
)


class BouncingBalls:
    """Pymunk-driven 2D bouncing-balls environment.

    Args:
        n_balls: Number of balls in the scene.
        box_size: Width/height of the (square) box, in world units == pixels.
        ball_radius: Radius of every ball (uniform), in pixels.
        dt: Physics timestep (seconds).
        seed: Optional seed for reproducible initial conditions.
    """

    def __init__(
        self,
        n_balls: int = 5,
        box_size: int = 64,
        ball_radius: float = 4.0,
        dt: float = 1.0 / 30.0,
        seed: int | None = None,
    ) -> None:
        self.n_balls = int(n_balls)
        self.box_size = int(box_size)
        self.ball_radius = float(ball_radius)
        self.dt = float(dt)
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        self._space: pymunk.Space | None = None
        self._bodies: list[pymunk.Body] = []
        self._shapes: list[pymunk.Circle] = []

        # Cache colors for rendering (one per ball).
        self._colors = _BALL_COLORS[
            np.arange(self.n_balls) % len(_BALL_COLORS)
        ]

        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> dict:
        """Reset the environment, returning the initial state dict."""
        # Re-seed only if a deterministic seed was given; otherwise keep
        # advancing the RNG so successive resets give different episodes.
        if self._seed is not None:
            self._rng = np.random.default_rng(self._seed)

        self._space = pymunk.Space()
        self._space.gravity = (0.0, 0.0)
        # Slight bias to keep contacts elastic; we override per-shape below.
        self._space.damping = 1.0

        self._add_walls()
        self._bodies, self._shapes = [], []

        # Sample non-overlapping initial positions via rejection sampling.
        positions = self._sample_positions()
        # Random initial velocities -- magnitude scaled so a typical episode
        # has several collisions in 64 frames at dt=1/30.
        speed = max(self.box_size * 0.5, 8.0)  # world units / sec
        angles = self._rng.uniform(0.0, 2.0 * np.pi, size=self.n_balls)
        velocities = np.stack(
            [np.cos(angles), np.sin(angles)], axis=1
        ).astype(np.float32) * speed

        for i in range(self.n_balls):
            body, shape = self._make_ball(positions[i], velocities[i])
            self._space.add(body, shape)
            self._bodies.append(body)
            self._shapes.append(shape)

        return self._state_dict()

    def step(self) -> tuple[np.ndarray, dict]:
        """Advance one physics step. Returns (frame, state_dict)."""
        assert self._space is not None, 'call reset() first'
        self._space.step(self.dt)
        return self.render(), self._state_dict()

    def render(self) -> np.ndarray:
        """Rasterize current state to an (H, W, 3) uint8 frame.

        Uses a simple analytic disk fill -- no external dependency on
        pygame/PIL surfaces, so this runs headless on the GPU box. For
        box_size=64, ball_radius=4, n_balls=5, this is sub-millisecond.
        """
        H = W = self.box_size
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        # Pre-compute pixel coordinate grid once per call. For 64x64 this
        # is 4096 floats; cheap enough not to bother caching.
        ys = np.arange(H, dtype=np.float32)[:, None]
        xs = np.arange(W, dtype=np.float32)[None, :]

        for i, body in enumerate(self._bodies):
            cx, cy = float(body.position.x), float(body.position.y)
            # Pymunk world -> image: invert y so y=0 is top of frame.
            cy_img = (H - 1) - cy
            # Anti-aliasing-free disk. `<= r*r` so center pixel is always lit.
            mask = (xs - cx) ** 2 + (ys - cy_img) ** 2 <= self.ball_radius ** 2
            frame[mask] = self._colors[i]

        return frame

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _state_dict(self) -> dict:
        positions = np.array(
            [(b.position.x, b.position.y) for b in self._bodies],
            dtype=np.float32,
        )
        velocities = np.array(
            [(b.velocity.x, b.velocity.y) for b in self._bodies],
            dtype=np.float32,
        )
        radii = np.full(
            (self.n_balls,), self.ball_radius, dtype=np.float32
        )
        return {
            'positions': positions,
            'velocities': velocities,
            'radii': radii,
        }

    def _add_walls(self) -> None:
        """Static walls around the box. World coords: x in [0, box_size]."""
        s = self.box_size
        static_body = self._space.static_body
        # Slightly inset so balls visually stay inside the frame.
        walls = [
            pymunk.Segment(static_body, (0, 0), (s, 0), 0.0),
            pymunk.Segment(static_body, (s, 0), (s, s), 0.0),
            pymunk.Segment(static_body, (s, s), (0, s), 0.0),
            pymunk.Segment(static_body, (0, s), (0, 0), 0.0),
        ]
        for w in walls:
            w.elasticity = 1.0
            w.friction = 0.0
            self._space.add(w)

    def _sample_positions(self) -> np.ndarray:
        """Rejection-sample non-overlapping ball centers."""
        margin = self.ball_radius + 1.0
        lo, hi = margin, self.box_size - margin
        min_sep = 2.0 * self.ball_radius + 1.0
        positions: list[np.ndarray] = []
        max_tries = 5000
        for _ in range(max_tries):
            if len(positions) == self.n_balls:
                break
            cand = self._rng.uniform(lo, hi, size=2).astype(np.float32)
            ok = True
            for p in positions:
                if np.linalg.norm(cand - p) < min_sep:
                    ok = False
                    break
            if ok:
                positions.append(cand)
        if len(positions) < self.n_balls:
            raise RuntimeError(
                f'Could not place {self.n_balls} balls in '
                f'{self.box_size}x{self.box_size} box without overlap'
            )
        return np.stack(positions, axis=0)

    def _make_ball(
        self, pos: np.ndarray, vel: np.ndarray
    ) -> tuple[pymunk.Body, pymunk.Circle]:
        mass = 1.0
        moment = pymunk.moment_for_circle(mass, 0.0, self.ball_radius)
        body = pymunk.Body(mass, moment)
        body.position = (float(pos[0]), float(pos[1]))
        body.velocity = (float(vel[0]), float(vel[1]))
        shape = pymunk.Circle(body, self.ball_radius)
        shape.elasticity = 1.0
        shape.friction = 0.0
        return body, shape
