"""Reduced-order engine model: throttle in, 8 coupled sensor channels out."""

import numpy as np

from contracts import Run

# Upper limit per channel. Set from the physics before looking at any runs.
REDLINES: dict[str, float] = {}


def make_throttle(seed: int) -> np.ndarray:
    """Duty cycle: startup ramp, steady hold, optional mid-run steps, shutdown. Shape (N_STEPS,)."""
    raise NotImplementedError


def simulate(throttle: np.ndarray, seed: int, noise: bool = True) -> Run:
    """Integrate the engine model over one run with explicit Euler at DT_S.

    The seed sets run-to-run variation (small parameter differences, per-channel offsets)
    and the sensor noise. noise=False turns off only the sensor noise, so tests see clean
    physics while the seed still sets the engine's parameters.

    Euler is stable only for DT_S / tau < 2 and accurate when that ratio is small, so every
    time constant stays above 0.05 s.
    """
    raise NotImplementedError
