"""Waveform models for toy gravitational-wave inference.

This module currently provides a simple sine-Gaussian burst model that is
used throughout the toy parameter estimation pipeline.
"""

from __future__ import annotations

import numpy as np


def sine_gaussian(
    t: np.ndarray,
    A: float,
    t0: float,
    f0: float,
    tau: float = 0.02,
    phi0: float = 0.0,
) -> np.ndarray:
    """Return a sine-Gaussian burst waveform.

    Parameters
    ----------
    t :
        1D array of time samples (seconds).
    A :
        Overall amplitude of the burst.
    t0 :
        Central time of the burst (seconds).
    f0 :
        Central frequency of the burst (Hz).
    tau :
        Width of the Gaussian envelope (seconds).  Roughly sets the
        duration of the signal around ``t0``.
    phi0 :
        Initial phase of the sinusoid (radians).

    Returns
    -------
    h :
        Array of the same shape as ``t`` containing the signal strain.
    """
    t = np.asarray(t)

    envelope = np.exp(-0.5 * ((t - t0) / tau) ** 2)
    phase = 2.0 * np.pi * f0 * (t - t0) + phi0
    h = A * envelope * np.cos(phase)
    return h