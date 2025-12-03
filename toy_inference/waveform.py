"""Waveform models for toy gravitational-wave inference."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from pycbc.waveform import get_td_waveform


def gw_waveform(
    mass1: float,
    mass2: float,
    spin1z: float,
    spin2z: float,
    delta_t: float,
    f_lower: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a simple GW waveform using PyCBC's get_td_waveform.

    Returns
    -------
    t :
        1D array of time samples.
    h :
        1D array of plus polarization strain values.
    """
    apx = "IMRPhenomD"

    hp, hc = get_td_waveform(
        approximant=apx,
        mass1=mass1,
        mass2=mass2,
        spin1z=spin1z,
        spin2z=spin2z,
        delta_t=delta_t,
        f_lower=f_lower,
    )

    t = np.array(hp.sample_times)
    h = np.array(hp)

    return t, h


def sine_gaussian(
    t: np.ndarray,
    A: float,
    t0: float,
    f0: float,
    tau: float = 0.02,
    phi0: float = 0.0,
) -> np.ndarray:
    """Sine-Gaussian burst waveform (kept for reference / comparison)."""
    t = np.asarray(t)
    envelope = np.exp(-0.5 * ((t - t0) / tau) ** 2)
    phase = 2.0 * np.pi * f0 * (t - t0) + phi0
    h = A * envelope * np.cos(phase)
    return h
