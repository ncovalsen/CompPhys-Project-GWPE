"""Likelihood and prior functions for the toy GW inference project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .waveform import gw_waveform
from .utils import mc_q_to_masses


@dataclass
class PriorBounds:
    """Uniform prior bounds for GW parameters.

    The prior is uniform inside each interval and zero outside.
    """

    mc: Tuple[float, float]
    q: Tuple[float, float]
    spin1z: Tuple[float, float]
    spin2z: Tuple[float, float]


DEFAULT_PRIOR = PriorBounds(
    mc=(5.0, 40.0),
    q=(1.0, 8.0),
    spin1z=(-0.99, 0.99),
    spin2z=(-0.99, 0.99),
)


def log_uniform_prior(theta: np.ndarray, bounds: PriorBounds = DEFAULT_PRIOR) -> float:
    """Log of a simple factorized uniform prior."""
    mc, q, spin1z, spin2z = theta

    if not (bounds.mc[0] < mc < bounds.mc[1]): return -np.inf
    if not (bounds.q[0] < q < bounds.q[1]): return -np.inf
    if not (bounds.spin1z[0] < spin1z < bounds.spin1z[1]): return -np.inf
    if not (bounds.spin2z[0] < spin2z < bounds.spin2z[1]): return -np.inf

    return 0.0


def inner(
    a: np.ndarray,
    b: np.ndarray,
    sigma: float,
    delta_t: float,
) -> float:
    """Noise-weighted inner product for white Gaussian noise.

    (a|b) = sum a(t) b(t) dt / sigma^2
    """
    a = np.asarray(a)
    b = np.asarray(b)
    return delta_t * np.sum(a * b) / (sigma**2)


def log_likelihood(
    theta: np.ndarray,
    d: np.ndarray,
    sigma: float,
    delta_t: float,
    f_lower: float,
) -> float:
    """Gaussian log-likelihood for the GW toy model."""
    mc, q, spin1z, spin2z = theta

    # Guard against nonsense / NaNs
    if not np.isfinite(mc) or not np.isfinite(q):
        return -np.inf

    try:
        mass1, mass2 = mc_q_to_masses(mc, q)

        # Extra safety: reject unphysical masses
        if mass1 <= 0 or mass2 <= 0:
            return -np.inf

        _, model = gw_waveform(
            mass1=mass1,
            mass2=mass2,
            spin1z=spin1z,
            spin2z=spin2z,
            delta_t=delta_t,
            f_lower=f_lower,
        )

        # If PyCBC returns an empty array, bail out
        if model is None or len(model) == 0:
            return -np.inf

    except Exception:
        # Any PyCBC error -> parameters effectively forbidden
        return -np.inf

    n = min(len(d), len(model))
    resid = d[:n] - model[:n]

    chi2 = inner(resid, resid, sigma=sigma, delta_t=delta_t)
    return -0.5 * chi2


def log_posterior(
    theta: np.ndarray,
    d: np.ndarray,
    sigma: float,
    delta_t: float,
    f_lower: float,
    bounds: PriorBounds = DEFAULT_PRIOR,
) -> float:
    """Log posterior (up to a constant) for the GW toy model."""
    lp = log_uniform_prior(theta, bounds=bounds)
    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(theta, d=d, sigma=sigma, delta_t=delta_t, f_lower=f_lower)
