"""Likelihood and prior functions for the toy GW inference project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .waveform import sine_gaussian


@dataclass
class PriorBounds:
    """Uniform prior bounds for the three sine-Gaussian parameters.

    The prior is taken to be uniform inside the interval and zero outside.

    Attributes
    ----------
    A :
        Tuple ``(A_min, A_max)`` for the amplitude.
    t0 :
        Tuple ``(t0_min, t0_max)`` for the central time in seconds.
    f0 :
        Tuple ``(f0_min, f0_max)`` for the central frequency in Hz.
    """

    A: Tuple[float, float]
    t0: Tuple[float, float]
    f0: Tuple[float, float]


DEFAULT_PRIOR = PriorBounds(
    A=(0.0, 3.0),
    t0=(0.1, 0.4),
    f0=(50.0, 300.0),
)


def log_uniform_prior(theta: np.ndarray, bounds: PriorBounds = DEFAULT_PRIOR) -> float:
    """Log of a simple factorized uniform prior.

    Parameters
    ----------
    theta :
        1D array-like with entries ``(A, t0, f0)``.
    bounds :
        :class:`PriorBounds` instance specifying the allowed intervals.

    Returns
    -------
    lp :
        Log prior density.  Returns ``-np.inf`` if any parameter lies
        outside the allowed interval.
    """
    A, t0, f0 = theta

    if not (bounds.A[0] < A < bounds.A[1]):
        return -np.inf
    if not (bounds.t0[0] < t0 < bounds.t0[1]):
        return -np.inf
    if not (bounds.f0[0] < f0 < bounds.f0[1]):
        return -np.inf

    # Uniform within the bounds: constant log-prior.
    return 0.0


def log_likelihood(
    theta: np.ndarray,
    t: np.ndarray,
    d: np.ndarray,
    sigma: float,
    tau: float = 0.02,
    phi0: float = 0.0,
) -> float:
    """Gaussian log-likelihood for the sine-Gaussian toy model.

    Parameters
    ----------
    theta :
        1D array-like with entries ``(A, t0, f0)``.
    t, d :
        Arrays of time samples and measured strain data.
    sigma :
        Standard deviation of the (assumed white) Gaussian noise.
    tau, phi0 :
        Extra waveform parameters passed to
        :func:`toy_inference.waveform.sine_gaussian`.

    Returns
    -------
    logL :
        Log-likelihood of the data given the parameters, up to an
        additive constant.
    """
    A, t0, f0 = theta
    model = sine_gaussian(t, A=A, t0=t0, f0=f0, tau=tau, phi0=phi0)
    resid = d - model
    # Gaussian likelihood ~ exp(-0.5 * sum((resid/sigma)^2))
    logL = -0.5 * np.sum((resid / sigma) ** 2)
    return logL


def log_posterior(
    theta: np.ndarray,
    t: np.ndarray,
    d: np.ndarray,
    sigma: float,
    bounds: PriorBounds = DEFAULT_PRIOR,
    tau: float = 0.02,
    phi0: float = 0.0,
) -> float:
    """Log posterior (up to a constant) for the toy model.

    This simply adds the uniform log-prior to the Gaussian log-likelihood.
    """
    lp = log_uniform_prior(theta, bounds=bounds)
    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(theta, t=t, d=d, sigma=sigma, tau=tau, phi0=phi0)