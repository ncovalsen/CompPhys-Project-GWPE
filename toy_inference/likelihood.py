"""Likelihood and prior functions for the toy GW inference project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .relative_binning import setup_relative_binning, rb_likelihood
from .waveform import gw_waveform
from .utils import mc_q_to_masses

# === Relative binning toggle ===
USE_RELATIVE_BINNING = False

# These will be filled in when RB is initialized
_rb_state = None
_h0_cached = None



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


def initialize_relative_binning(d, sigma, delta_t, f_lower):
    """
    Build the fiducial waveform h0 and binning structure.
    Called once at the beginning of an MCMC run.
    """
    global _rb_state, _h0_cached

    # Fiducial theta = center of prior
    theta0 = np.array([
        0.5*(DEFAULT_PRIOR.mc[0] + DEFAULT_PRIOR.mc[1]),
        0.5*(DEFAULT_PRIOR.q[0] + DEFAULT_PRIOR.q[1]),
        0.0, 0.0
    ])
    mc0, q0, spin1z0, spin2z0 = theta0
    m10, m20 = mc_q_to_masses(mc0, q0)

    _, h0 = gw_waveform(
        mass1=m10,
        mass2=m20,
        spin1z=spin1z0,
        spin2z=spin2z0,
        delta_t=delta_t,
        f_lower=f_lower,
    )
    if h0 is None:
        raise RuntimeError("Fiducial waveform for RB was empty.")

    n = min(len(d), len(h0))
    d = d[:n]
    h0 = h0[:n]

    # FFT
    df = 1.0 / (n * delta_t)
    H0 = np.fft.rfft(h0)
    Df = np.fft.rfft(d)

    # White noise PSD = sigma^2 * dt
    psd = sigma**2 * delta_t * np.ones_like(H0)
    freqs = np.fft.rfftfreq(n, delta_t)

    _h0_cached = H0
    _rb_state = setup_relative_binning(
        data_f=Df,
        h0=H0,
        psd=psd,
        df=df,
        freqs=freqs,
    )


def log_likelihood(
    theta: np.ndarray,
    d: np.ndarray,
    sigma: float,
    delta_t: float,
    f_lower: float,
) -> float:
    """Gaussian or RB log-likelihood for the GW toy model."""
    global _rb_state, _h0_cached

    # If RB mode enabled, we call rb_likelihood
    if USE_RELATIVE_BINNING:
        if _rb_state is None:
            raise RuntimeError("Relative binning requested but not initialized. "
                               "Call initialize_relative_binning(...) first.")

        # Compute frequency-domain waveform
        mc, q, spin1z, spin2z = theta
        try:
            m1, m2 = mc_q_to_masses(mc, q)
            _, h = gw_waveform(
                mass1=m1,
                mass2=m2,
                spin1z=spin1z,
                spin2z=spin2z,
                delta_t=delta_t,
                f_lower=f_lower,
            )
        except Exception:
            return -np.inf

        if h is None or len(h) == 0:
            return -np.inf

        # Match length to data
        n = min(len(d), len(h))
        H = np.fft.rfft(h[:n])

        return rb_likelihood(theta, _rb_state, lambda *args: H)

    # === Standard likelihood mode (existing code) ===
    mc, q, spin1z, spin2z = theta

    if not np.isfinite(mc) or not np.isfinite(q):
        return -np.inf

    try:
        mass1, mass2 = mc_q_to_masses(mc, q)
        if mass1 <= 0 or mass2 <= 0 or q < 1 or mass1 > 200 or mass2 > 200:
            return -np.inf

        _, model = gw_waveform(
            mass1=mass1,
            mass2=mass2,
            spin1z=spin1z,
            spin2z=spin2z,
            delta_t=delta_t,
            f_lower=f_lower,
        )

        if model is None or len(model) == 0:
            return -np.inf

    except Exception:
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
