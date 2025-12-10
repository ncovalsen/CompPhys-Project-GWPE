# Tests/test_likelihood.py

import numpy as np
import pytest

pytest.importorskip("pycbc")  # likelihood uses gw_waveform (PyCBC) under the hood

from toy_inference.likelihood import (  # :contentReference[oaicite:15]{index=15}
    PriorBounds,
    DEFAULT_PRIOR,
    log_uniform_prior,
    log_likelihood,
    log_posterior,
    initialize_relative_binning,
)
from toy_inference.utils import mc_q_to_masses  # :contentReference[oaicite:16]{index=16}
from toy_inference.waveform import gw_waveform  # :contentReference[oaicite:17]{index=17}
import toy_inference.likelihood as lk_mod  # for toggling USE_RELATIVE_BINNING


def _masses_from_theta(theta):
    mc, q, spin1z, spin2z = theta
    return mc_q_to_masses(mc, q), spin1z, spin2z


def _make_gw_data_from_theta(theta, fs=1024.0, f_lower=40.0, sigma=1e-22, seed=0):
    rng = np.random.default_rng(seed)
    (m1, m2), spin1z, spin2z = _masses_from_theta(theta)
    delta_t = 1.0 / fs

    t, h = gw_waveform(
        mass1=m1,
        mass2=m2,
        spin1z=spin1z,
        spin2z=spin2z,
        delta_t=delta_t,
        f_lower=f_lower,
    )

    noise = rng.normal(0.0, sigma, size=h.shape)
    d = h + noise
    return t, d, sigma, delta_t, f_lower


def test_log_uniform_prior_inside_bounds_is_zero():
    theta = np.array(
        [
            0.5 * (DEFAULT_PRIOR.mc[0] + DEFAULT_PRIOR.mc[1]),
            0.5 * (DEFAULT_PRIOR.q[0] + DEFAULT_PRIOR.q[1]),
            0.0,
            0.0,
        ]
    )

    lp = log_uniform_prior(theta, bounds=DEFAULT_PRIOR)
    assert lp == 0.0


def test_log_uniform_prior_outside_bounds_is_minus_inf():
    # mc too large
    theta = np.array([DEFAULT_PRIOR.mc[1] + 5.0, 2.0, 0.0, 0.0])
    lp = log_uniform_prior(theta, bounds=DEFAULT_PRIOR)
    assert not np.isfinite(lp)


def test_log_likelihood_prefers_injected_parameters():
    # Choose a theta well inside the prior
    theta_true = np.array([15.0, 1.5, 0.1, -0.2])

    t, d, sigma, delta_t, f_lower = _make_gw_data_from_theta(
        theta_true, fs=512.0, f_lower=30.0, sigma=1e-22, seed=1
    )

    theta_wrong = theta_true + np.array([5.0, 0.5, 0.3, 0.3])

    logL_true = log_likelihood(theta_true, d=d, sigma=sigma, delta_t=delta_t, f_lower=f_lower)
    logL_wrong = log_likelihood(theta_wrong, d=d, sigma=sigma, delta_t=delta_t, f_lower=f_lower)

    assert np.isfinite(logL_true)
    assert np.isfinite(logL_wrong)
    assert logL_true > logL_wrong


def test_log_posterior_respects_prior_bounds():
    theta_true = np.array([15.0, 1.5, 0.0, 0.0])
    t, d, sigma, delta_t, f_lower = _make_gw_data_from_theta(theta_true)

    # Outside prior for mc
    theta_bad = np.array([DEFAULT_PRIOR.mc[1] + 10.0, 1.5, 0.0, 0.0])
    lp_bad = log_posterior(
        theta_bad, d=d, sigma=sigma, delta_t=delta_t, f_lower=f_lower, bounds=DEFAULT_PRIOR
    )
    assert not np.isfinite(lp_bad)

    # Inside prior
    lp_good = log_posterior(
        theta_true, d=d, sigma=sigma, delta_t=delta_t, f_lower=f_lower, bounds=DEFAULT_PRIOR
    )
    assert np.isfinite(lp_good)


def test_relative_binning_log_likelihood_runs_and_is_finite():
    """
    Smoke test: RB initialization + RB likelihood path produce a finite value.
    We don't demand numerical equality with the time-domain likelihood,
    just that the RB machinery runs without error.
    """
    theta_true = np.array([15.0, 1.5, 0.0, 0.0])
    t, d, sigma, delta_t, f_lower = _make_gw_data_from_theta(theta_true, sigma=1e-22)

    # Initialize RB state
    initialize_relative_binning(d=d, sigma=sigma, delta_t=delta_t, f_lower=f_lower)

    # Standard likelihood
    lk_mod.USE_RELATIVE_BINNING = False
    logL_std = log_likelihood(theta_true, d=d, sigma=sigma, delta_t=delta_t, f_lower=f_lower)
    assert np.isfinite(logL_std)

    # RB likelihood
    lk_mod.USE_RELATIVE_BINNING = True
    logL_rb = log_likelihood(theta_true, d=d, sigma=sigma, delta_t=delta_t, f_lower=f_lower)
    lk_mod.USE_RELATIVE_BINNING = False  # reset for other tests

    assert np.isfinite(logL_rb)
