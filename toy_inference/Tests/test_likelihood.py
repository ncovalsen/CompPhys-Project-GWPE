import numpy as np

from toy_inference.likelihood import (
    DEFAULT_PRIOR,
    log_uniform_prior,
    log_likelihood,
    log_posterior,
)
from toy_inference.waveform import sine_gaussian


def _make_toy_data(noise_sigma=0.05, random_seed=0):
    """
    Build a simple synthetic dataset from the analytic sine-Gaussian
    waveform used inside the likelihood code. This avoids depending
    on any particular data helper.
    """
    rng = np.random.default_rng(random_seed)

    duration = 0.5
    fs = 2048.0
    t = np.arange(0.0, duration, 1.0 / fs)

    A_true = 1.0
    t0_true = 0.25
    f0_true = 150.0

    h_true = sine_gaussian(t, A=A_true, t0=t0_true, f0=f0_true)
    noise = rng.normal(0.0, noise_sigma, size=t.shape)
    d = h_true + noise

    theta_true = np.array([A_true, t0_true, f0_true])
    return t, d, noise_sigma, theta_true


def test_log_uniform_prior_inside_bounds_is_zero():
    theta = np.array(
        [
            0.5 * (DEFAULT_PRIOR.A[0] + DEFAULT_PRIOR.A[1]),
            0.5 * (DEFAULT_PRIOR.t0[0] + DEFAULT_PRIOR.t0[1]),
            0.5 * (DEFAULT_PRIOR.f0[0] + DEFAULT_PRIOR.f0[1]),
        ]
    )

    lp = log_uniform_prior(theta, bounds=DEFAULT_PRIOR)
    assert lp == 0.0


def test_log_uniform_prior_outside_bounds_is_minus_inf():
    # Amplitude too large
    theta = np.array([DEFAULT_PRIOR.A[1] + 1.0, 0.2, 100.0])
    lp = log_uniform_prior(theta, bounds=DEFAULT_PRIOR)
    assert not np.isfinite(lp)


def test_log_likelihood_prefers_true_parameters():
    t, d, sigma, theta_true = _make_toy_data(noise_sigma=0.05, random_seed=0)

    theta_wrong = theta_true + np.array([0.5, 0.01, 20.0])

    logL_true = log_likelihood(theta_true, t=t, d=d, sigma=sigma)
    logL_wrong = log_likelihood(theta_wrong, t=t, d=d, sigma=sigma)

    assert logL_true > logL_wrong


def test_log_posterior_respects_prior():
    t, d, sigma, theta_true = _make_toy_data()

    # Outside prior bounds
    theta_bad = np.array(
        [DEFAULT_PRIOR.A[1] + 1.0, theta_true[1], theta_true[2]]
    )
    lp_bad = log_posterior(theta_bad, t=t, d=d, sigma=sigma)
    assert not np.isfinite(lp_bad)

    # Inside prior bounds
    lp_good = log_posterior(theta_true, t=t, d=d, sigma=sigma)
    assert np.isfinite(lp_good)
