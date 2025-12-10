# Tests/test_mcmc.py

import numpy as np

from toy_inference.mcmc import MCMCConfig, MCMCResult, run_mcmc  # :contentReference[oaicite:9]{index=9}


def gaussian_log_posterior(theta):
    """Simple 2D standard normal posterior for testing."""
    theta = np.asarray(theta)
    return -0.5 * np.dot(theta, theta)


def test_run_mcmc_basic_shapes_and_acceptance():
    n_steps = 500
    config = MCMCConfig(
        n_steps=n_steps,
        step_sizes=np.array([0.5, 0.5]),
        random_seed=1,
    )

    result = run_mcmc(gaussian_log_posterior, initial_theta=np.zeros(2), config=config)

    assert isinstance(result, MCMCResult)
    assert result.chain.shape == (n_steps, 2)
    assert result.logp_chain.shape == (n_steps,)
    assert 0.0 < result.acceptance_rate < 1.0


def test_run_mcmc_reproducibility_with_seed():
    config = MCMCConfig(
        n_steps=200,
        step_sizes=np.array([0.5, 0.5]),
        random_seed=123,
    )

    res1 = run_mcmc(gaussian_log_posterior, np.zeros(2), config=config)
    res2 = run_mcmc(gaussian_log_posterior, np.zeros(2), config=config)

    np.testing.assert_allclose(res1.chain, res2.chain)
    np.testing.assert_allclose(res1.logp_chain, res2.logp_chain)


def test_run_mcmc_raises_on_bad_step_sizes_shape():
    config = MCMCConfig(
        n_steps=10,
        step_sizes=np.array([0.5]),  # wrong shape for 2D theta
        random_seed=1,
    )

    raised = False
    try:
        run_mcmc(gaussian_log_posterior, np.zeros(2), config=config)
    except ValueError:
        raised = True

    assert raised
