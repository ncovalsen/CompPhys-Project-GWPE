# Tests/test_config.py

import numpy as np

from toy_inference.config import ToyModelConfig  # :contentReference[oaicite:11]{index=11}
from toy_inference.likelihood import DEFAULT_PRIOR  # :contentReference[oaicite:12]{index=12}
from toy_inference.mcmc import MCMCConfig  # :contentReference[oaicite:13]{index=13}


def test_toy_model_config_defaults_match_prior_and_mcmc():
    cfg = ToyModelConfig()

    assert isinstance(cfg.mcmc, MCMCConfig)
    assert cfg.burn_in == 5_000
    assert cfg.use_relative_binning is False

    # Defaults should mirror DEFAULT_PRIOR
    assert cfg.prior_bounds.mc == DEFAULT_PRIOR.mc
    assert cfg.prior_bounds.q == DEFAULT_PRIOR.q
    assert cfg.prior_bounds.spin1z == DEFAULT_PRIOR.spin1z
    assert cfg.prior_bounds.spin2z == DEFAULT_PRIOR.spin2z

    # MCMC step sizes should be of length 4 (mc, q, spin1z, spin2z)
    assert cfg.mcmc.step_sizes.shape == (4,)


def test_toy_model_config_can_be_overridden():
    custom_prior = DEFAULT_PRIOR.__class__(
        mc=(10.0, 20.0),
        q=DEFAULT_PRIOR.q,
        spin1z=DEFAULT_PRIOR.spin1z,
        spin2z=DEFAULT_PRIOR.spin2z,
    )
    custom_mcmc = MCMCConfig(n_steps=123, step_sizes=np.array([1.0, 1.0, 0.1, 0.1]))

    cfg = ToyModelConfig(
        prior_bounds=custom_prior,
        mcmc=custom_mcmc,
        burn_in=10,
        use_relative_binning=True,
    )

    assert cfg.prior_bounds.mc == (10.0, 20.0)
    assert cfg.mcmc.n_steps == 123
    assert cfg.burn_in == 10
    assert cfg.use_relative_binning is True
