"""Configuration helpers for the GW toy inference project."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from .likelihood import PriorBounds, DEFAULT_PRIOR
from .mcmc import MCMCConfig


@dataclass
class ToyModelConfig:
    """Bundle together common configuration for the toy GW run.

    Attributes
    ----------
    prior_bounds :
        Prior intervals for (mc, q, spin1z, spin2z).
    mcmc :
        MCMC configuration (number of steps, proposal widths, seed).
    burn_in :
        Number of initial MCMC samples to discard as burn-in.
    """

    prior_bounds: PriorBounds = field(
        default_factory=lambda: PriorBounds(
            mc=DEFAULT_PRIOR.mc,
            q=DEFAULT_PRIOR.q,
            spin1z=DEFAULT_PRIOR.spin1z,
            spin2z=DEFAULT_PRIOR.spin2z,
        )
    )
    mcmc: MCMCConfig = field(
        default_factory=lambda: MCMCConfig(
            n_steps=100_000,
            step_sizes=np.array([0.02, 0.02, 0.005, 0.005]),  # will be set in MCMCConfig.__post_init__
        )
    )
    burn_in: int = 5_000
    use_relative_binning: bool = False
