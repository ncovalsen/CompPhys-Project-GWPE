"""Toy gravitational-wave parameter inference package.

This small package provides a worked example of Bayesian parameter
estimation for a sine-Gaussian burst signal in Gaussian noise.
"""

from .waveform import sine_gaussian
from .data import simulate_sine_gaussian_data, SimulatedData, TrueParameters
from .likelihood import log_likelihood, log_posterior, PriorBounds, DEFAULT_PRIOR
from .mcmc import MCMCConfig, MCMCResult, run_mcmc
from .config import ToyModelConfig
