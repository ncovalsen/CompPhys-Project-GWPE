"""Toy gravitational-wave parameter inference package.

This small package provides a worked example of Bayesian parameter
estimation for a sine-Gaussian burst signal in Gaussian noise.
"""

from .toy_inference.waveform import sine_gaussian
from .toy_inference.data import simulate_sine_gaussian_data, SimulatedData, TrueParameters
from .toy_inference.likelihood import log_likelihood, log_posterior, PriorBounds, DEFAULT_PRIOR
from .toy_inference.mcmc import MCMCConfig, MCMCResult, run_mcmc
from .toy_inference.config import ToyModelConfig