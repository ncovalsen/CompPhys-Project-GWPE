"""Data generation helpers for the toy GW inference project.

For now we only generate synthetic time-series data containing a
sine-Gaussian burst embedded in white Gaussian noise.  The module is
written so that we can later plug in real detector data instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

from .waveform import sine_gaussian


@dataclass
class TrueParameters:
    """Container for the true parameters used to generate synthetic data.

    Attributes
    ----------
    A :
        True amplitude of the injected burst.
    t0 :
        True central time of the burst (seconds).
    f0 :
        True central frequency of the burst (Hz).
    """

    A: float
    t0: float
    f0: float


@dataclass
class SimulatedData:
    """Container for simulated time-series data.

    Attributes
    ----------
    t :
        1D array of time samples.
    d :
        1D array of measured strain (signal + noise).
    sigma :
        Noise standard deviation (assumed constant over time).
    true_params :
        :class:`TrueParameters` instance with injection values.
    """

    t: np.ndarray
    d: np.ndarray
    sigma: float
    true_params: TrueParameters


def simulate_sine_gaussian_data(
    duration: float = 0.5,
    fs: float = 2048.0,
    noise_sigma: float = 0.1,
    A: float = 1.0,
    t0: float = 0.25,
    f0: float = 150.0,
    tau: float = 0.02,
    phi0: float = 0.0,
    random_seed: Optional[int] = 42,
) -> SimulatedData:
    """Generate synthetic data containing a sine-Gaussian burst.

    Parameters
    ----------
    duration :
        Total length of the simulated time series in seconds.
    fs :
        Sampling frequency in Hz.
    noise_sigma :
        Standard deviation of the additive white Gaussian noise.
    A, t0, f0, tau, phi0 :
        Parameters passed to :func:`~toy_inference.waveform.sine_gaussian`.
        These define the injected signal.
    random_seed :
        Optional integer seed to make simulations reproducible.

    Returns
    -------
    simulated :
        :class:`SimulatedData` object with arrays and true parameters.
    """
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

    dt = 1.0 / fs
    t = np.arange(0.0, duration, dt)

    true_params = TrueParameters(A=A, t0=t0, f0=f0)

    signal = sine_gaussian(t, A=A, t0=t0, f0=f0, tau=tau, phi0=phi0)
    noise = noise_sigma * rng.standard_normal(size=t.shape)
    d = signal + noise

    return SimulatedData(t=t, d=d, sigma=noise_sigma, true_params=true_params)


def save_simulated_data(sim_data: SimulatedData, out_path: Path) -> None:
    """Save simulated data to a compressed ``.npz`` file.

    The file contains the fields ``t``, ``d``, ``sigma``, and
    ``true_params`` (as a tiny structured array).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tp = sim_data.true_params
    true_params_arr = np.array((tp.A, tp.t0, tp.f0))

    np.savez_compressed(
        out_path,
        t=sim_data.t,
        d=sim_data.d,
        sigma=sim_data.sigma,
        true_params=true_params_arr,
    )


def load_simulated_data(path: Path) -> SimulatedData:
    """Load simulated data previously saved with :func:`save_simulated_data`.

    Parameters
    ----------
    path :
        Path to a ``.npz`` file on disk.

    Returns
    -------
    simulated :
        :class:`SimulatedData` instance.
    """
    path = Path(path)
    data = np.load(path)

    t = data["t"]
    d = data["d"]
    sigma = float(data["sigma"])
    true_A, true_t0, true_f0 = data["true_params"]
    true_params = TrueParameters(A=float(true_A), t0=float(true_t0), f0=float(true_f0))
    return SimulatedData(t=t, d=d, sigma=sigma, true_params=true_params)