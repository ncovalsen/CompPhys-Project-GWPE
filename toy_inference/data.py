"""Data generation helpers for the toy GW inference project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .waveform import gw_waveform


@dataclass
class TrueParameters:
    """Container for the true GW parameters used to generate synthetic data."""

    mass1: float
    mass2: float
    spin1z: float
    spin2z: float
    f_lower: float


@dataclass
class SimulatedData:
    """Container for simulated time-series GW data."""

    t: np.ndarray
    d: np.ndarray
    sigma: float
    delta_t: float
    true_params: TrueParameters


def simulate_gw_data(
    fs: float = 2048.0,
    noise_sigma: float = 1e-22,
    mass1: float = 10.0,
    mass2: float = 10.0,
    spin1z: float = 0.9,
    spin2z: float = 0.0,
    f_lower: float = 40.0,
    random_seed: Optional[int] = 42,
) -> SimulatedData:
    """Generate synthetic data containing a GW signal in white Gaussian noise."""
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

    delta_t = 1.0 / fs

    t, signal = gw_waveform(
        mass1=mass1,
        mass2=mass2,
        spin1z=spin1z,
        spin2z=spin2z,
        delta_t=delta_t,
        f_lower=f_lower,
    )

    noise = rng.normal(0.0, noise_sigma, size=signal.shape)
    d = signal + noise

    true_params = TrueParameters(
        mass1=mass1,
        mass2=mass2,
        spin1z=spin1z,
        spin2z=spin2z,
        f_lower=f_lower,
    )

    return SimulatedData(t=t, d=d, sigma=noise_sigma, delta_t=delta_t, true_params=true_params)


def save_simulated_data(sim_data: SimulatedData, out_path: Path) -> None:
    """Save simulated GW data to a compressed ``.npz`` file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tp = sim_data.true_params
    true_params_arr = np.array(
        [tp.mass1, tp.mass2, tp.spin1z, tp.spin2z, tp.f_lower],
        dtype=float,
    )

    np.savez_compressed(
        out_path,
        t=sim_data.t,
        d=sim_data.d,
        sigma=sim_data.sigma,
        delta_t=sim_data.delta_t,
        true_params=true_params_arr,
    )


def load_simulated_data(path: Path) -> SimulatedData:
    """Load simulated GW data previously saved with :func:`save_simulated_data`."""
    path = Path(path)
    data = np.load(path)

    t = data["t"]
    d = data["d"]
    sigma = float(data["sigma"])
    delta_t = float(data["delta_t"])
    (
        true_mass1,
        true_mass2,
        true_spin1z,
        true_spin2z,
        true_f_lower,
    ) = data["true_params"]

    true_params = TrueParameters(
        mass1=float(true_mass1),
        mass2=float(true_mass2),
        spin1z=float(true_spin1z),
        spin2z=float(true_spin2z),
        f_lower=float(true_f_lower),
    )

    return SimulatedData(
        t=t,
        d=d,
        sigma=sigma,
        delta_t=delta_t,
        true_params=true_params,
    )
