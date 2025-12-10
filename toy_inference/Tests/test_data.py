# Tests/test_data.py

import numpy as np
import pytest

pytest.importorskip("pycbc")  # data helper uses gw_waveform under the hood

from toy_inference.data import (  # :contentReference[oaicite:4]{index=4}
    simulate_gw_data,
    save_simulated_data,
    load_simulated_data,
    SimulatedData,
    TrueParameters,
)
from toy_inference.waveform import gw_waveform  # :contentReference[oaicite:5]{index=5}


def test_simulate_gw_data_basic_properties():
    """Check shapes, metadata and parameter propagation."""
    sim = simulate_gw_data(
        fs=1024.0,
        noise_sigma=1e-23,
        mass1=12.0,
        mass2=8.0,
        spin1z=0.1,
        spin2z=-0.2,
        f_lower=30.0,
        random_seed=123,
    )

    assert isinstance(sim, SimulatedData)
    assert isinstance(sim.true_params, TrueParameters)

    # Time series shape and monotonicity
    assert sim.t.shape == sim.d.shape
    assert sim.t.ndim == 1
    assert sim.t.size > 0
    assert np.all(np.diff(sim.t) > 0)

    # Delta t and sigma should match fs and noise_sigma
    assert sim.sigma == pytest.approx(1e-23)
    assert sim.delta_t == pytest.approx(1.0 / 1024.0)

    # True parameters match inputs
    tp = sim.true_params
    assert tp.mass1 == pytest.approx(12.0)
    assert tp.mass2 == pytest.approx(8.0)
    assert tp.spin1z == pytest.approx(0.1)
    assert tp.spin2z == pytest.approx(-0.2)
    assert tp.f_lower == pytest.approx(30.0)


def test_simulate_gw_data_zero_noise_matches_model():
    """With zero noise, stored data should equal the model waveform."""
    fs = 1024.0
    delta_t = 1.0 / fs
    mass1 = 10.0
    mass2 = 11.0
    spin1z = 0.3
    spin2z = -0.1
    f_lower = 40.0

    sim = simulate_gw_data(
        fs=fs,
        noise_sigma=0.0,
        mass1=mass1,
        mass2=mass2,
        spin1z=spin1z,
        spin2z=spin2z,
        f_lower=f_lower,
        random_seed=1,
    )

    t_model, h_model = gw_waveform(
        mass1=mass1,
        mass2=mass2,
        spin1z=spin1z,
        spin2z=spin2z,
        delta_t=delta_t,
        f_lower=f_lower,
    )

    np.testing.assert_allclose(sim.t, t_model)
    np.testing.assert_allclose(sim.d, h_model)
    assert sim.sigma == pytest.approx(0.0)
    assert sim.delta_t == pytest.approx(delta_t)


def test_save_and_load_simulated_data_roundtrip(tmp_path):
    """Save to NPZ and load back, ensuring all fields are preserved."""
    sim = simulate_gw_data(
        fs=1024.0,
        noise_sigma=1e-22,
        mass1=9.0,
        mass2=7.0,
        spin1z=0.2,
        spin2z=0.0,
        f_lower=35.0,
        random_seed=999,
    )

    out_path = tmp_path / "sim_data.npz"
    save_simulated_data(sim, out_path)

    assert out_path.exists()

    sim2 = load_simulated_data(out_path)

    # Arrays
    np.testing.assert_allclose(sim.t, sim2.t)
    np.testing.assert_allclose(sim.d, sim2.d)

    # Scalars
    assert sim.sigma == pytest.approx(sim2.sigma)
    assert sim.delta_t == pytest.approx(sim2.delta_t)

    # True parameters
    tp1 = sim.true_params
    tp2 = sim2.true_params
    assert tp1.mass1 == pytest.approx(tp2.mass1)
    assert tp1.mass2 == pytest.approx(tp2.mass2)
    assert tp1.spin1z == pytest.approx(tp2.spin1z)
    assert tp1.spin2z == pytest.approx(tp2.spin2z)
    assert tp1.f_lower == pytest.approx(tp2.f_lower)
