# Tests/test_relative_binning.py

import numpy as np

from toy_inference.relative_binning import (  # :contentReference[oaicite:19]{index=19}
    compute_bins,
    setup_relative_binning,
    rb_likelihood,
)


def test_compute_bins_returns_monotonic_edges():
    n = 256
    freqs = np.linspace(20.0, 512.0, n)
    # Simple "waveform": linear phase chirp
    phase = 2.0 * np.pi * freqs * 0.01
    h0 = np.exp(1j * phase)

    edges = compute_bins(h0, freqs, max_phase_error=0.05)

    # Edges should be valid indices and strictly increasing
    assert edges.ndim == 1
    assert edges[0] >= 0
    assert edges[-1] <= n
    assert np.all(np.diff(edges) > 0)


def test_setup_relative_binning_produces_consistent_shapes():
    n = 256
    freqs = np.linspace(20.0, 512.0, n)
    phase = 2.0 * np.pi * freqs * 0.01
    h0 = np.exp(1j * phase)

    df = freqs[1] - freqs[0]
    psd = np.ones_like(h0)
    data_f = h0.copy()  # pretend data matches fiducial waveform

    rb = setup_relative_binning(data_f=data_f, h0=h0, psd=psd, df=df, freqs=freqs)

    n_bins = len(rb.idx_edges) - 1
    assert rb.A0.shape == (n_bins,)
    assert rb.B0.shape == (n_bins,)
    assert np.isfinite(rb.D)
    assert rb.h0 is h0
    assert rb.df == df


def test_rb_likelihood_prefers_matching_waveform():
    n = 256
    freqs = np.linspace(20.0, 512.0, n)
    phase = 2.0 * np.pi * freqs * 0.01
    h0 = np.exp(1j * phase)

    df = freqs[1] - freqs[0]
    psd = np.ones_like(h0)
    data_f = h0.copy()

    rb = setup_relative_binning(data_f=data_f, h0=h0, psd=psd, df=df, freqs=freqs)

    # Waveform generator that returns exactly h0 (perfect match)
    def gwfunc_true(*theta):
        return h0

    # Waveform generator with wrong amplitude (mismatch)
    def gwfunc_half(*theta):
        return 0.5 * h0

    theta_dummy = np.array([0.0])  # rb_likelihood ignores theta structure

    match_true = rb_likelihood(theta_dummy, rb, gwfunc_true)
    match_half = rb_likelihood(theta_dummy, rb, gwfunc_half)

    assert np.isfinite(match_true)
    assert np.isfinite(match_half)
    assert match_true > match_half
