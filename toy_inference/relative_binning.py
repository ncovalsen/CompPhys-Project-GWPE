# relative_binning.py

import numpy as np
from dataclasses import dataclass

@dataclass
class RBinned:
    idx_edges: np.ndarray
    A0: np.ndarray
    B0: np.ndarray
    D: float
    h0: np.ndarray
    freqs: np.ndarray
    psd: np.ndarray
    df: float


def compute_bins(h0, freqs, max_phase_error=0.03):
    phi = np.unwrap(np.angle(h0))
    dphi = np.gradient(phi, freqs)
    d2phi = np.abs(np.gradient(dphi, freqs))

    weight = np.sqrt(1 + d2phi**2)
    cum = np.cumsum(weight)
    cum /= cum[-1]

    N_bins = int(50 / max_phase_error)
    q = np.linspace(0, 1, N_bins + 1)
    edges = np.searchsorted(cum, q)
    return np.unique(edges)


def setup_relative_binning(data_f, h0, psd, df, freqs):
    idx_edges = compute_bins(h0, freqs)

    A0, B0 = [], []
    D = 0.0

    for i in range(len(idx_edges) - 1):
        i1, i2 = idx_edges[i], idx_edges[i+1]

        dseg = data_f[i1:i2]
        hseg = h0[i1:i2]
        inv_psd = 1.0 / psd[i1:i2]

        A0.append(np.sum(dseg * np.conj(hseg) * inv_psd) * df)
        B0.append(np.sum(hseg * np.conj(hseg) * inv_psd) * df)
        D  += np.sum(dseg * np.conj(dseg) * inv_psd) * df

    return RBinned(
        idx_edges=np.array(idx_edges),
        A0=np.array(A0),
        B0=np.array(B0),
        D=D,
        h0=h0,
        freqs=freqs,
        psd=psd,
        df=df
    )


def rb_likelihood(theta, rb: RBinned, gwfunc):
    h = gwfunc(*theta)
    ratio = h / rb.h0

    alpha, beta = [], []
    for i in range(len(rb.idx_edges) - 1):
        i1, i2 = rb.idx_edges[i], rb.idx_edges[i+1]

        rseg = ratio[i1:i2]
        hseg = h[i1:i2]
        inv_psd = 1.0 / rb.psd[i1:i2]

        alpha.append(np.sum(rseg * inv_psd) * rb.df)
        beta.append(np.sum((rseg * np.conj(rseg)) * inv_psd) * rb.df)

    alpha, beta = np.array(alpha), np.array(beta)

    # Relative-binning likelihood approximation
    match = np.real(np.sum(rb.A0 * alpha) - 0.5 * np.sum(rb.B0 * beta))

    return match

