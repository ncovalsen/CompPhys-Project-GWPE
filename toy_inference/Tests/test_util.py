# Tests/test_utils.py

import numpy as np

from toy_inference.utils import mc_q_to_masses  # :contentReference[oaicite:0]{index=0}


def chirp_mass(m1, m2):
    """Helper used only in tests."""
    m_total = m1 + m2
    eta = m1 * m2 / m_total**2
    return m_total * eta ** (3.0 / 5.0)


def test_mc_q_to_masses_recovers_chirp_mass_and_q():
    mc_true = 15.0
    q_true = 3.0

    m1, m2 = mc_q_to_masses(mc_true, q_true)

    # By definition q = m1/m2 >= 1
    assert m1 >= m2
    np.testing.assert_allclose(m1 / m2, q_true, rtol=1e-12)

    # Check that feeding masses back into chirp-mass relation reproduces Mc
    mc_from_masses = chirp_mass(m1, m2)
    np.testing.assert_allclose(mc_from_masses, mc_true, rtol=1e-12)


def test_mc_q_to_masses_equal_masses_when_q_is_one():
    mc_true = 10.0
    q_true = 1.0

    m1, m2 = mc_q_to_masses(mc_true, q_true)

    # For q = 1 we expect equal masses.
    np.testing.assert_allclose(m1, m2, rtol=1e-12)
