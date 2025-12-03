def mc_q_to_masses(mc, q):
    """
    Convert chirp mass and mass ratio into component masses.
    q >= 1 by definition.
    """
    # m2 = Mc * (1+q)^(1/5) / q^(3/5)
    m2 = mc * (1 + q)**(1/5) / (q**(3/5))
    m1 = q * m2
    return m1, m2
