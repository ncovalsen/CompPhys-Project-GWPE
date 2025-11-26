import numpy as np
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. Toy GW signal model
# =========================

def sine_gaussian(t, A, t0, f0, tau=0.02, phi0=0.0):
    """
    Simple sine-Gaussian burst:
    h(t) = A * exp(-(t - t0)^2 / (2 tau^2)) * cos(2π f0 (t - t0) + phi0)
    """
    envelope = np.exp(-0.5 * ((t - t0) / tau) ** 2)
    phase = 2.0 * np.pi * f0 * (t - t0) + phi0
    return A * envelope * np.cos(phase)

# =========================
# 2. Simulate data
# =========================

def simulate_data():
    # Time grid
    duration = 0.5       # seconds
    fs = 2048.0          # Hz
    dt = 1.0 / fs
    t = np.arange(0, duration, dt)

    # True parameters
    true_A = 1.0
    true_t0 = 0.25       # seconds
    true_f0 = 150.0      # Hz

    # Noise
    sigma = 0.3          # std dev of Gaussian noise
    noise = sigma * np.random.randn(len(t))

    # Signal + noise
    h_signal = sine_gaussian(t, true_A, true_t0, true_f0)
    d = h_signal + noise

    theta_true = np.array([true_A, true_t0, true_f0])
    return t, d, sigma, theta_true

# =========================
# 3. Log-likelihood & prior
# =========================

def log_likelihood(theta, t, d, sigma):
    """
    Gaussian likelihood:
    ln L ∝ -0.5 * sum( (d - h(theta))^2 / sigma^2 )
    """
    A, t0, f0 = theta
    h = sine_gaussian(t, A, t0, f0)
    resid = d - h
    return -0.5 * np.sum((resid / sigma) ** 2)

def log_prior(theta):
    """
    Simple uniform priors:
      A   ~ U(0, 3)
      t0  ~ U(0.1, 0.4)
      f0  ~ U(50, 300)
    Returns -inf if out of bounds.
    """
    A, t0, f0 = theta

    if not (0.0 < A < 3.0):
        return -np.inf
    if not (0.1 < t0 < 0.4):
        return -np.inf
    if not (50.0 < f0 < 300.0):
        return -np.inf

    # uniform priors -> constant log-prior inside bounds
    return 0.0

def log_posterior(theta, t, d, sigma):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, t, d, sigma)

# =========================
# 4. Metropolis–Hastings MCMC
# =========================

def run_mcmc(t, d, sigma, n_steps=20000, step_sizes=None):
    if step_sizes is None:
        # proposal widths in (A, t0, f0)
        step_sizes = np.array([0.05, 0.001, 3.0])

    # Initialize at some reasonable point
    theta_cur = np.array([0.8, 0.24, 120.0])
    logp_cur = log_posterior(theta_cur, t, d, sigma)

    chain = np.zeros((n_steps, len(theta_cur)))
    logp_chain = np.zeros(n_steps)

    for i in range(n_steps):
        # Propose new point: Gaussian jump
        theta_prop = theta_cur + step_sizes * np.random.randn(len(theta_cur))
        logp_prop = log_posterior(theta_prop, t, d, sigma)

        # MH acceptance
        log_alpha = logp_prop - logp_cur
        if np.log(np.random.rand()) < log_alpha:
            theta_cur = theta_prop
            logp_cur = logp_prop

        chain[i] = theta_cur
        logp_chain[i] = logp_cur

    return chain, logp_chain

# =========================
# 5. Main: simulate, sample, plot
# =========================

def main():
    np.random.seed(42)

    # Simulate data
    t, d, sigma, theta_true = simulate_data()
    print("True parameters (A, t0, f0):", theta_true)

    # Run MCMC
    chain, logp_chain = run_mcmc(t, d, sigma, n_steps=30000)

    # Throw away burn-in
    burn = 5000
    samples = chain[burn:]

    A_samples = samples[:, 0]
    t0_samples = samples[:, 1]
    f0_samples = samples[:, 2]

    # =========================
    # 5a. Plot data & best-fit
    # =========================
    # Use posterior means as a crude "best-fit"
    theta_mean = np.array([
        np.mean(A_samples),
        np.mean(t0_samples),
        np.mean(f0_samples),
    ])
    print("Posterior mean (A, t0, f0):", theta_mean)

    h_best = sine_gaussian(t, *theta_mean)

    plt.figure(figsize=(10, 4))
    plt.plot(t, d, label="data", lw=1, alpha=0.7)
    plt.plot(t, h_best, label="best-fit model", lw=2)
    plt.xlabel("t [s]")
    plt.ylabel("strain (arb. units)")
    plt.legend()
    plt.title("Simulated GW-like data and best-fit sine-Gaussian")
    plt.tight_layout()

    # =========================
    # 5b. 1D posteriors
    # =========================
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    labels = [r"$A$", r"$t_0$ [s]", r"$f_0$ [Hz]"]
    truths = theta_true

    for i, (ax, s, lab, truth) in enumerate(zip(
        axes,
        [A_samples, t0_samples, f0_samples],
        labels,
        truths
    )):
        ax.hist(s, bins=50, density=True, alpha=0.7, histtype="stepfilled")
        ax.axvline(truth, color="r", lw=2, label="truth")
        ax.set_xlabel(lab)
        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()