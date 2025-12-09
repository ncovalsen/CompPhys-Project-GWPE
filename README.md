# Accelerated Likelihood Evaluation for Gravitational Wave Parameter Estimation: Relative Binning

This repository contains the code for a *PHYS381C* final project, which implements a **toy model** for gravitational-wave parameter estimation with a focus on accelerating the likelihood evaluation. The goal is to explore the **relative binning** (heterodyning) technique for speeding up Bayesian inference of long-duration signals. In this project, we restrict to a subset of parameters (two masses and two spin components) and compare a baseline MCMC inference with a future accelerated (relative-binning) approach. All code is written in Python and makes use of the PyCBC library for waveform generation.

## Installation and Setup

To set up the environment and install dependencies for this project, follow these steps:

1. **Clone or download the repository** to your local machine (e.g., via `git clone` or by extracting the provided archive).
2. **Ensure you have Python 3.7+** installed.
3. **Install required Python packages**. You can use `pip` to install the dependencies:

   ```bash
   pip install numpy matplotlib pycbc
   ```

   This will install NumPy for numerical computations, Matplotlib for plotting, and PyCBC (a gravitational wave data analysis library) which is used to generate waveforms.

No other compilation or build steps are needed. Once the above packages are installed, you are ready to run the code.

## Example Usage (Minimum Working Example)

After installation, you can run the main simulation and inference script to perform a minimal working example of the gravitational wave parameter estimation:

```bash
python run_toy_inference.py
```

This script will:

1. **Simulate Data**  
   Generate a synthetic time-series dataset containing a gravitational wave signal injected into Gaussian noise. The default injection is a binary system with:
   - `mass1 = 10 M⊙`
   - `mass2 = 10 M⊙`
   - `spin1z = 0.9`
   - `spin2z = 0.0`
   - `f_lower = 40 Hz`

   The signal is sampled at 2048 Hz, white Gaussian noise is added, and the resulting time array and strain data are saved to `data/gw_injection.npz`.

2. **Run MCMC Inference**  
   Perform Bayesian parameter estimation using a Metropolis–Hastings MCMC sampler. The code constructs a log-posterior function for the four parameters – chirp mass, mass ratio, and the two spin components – and runs an MCMC chain (default 30,000 samples, with a burn-in of 5,000) to sample from the posterior.

3. **Save Results**  
   After the MCMC finishes, the outputs are saved in the `results/` directory, including:
   - `results/mcmc_chain.npy` – full Markov chain
   - `results/mcmc_logp.npy` – log-posterior values
   - `results/summary.json` – summary of injected vs recovered parameters and acceptance rate

4. **Generate Plots**  
   The script also creates:
   - `results/timeseries_with_bestfit.png` – simulated strain data with best-fit waveform overlay
   - `results/posterior_1d_histograms.png` – 1D posterior histograms for each parameter (chirp mass, mass ratio, spin1z, spin2z) with true values indicated

These default settings should run out-of-the-box and demonstrate the full workflow from data generation to parameter estimation and visualization.

## Results and Benchmarks *(to be added)*

> **Note:** You should update this section once you have benchmark results.

Planned contents for this section:

- **Runtime Benchmarks**
  - Wall-clock time for baseline (direct) likelihood evaluation.
  - Wall-clock time for accelerated relative-binning implementation.
  - Speed-up factor as a function of signal duration or waveform length.

- **Accuracy / Validation**
  - Comparison of posteriors from baseline vs relative-binning.
  - Plots demonstrating that the accelerated likelihood reproduces the full likelihood to acceptable accuracy.

Suggested placeholder structure (fill these in later):

- `results/benchmark_table.md` – table of timing results.
- `results/benchmark_plots.png` – plot of runtime vs configuration.
- `results/posterior_comparison.png` – overlay of posteriors (baseline vs accelerated).

## Code Structure and Major Components

The project is organized as a Python package `toy_inference` containing modules for each aspect of the simulation and inference, along with an entry-point script.

### Main Script: `run_toy_inference.py`

Coordinates the end-to-end workflow:

- Sets up output directories (`data/`, `results/`).
- Instantiates a `ToyModelConfig` with prior bounds and MCMC settings.
- Calls `simulate_gw_data(...)` to generate the synthetic dataset and `save_simulated_data(...)` to store it.
- Wraps `log_posterior(...)` in a closure to fix the data and noise parameters.
- Runs the MCMC with `run_mcmc(...)`, starting near the true parameters (in chirp-mass and mass-ratio space).
- Computes posterior means (after burn-in) and converts them back to component masses.
- Saves numerical results and generates plots.

### Configuration – `toy_inference/config.py`

- **`ToyModelConfig` (dataclass)**  
  Collects configuration options for a run:
  - `prior_bounds`: instance of `PriorBounds` (see below) defining parameter ranges.
  - `mcmc`: instance of `MCMCConfig` controlling MCMC length and step sizes.
  - `burn_in`: number of initial samples to discard (default ~5000).

  Default priors are broad uniform ranges:
  - Chirp mass `Mc ∈ [5, 40] M⊙`
  - Mass ratio `q ∈ [1, 8]`
  - Spins `χ_{1z}, χ_{2z} ∈ [-0.99, 0.99]`

### Data Generation – `toy_inference/data.py`

- **`TrueParameters` (dataclass)**  
  Holds the “true” injection parameters:
  - `mass1`, `mass2` – component masses (solar masses)
  - `spin1z`, `spin2z` – dimensionless spin components along z
  - `f_lower` – low-frequency cutoff in Hz

- **`SimulatedData` (dataclass)**  
  Encapsulates the simulated time-series data:
  - `t` – time array
  - `d` – strain array (signal + noise)
  - `delta_t` – sampling interval
  - `sigma` – noise standard deviation
  - `true_params` – the `TrueParameters` instance used for injection

- **`simulate_gw_data(...)`**  
  Generates synthetic gravitational wave data:
  - Calls the waveform generator (`gw_waveform`) for the injected parameters.
  - Adds white Gaussian noise with specified `noise_sigma`.
  - Returns a `SimulatedData` instance.

- **`save_simulated_data(sim_data, out_path)`**  
  Saves a `SimulatedData` object to a compressed `.npz` file for later reuse.

### Waveform Generation – `toy_inference/waveform.py`

- **`gw_waveform(mass1, mass2, spin1z, spin2z, delta_t, f_lower)`**  
  Generates a time-domain gravitational wave plus-polarization strain using PyCBC’s `get_td_waveform`, typically with an IMRPhenomD-type approximant. Returns `(t, h_plus)` where:
  - `t` is the time array
  - `h_plus` is the strain waveform

- **`sine_gaussian(t, A, t0, f0, tau, phi0)`** *(optional utility)*  
  Produces a simple sine-Gaussian test signal (not required for the main workflow, but useful for quick tests).

### Likelihood and Prior – `toy_inference/likelihood.py`

- **`PriorBounds` (dataclass)**  
  Defines uniform prior ranges for each parameter:
  - `mc_bounds`, `q_bounds`, `spin1z_bounds`, `spin2z_bounds` – each a `(min, max)` tuple.

- **`log_uniform_prior(theta, bounds)`**  
  Returns the log prior probability for parameter vector `theta`:
  - `0.0` if all components lie within the specified bounds.
  - `-inf` if any parameter is out of bounds.

- **`log_likelihood(theta, d, sigma, delta_t, f_lower)`**  
  Computes the Gaussian log-likelihood:
  1. Converts `(Mc, q)` to `(m1, m2)` using `mc_q_to_masses`.
  2. Generates a model waveform using `gw_waveform`.
  3. Aligns model and data, forming the residual `resid = d - h`.
  4. Evaluates the noise-weighted inner product `(resid | resid)`.
  5. Returns `-0.5 * chi2`, where `chi2 = (resid | resid)`.

- **`log_posterior(theta, d, sigma, delta_t, f_lower, bounds)`**  
  Adds `log_uniform_prior` and `log_likelihood` to form the log posterior (up to a constant). If any term is `-inf`, the sum is `-inf`.

### MCMC Sampler – `toy_inference/mcmc.py`

- **`MCMCConfig` (dataclass)**  
  Controls the Metropolis–Hastings sampler:
  - `n_steps` – number of MCMC steps.
  - `step_sizes` – proposal standard deviations for each parameter.
  - `random_seed` – optional RNG seed for reproducibility.

- **`MCMCResult` (dataclass)**  
  Stores MCMC output:
  - `chain` – array of shape `(n_steps, n_dim)` with sampled parameters.
  - `logp_chain` – log-posterior values at each step.
  - `acceptance_rate` – fraction of proposed steps accepted.

- **`run_mcmc(log_posterior, initial_theta, config)`**  
  Implements a basic Metropolis–Hastings algorithm:
  - Proposes new parameters from a Gaussian centered on the current state.
  - Accepts or rejects proposals using the usual Metropolis criterion.
  - Returns an `MCMCResult` with the full chain and diagnostics.

### Plotting Utilities – `toy_inference/plotting.py`

- **`plot_data_and_model(t, data, model, title, outfile)`**  
  Creates a time-series plot of the data and a model waveform:
  - Plots `data` and `model` vs `t` with labels.
  - Adds axis labels, title, legend.
  - Saves to `outfile` (PNG) or shows interactively if `outfile` is `None`.

- **`plot_1d_posteriors(samples, labels, truths, outfile, bins)`**  
  Creates side-by-side 1D histograms of posterior samples:
  - One histogram per parameter, labeled with `labels`.
  - Optionally marks true values with vertical lines (`truths`).
  - Saves to `outfile` or displays interactively.

### Utility – `toy_inference/utils.py`

- **`mc_q_to_masses(mc, q)`**  
  Converts chirp mass `mc` and mass ratio `q` into component masses `(m1, m2)` using the standard chirp-mass relation:
  - Assumes `q = m1 / m2 ≥ 1`.
  - Returns `(m1, m2)` consistent with `mc` and `q`.

## Author

**Will Suan**
**Snehal Tibrewal**
**Nicholas Covalsen**
PHYS 381C Final Project  

## References

You may find the following references useful for context (not required to run the code):

- N. J. Cornish, **“Fast Fisher Matrices and Lazy Likelihoods”**, arXiv:1007.4820.
- B. Zackay, L. Dai, and T. Venumadhav, **“Relative Binning and Fast Likelihood Evaluation for Gravitational Wave Parameter Estimation”**, arXiv:1806.08792.
