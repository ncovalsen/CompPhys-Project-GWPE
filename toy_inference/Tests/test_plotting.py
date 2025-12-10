# Tests/test_plotting.py

import numpy as np

from toy_inference.plotting import plot_data_and_model, plot_1d_posteriors  # :contentReference[oaicite:7]{index=7}


def test_plot_data_and_model_saves_file(tmp_path):
    t = np.linspace(0, 1, 100)
    data = np.sin(2 * np.pi * t)
    model = np.sin(2 * np.pi * t + 0.1)

    out = tmp_path / "data_model.png"
    plot_data_and_model(t, data, model, outfile=out)

    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_1d_posteriors_saves_file(tmp_path):
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(1000, 3))
    labels = [r"$m_c$", r"$q$", r"$\chi_1$"]
    truths = [0.0, 1.0, 0.0]

    out = tmp_path / "posteriors.png"
    plot_1d_posteriors(samples, labels, truths=truths, outfile=out, bins=30)

    assert out.exists()
    assert out.stat().st_size > 0
