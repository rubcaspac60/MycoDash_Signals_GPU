# GPU‑Accelerated Signal and Propagation Analysis

This repository contains a collection of Python modules and tools for the
analysis of multi‑channel time series recorded from spatially separated
probes.  It extends the original CPU‑only pipeline (`signal_analysis_pipeline.py`)
with a PyTorch/CUDA implementation capable of processing gigabyte‑sized
datasets on modern GPUs.  The core features include spectral similarity
analysis, candidate pair selection, continuous wavelet transforms (CWT),
wavelet coherence, propagation delay estimation and visualisation.

## Contents

| File/Folder                         | Description |
|------------------------------------|-------------|
| `signal_analysis_pipeline.py`       | Original CPU implementation. |
| `propagation_analysis.py`           | High‑level propagation analysis utilities. |
| `signal_analysis_pipeline_torch.py` | PyTorch/CUDA accelerated pipeline (this is the primary module for large data). |
| `tutorial.ipynb`                    | Original tutorial demonstrating the basic GPU pipeline on a synthetic dataset. |
| `extended_tutorial.ipynb`           | Extended tutorial covering multi‑band similarity, phase drift analysis and 3D visualisation. |
| `app.py` and `pages.py`             | Streamlit application for interactive analysis and visualisation (see below). |
| `README.md`                         | This document. |

## Installation

1. **Python and dependencies**

   The GPU pipeline depends on [PyTorch](https://pytorch.org) compiled with
   CUDA 12.x support and the [SciPy](https://scipy.org) stack.  Install
   these using `pip` or `conda` for your platform.  For example:

   ```bash
   # Create a virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate

   # Install PyTorch with CUDA
   pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

   # Install SciPy and other requirements
   pip install numpy scipy matplotlib
   ```

   Adapt the CUDA version (`cu118` above) to match your GPU drivers.

2. **Clone the repository**

   ```bash
   git clone <repository_url>
   cd <repository>
   ```

3. **Run the tutorial**

   Launch Jupyter and open `tutorial.ipynb` to follow a step‑by‑step
   introduction to the pipeline.  The notebook generates synthetic data,
   computes spectral similarity, performs the CWT on the GPU and
   visualises coherence and velocities.  Make sure that a CUDA device is
   visible to PyTorch (`torch.cuda.is_available()` should be `True`).

## Using the GPU Pipeline

The primary entry points are defined in
`signal_analysis_pipeline_torch.py`.  Typical usage:

```python
from signal_analysis_pipeline_torch import (
    detrend_signals,
    compute_fft_magnitude,
    cosine_similarity_matrix,
    select_candidate_pairs,
    compute_cwt,
    wavelet_coherence,
    estimate_propagation_delays,
    filtered_cosine_similarity,
    ProbeLayout,
    build_probe_positions,
    save_results_to_db,
)

# Load your multi‑channel data as a NumPy array of shape (n_channels, n_samples)
signals = ...
fs = 1000.0  # sampling rate in Hz

# Preprocess
signals = detrend_signals(signals)

# Spectral features
freqs_fft, mags = compute_fft_magnitude(signals, sampling_rate=fs)
sim = cosine_similarity_matrix(mags)
pairs = select_candidate_pairs(sim, threshold=0.8)

# Wavelet analysis on candidate pairs
for i, j in pairs:
    freqs, coeffs_i = compute_cwt(signals[i], sampling_rate=fs, f_min=0.5, f_max=100.0, n_freqs=64)
    _,      coeffs_j = compute_cwt(signals[j], sampling_rate=fs, f_min=0.5, f_max=100.0, n_freqs=64)
    coh, phase = wavelet_coherence(coeffs_i, coeffs_j)
    velocities = estimate_propagation_delays(phase, freqs, distance_in_um, sampling_rate=fs)
    # further analysis ...

# Persist results
save_results_to_db('results.db', mags, sim, pairs, coherence_results, freqs_fft, freqs_cwt_dict, sensor_labels, layout)
```

See the docstrings within `signal_analysis_pipeline_torch.py` for detailed
descriptions of each function and the mathematical conventions used.

### Frequency‑Selective Similarity

The helper function `filtered_cosine_similarity` applies a bank of
Butterworth band‑pass filters to the signals and computes a separate
cosine similarity matrix for each band.  This can help identify
correlations confined to specific frequency ranges (e.g. theta or gamma
rhythms).  Provide a list of `(low, high)` tuples in Hertz and the
function returns a dictionary mapping bands to similarity matrices.

### Extended Analysis

Building on the basic functionality, the GPU pipeline exposes additional
utilities for deeper inspection of your signals:

* **Multi‑band similarity (`multi_band_similarity`)** – computes both
  frequency‑domain cosine similarity and time‑domain correlation across
  user‑defined frequency bands.  This dual perspective can reveal
  band‑limited synchrony and amplitude correlations that are not
  apparent in the full‑band spectra.  See the *Band Similarity* page
  in the Streamlit app or the `extended_tutorial.ipynb` for examples.
* **Phase drift analysis (`phase_drift_analysis`)** – fits a straight
  line to the unwrapped wavelet phase as a function of time and
  converts the slope into a drift value for each frequency.  A
  non‑zero drift indicates a systematic change in relative phase
  (e.g. due to changes in propagation speed).  The *Drift Analysis*
  page in the Streamlit app demonstrates this metric.
* **3D sensor visualisation (`plot_sensor_positions_3d`)** – draws
  interactive scatter plots of your probe geometry and can map
  per‑sensor values (mean velocity, drift magnitude, etc.) to marker
  colours.  Useful when working with large multi‑probe arrays.  This
  function is available in the `visualization_tools.py` module.

These extended features are optional; if you simply wish to run the
basic analysis pipeline you can ignore them.  When exploring large
datasets or seeking more nuanced insights, they offer valuable
additional perspectives.

## Visualisation

Although the heavy analysis runs on the GPU, visualisation is handled
on the CPU.  The tutorial notebook demonstrates basic plots of the
coherence spectrogram and propagation velocities using Matplotlib.  For
larger datasets we recommend using [Plotly](https://plotly.com/python/)
for interactive heatmaps and 3D surfaces.  You can also create
animations or GIFs by iterating over time slices and saving frames via
`matplotlib.animation` or `imageio`.

## Streamlit Application

A simple Streamlit application is provided in `app.py`.  It allows you
to upload your own CSV files, run the analysis pipeline on the fly and
visualise the results directly in the browser.  To launch the app run:

```bash
streamlit run app.py
```

The application supports large datasets by computing on the GPU and
streaming results to the client as they become available.  It exposes
multiple pages defined in `pages.py` to guide users through data
ingestion, parameter selection, signal inspection, coherence maps and
propagation analysis.  Feel free to extend the app with additional
plots or metrics tailored to your experiments.

## Notes and Limitations

* The PyTorch implementation assumes that a compatible GPU is
  available.  If `torch.cuda.is_available()` returns `False` the
  pipeline will transparently fall back to CPU tensors.  Processing
  large datasets on the CPU may be slower than the original NumPy
  version due to the overhead of PyTorch abstractions.
* Continuous wavelet transforms are computed using an analytic Morlet
  filter bank.  Filter lengths scale with inverse frequency, so very
  low analysis frequencies lead to large convolution kernels.  Ensure
  that your GPU has sufficient memory or restrict the minimum
  frequency accordingly.
* When saving results to the SQLite database all tensors are converted
  to NumPy arrays.  Binary blobs contain the raw floats and can be
  reloaded by calling `numpy.frombuffer(...)` with appropriate
  dtypes and shapes.

## Credits

The GPU implementation is inspired by the work of Tom Runia and
colleagues on the **PyTorchWavelets** project and the Morlet filter
design described in Torrence & Compo (1998).  The original CPU
pipeline was developed by the project owners and remains available in
`signal_analysis_pipeline.py`.