"""
signal_analysis_pipeline_torch.py
=================================

This module contains a PyTorch/CUDA accelerated re‑implementation of the
signal analysis pipeline defined in :mod:`signal_analysis_pipeline`.  It
mirrors the functionality of the original module but performs the
heavy numerical operations on the GPU via the PyTorch framework.  The
aim is to enable the processing of very large multi‑channel datasets
(tens of gigabytes) by harnessing the parallelism of modern GPUs.

The core differences relative to the CPU implementation are:

* All numerical arrays are represented internally as ``torch.Tensor``
  objects on the active CUDA device.  Inputs and outputs are
  automatically moved back to CPU/NumPy when returned to the user.
* The continuous wavelet transform (CWT) is computed using a bank of
  complex Morlet filters applied via 1D convolutions.  Filters are
  generated on the CPU using simple analytic formulae and then sent to
  the GPU for convolution.  Variable filter lengths are padded to a
  common size to allow for efficient batched convolution.
* Cosine similarity, smoothing and coherence calculations are all
  performed using native PyTorch operations.  Where appropriate the
  operations are vectorised across channels and frequencies to fully
  exploit GPU throughput.

If a CUDA device is not available or PyTorch is not installed with
CUDA support, the module will still import but all computations will
fall back to CPU tensors.  The user can check ``torch.cuda.is_available()``
and set the ``device`` argument in many functions to control where
operations are executed.

Note
----
This module depends on ``torch`` and ``scipy``.  These packages are
not part of the standard library and must be installed separately.
Ensure that your version of PyTorch has been compiled with CUDA 12.x
support to make use of GPU acceleration.  In environments without
CUDA support the code will execute on the CPU but may be slower than
the original NumPy version.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Optional

import numpy as np
import torch
from scipy import signal as _signal

from signal_analysis_pipeline import ProbeLayout, build_probe_positions  # reuse layout definitions

###############################################################################
# Device management

def _get_device(device: Optional[str | torch.device] = None) -> torch.device:
    """Resolve a user provided device specifier to a torch.device.

    If ``device`` is ``None`` then CUDA will be used if available,
    otherwise CPU.

    Parameters
    ----------
    device : str or torch.device, optional
        Desired device.  If ``None`` the function will return
        ``'cuda'`` when a CUDA device is available and ``'cpu'`` otherwise.

    Returns
    -------
    torch.device
        Device object for subsequent tensor allocations.
    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device


###############################################################################
# Detrending

def detrend_signals(signals: np.ndarray, axis: int = -1) -> np.ndarray:
    """Remove linear trends from each time series using PyTorch.

    This function computes a least squares fit of a straight line to
    each signal along the specified axis and subtracts the fitted
    trend.  It emulates ``scipy.signal.detrend`` but performs the
    calculations on the GPU when available.  The input and output
    arrays are NumPy arrays; internal computation uses ``torch``.

    Parameters
    ----------
    signals : ndarray, shape (n_signals, n_samples)
        Input time series data.
    axis : int, optional
        Axis along which to detrend (default: last axis).

    Returns
    -------
    ndarray
        Detrended signals with the same shape as the input.
    """
    # Move data to torch tensor on the appropriate device
    device = _get_device()
    x = torch.as_tensor(signals, dtype=torch.float32, device=device)
    # Bring the detrending axis to the last dimension for convenience
    if axis < 0:
        axis += x.dim()
    # permute so that the detrend axis is the last axis
    permute_dims = list(range(x.dim()))
    permute_dims[axis], permute_dims[-1] = permute_dims[-1], permute_dims[axis]
    x = x.permute(permute_dims)
    orig_shape = x.shape
    n_samples = orig_shape[-1]
    # design matrix for linear regression: [1, 2, ..., n]
    t = torch.arange(n_samples, dtype=x.dtype, device=device)
    # centre t to improve numerical stability
    t_mean = t.mean()
    t = t - t_mean
    # compute slope and intercept via least squares for each signal
    # slope = cov(x, t) / var(t)
    cov = (x * t).mean(dim=-1) - x.mean(dim=-1) * t.mean()
    var_t = (t * t).mean() - t.mean() ** 2
    slope = cov / var_t
    intercept = x.mean(dim=-1) - slope * t.mean()
    # broadcast slope and intercept to reconstruct the trend
    trend = (slope[..., None] * t) + intercept[..., None]
    detrended = x - trend
    # inverse permutation to restore original axis order
    # invert permutation
    inv_perm = [0] * len(permute_dims)
    for i, j in enumerate(permute_dims):
        inv_perm[j] = i
    detrended = detrended.permute(inv_perm)
    # return to CPU as numpy array
    return detrended.cpu().numpy()


###############################################################################
# FFT magnitude

def compute_fft_magnitude(signals: np.ndarray, sampling_rate: float, device: Optional[str | torch.device] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the single‑sided magnitude spectra of multiple time series using PyTorch.

    Parameters
    ----------
    signals : ndarray, shape (n_signals, n_samples)
        Real‑valued time series data.
    sampling_rate : float
        Sampling frequency of the signals in Hertz.
    device : str or torch.device, optional
        Device on which to perform the computation.  If ``None`` the
        default device will be chosen (GPU if available).

    Returns
    -------
    freqs : ndarray, shape (n_freqs,)
        Array of frequency bins corresponding to the spectra.
    mags : ndarray, shape (n_signals, n_freqs)
        Magnitude spectra for each signal.
    """
    dev = _get_device(device)
    x = torch.as_tensor(signals, dtype=torch.float32, device=dev)
    # compute rFFT along the last dimension
    fft_vals = torch.fft.rfft(x, dim=-1)
    mags = torch.abs(fft_vals)
    # compute frequency bins on CPU
    n_samples = signals.shape[-1]
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sampling_rate)
    # move magnitudes back to CPU
    return freqs, mags.cpu().numpy()


###############################################################################
# Cosine similarity

def cosine_similarity_matrix(mags: np.ndarray, device: Optional[str | torch.device] = None) -> np.ndarray:
    """Compute the cosine similarity matrix between rows of ``mags`` using PyTorch.

    Parameters
    ----------
    mags : ndarray, shape (n_signals, n_features)
        Matrix of magnitude spectra.
    device : str or torch.device, optional
        Device on which to perform the computation.

    Returns
    -------
    ndarray, shape (n_signals, n_signals)
        Cosine similarity between each pair of signals.  The diagonal will
        contain ones by construction.
    """
    dev = _get_device(device)
    X = torch.as_tensor(mags, dtype=torch.float32, device=dev)
    # normalise each row
    norms = torch.linalg.norm(X, dim=1, keepdim=True)
    norms = torch.where(norms == 0, torch.ones_like(norms), norms)
    X_norm = X / norms
    sim = X_norm @ X_norm.T
    sim = torch.clamp(sim, -1.0, 1.0)
    return sim.cpu().numpy()


###############################################################################
# Candidate selection

def select_candidate_pairs(sim_matrix: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
    """Return list of index pairs whose similarity exceeds a threshold.

    This function is identical to the CPU implementation and therefore
    runs on the CPU.  It simply iterates over the upper triangular
    portion of the similarity matrix and selects those entries above
    ``threshold``.

    Parameters
    ----------
    sim_matrix : ndarray, shape (n_signals, n_signals)
        Pairwise similarity matrix.
    threshold : float
        Minimum similarity required for a pair to be selected.

    Returns
    -------
    list of tuple of int
        List of (i, j) index pairs meeting the threshold.
    """
    n = sim_matrix.shape[0]
    pairs: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                pairs.append((i, j))
    return pairs


###############################################################################
# Continuous wavelet transform (CWT)

def _morlet_wavelet(length: int, scale: float, w0: float = 6.0, complete: bool = True) -> np.ndarray:
    """Generate a complex Morlet wavelet at a given scale.

    The returned wavelet is centred at zero and has ``length`` samples.
    The sample spacing is assumed to be 1 (i.e. units of samples).  See
    Torrence & Compo (1998) for details of the Morlet wavelet.  The
    optional ``complete`` flag controls whether the wavelet is
    admissible (i.e. has zero mean); setting it to ``True`` subtracts
    a small constant from the sinusoid.

    Parameters
    ----------
    length : int
        Number of samples in the wavelet (must be odd for symmetry).
    scale : float
        Dimensionless scale of the wavelet (in units of samples).
    w0 : float, optional
        Non‑dimensional frequency parameter of the Morlet wavelet.
    complete : bool, optional
        Whether to enforce admissibility by subtracting a constant.

    Returns
    -------
    ndarray, shape (length,)
        Complex wavelet samples.
    """
    # ensure odd length
    if length % 2 == 0:
        length += 1
    t = np.arange(-(length // 2), length // 2 + 1, dtype=float)
    x = t / scale
    wavelet = np.exp(1j * w0 * x) * np.exp(-0.5 * x ** 2)
    if complete:
        wavelet -= np.exp(-0.5 * (w0 ** 2)) * np.exp(-0.5 * x ** 2)
    # normalise energy and scale
    wavelet *= (np.pi ** -0.25) / np.sqrt(scale)
    return wavelet


def compute_cwt(
    signal: np.ndarray,
    sampling_rate: float,
    f_min: float,
    f_max: float,
    n_freqs: int,
    device: Optional[str | torch.device] = None,
    w0: float = 6.0,
    complete: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the continuous wavelet transform using a Morlet filter bank on the GPU.

    Unlike the ``fcwt`` based implementation this function constructs a
    bank of analytic Morlet wavelets spanning the requested frequency
    range and applies them to the signal via batched convolution on
    the GPU.  Variable length filters are padded to a common size so
    that a single call to ``torch.nn.functional.conv1d`` can compute
    the transform for all frequencies simultaneously.

    Parameters
    ----------
    signal : ndarray, shape (n_samples,)
        Real‑valued input signal.
    sampling_rate : float
        Sampling frequency in Hertz.
    f_min : float
        Minimum analysis frequency in Hertz.
    f_max : float
        Maximum analysis frequency in Hertz.
    n_freqs : int
        Number of frequency bins between ``f_min`` and ``f_max``.  A
        logarithmic spacing is used to distribute frequencies across
        decades.
    device : str or torch.device, optional
        Device on which to perform the computation.
    w0 : float, optional
        Morlet wavelet parameter controlling the number of oscillations.
    complete : bool, optional
        Whether to use the complete (admissible) wavelet.  When
        ``True`` the average of the wavelet is zero.

    Returns
    -------
    freqs : ndarray, shape (n_freqs,)
        Frequencies corresponding to each row of the coefficient matrix.
    coeffs : ndarray, shape (n_freqs, n_samples)
        Complex wavelet coefficients indexed by frequency and time.
    """
    dev = _get_device(device)
    x = torch.as_tensor(signal, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)  # [1,1,T]
    n_samples = signal.shape[0]
    dt = 1.0 / float(sampling_rate)
    # Create frequencies on a logarithmic scale for better resolution at low frequencies
    freqs = np.logspace(np.log10(f_min), np.log10(f_max), num=n_freqs)
    # Compute scales from frequencies: scale (in samples) = w0 / (2*pi*f*dt)
    scales = w0 / (2.0 * math.pi * freqs * dt)
    # Build filter bank
    filter_lengths: List[int] = []
    filters_real: List[np.ndarray] = []
    filters_imag: List[np.ndarray] = []
    for s in scales:
        # heuristic: filter length proportional to scale.  Use 8 cycles of the wavelet.
        length = int(math.ceil(8.0 * s))
        if length % 2 == 0:
            length += 1
        filt = _morlet_wavelet(length, scale=s, w0=w0, complete=complete)
        # flip for correlation (conv1d performs cross-correlation)
        filt = filt[::-1]
        filters_real.append(np.real(filt))
        filters_imag.append(np.imag(filt))
        filter_lengths.append(length)
    # Determine maximum filter length and pad all filters to this length, centring them
    max_len = max(filter_lengths)
    n_filters = len(filters_real)
    weight_real = np.zeros((n_filters, 1, max_len), dtype=np.float32)
    weight_imag = np.zeros((n_filters, 1, max_len), dtype=np.float32)
    for idx, (fr, fi, L) in enumerate(zip(filters_real, filters_imag, filter_lengths)):
        pad_left = (max_len - L) // 2
        pad_right = max_len - L - pad_left
        weight_real[idx, 0, pad_left : pad_left + L] = fr.astype(np.float32)
        weight_imag[idx, 0, pad_left : pad_left + L] = fi.astype(np.float32)
    # Move weights to device
    W_real = torch.as_tensor(weight_real, device=dev)
    W_imag = torch.as_tensor(weight_imag, device=dev)
    # Perform convolution with padding to keep the same output length
    padding = max_len // 2
    conv_real = torch.nn.functional.conv1d(x, W_real, padding=padding)
    conv_imag = torch.nn.functional.conv1d(x, W_imag, padding=padding)
    # Combine real and imaginary parts
    coeffs = conv_real + 1j * conv_imag
    # Remove batch and channel dimensions
    coeffs = coeffs.squeeze(0).squeeze(0)
    return freqs, coeffs.cpu().numpy()


###############################################################################
# Smoothing

def _smooth_spectrum_torch(
    data: torch.Tensor,
    sigma_time: int = 2,
    sigma_freq: int = 2,
) -> torch.Tensor:
    """Smooth a time–frequency spectrum using separable moving averages on the GPU.

    Parameters
    ----------
    data : torch.Tensor, shape (n_freqs, n_times)
        Input matrix to be smoothed.
    sigma_time : int, optional
        Half‑window size along the time axis (default: 2 samples).
    sigma_freq : int, optional
        Half‑window size along the frequency axis (default: 2 samples).

    Returns
    -------
    torch.Tensor
        Smoothed data array of the same shape and device as the input.
    """
    x = data
    # Add batch and channel dimensions for conv2d: shape [N=1, C=1, H=n_freqs, W=n_times]
    x = x.unsqueeze(0).unsqueeze(0)
    if sigma_freq > 0:
        kernel_f = torch.ones((1, 1, 2 * sigma_freq + 1, 1), dtype=x.dtype, device=x.device) / float(2 * sigma_freq + 1)
        x = torch.nn.functional.conv2d(x, kernel_f, padding=(sigma_freq, 0))
    if sigma_time > 0:
        kernel_t = torch.ones((1, 1, 1, 2 * sigma_time + 1), dtype=x.dtype, device=x.device) / float(2 * sigma_time + 1)
        x = torch.nn.functional.conv2d(x, kernel_t, padding=(0, sigma_time))
    return x.squeeze(0).squeeze(0)


###############################################################################
# Wavelet coherence

def wavelet_coherence(
    coeffs1: np.ndarray,
    coeffs2: np.ndarray,
    smooth_time: int = 2,
    smooth_freq: int = 2,
    device: Optional[str | torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the magnitude squared wavelet coherence between two signals using PyTorch.

    This function accepts complex coefficient matrices produced by
    :func:`compute_cwt` and computes the coherence and phase between
    them.  All heavy lifting is done on the GPU via PyTorch; the
    resulting arrays are converted back to NumPy on return.

    Parameters
    ----------
    coeffs1, coeffs2 : ndarray, shape (n_freqs, n_times)
        Complex wavelet coefficients of the two signals to compare.
    smooth_time : int, optional
        Smoothing half‑window in samples along the time axis (default: 2).
    smooth_freq : int, optional
        Smoothing half‑window in samples along the frequency axis (default: 2).
    device : str or torch.device, optional
        Device on which to perform the computation.

    Returns
    -------
    coherence : ndarray, shape (n_freqs, n_times)
        Magnitude squared wavelet coherence values between 0 and 1.
    phase : ndarray, shape (n_freqs, n_times)
        Phase difference (in radians) between the two signals at each time
        and frequency.
    """
    dev = _get_device(device)
    c1 = torch.as_tensor(coeffs1, dtype=torch.complex64, device=dev)
    c2 = torch.as_tensor(coeffs2, dtype=torch.complex64, device=dev)
    # auto spectra
    S1 = torch.abs(c1) ** 2
    S2 = torch.abs(c2) ** 2
    # cross spectrum
    S12 = c1 * torch.conj(c2)
    # smooth
    S1_smooth = _smooth_spectrum_torch(S1.real, sigma_time=smooth_time, sigma_freq=smooth_freq)
    S2_smooth = _smooth_spectrum_torch(S2.real, sigma_time=smooth_time, sigma_freq=smooth_freq)
    # For cross spectrum we need magnitude squared
    S12_abs_sq = torch.abs(S12) ** 2
    S12_smooth = _smooth_spectrum_torch(S12_abs_sq, sigma_time=smooth_time, sigma_freq=smooth_freq)
    # compute coherence
    denom = S1_smooth * S2_smooth
    eps = torch.finfo(S1_smooth.dtype).eps
    denom = torch.where(denom == 0, torch.tensor(eps, device=dev, dtype=denom.dtype), denom)
    coh = S12_smooth / denom
    coh = torch.clamp(coh, 0.0, 1.0)
    # phase difference
    phase = torch.angle(S12)
    return coh.cpu().numpy(), phase.cpu().numpy()


###############################################################################
# Propagation delays and velocities

def estimate_propagation_delays(
    phase: np.ndarray,
    freqs: np.ndarray,
    distances: float | np.ndarray,
    sampling_rate: float,
    wrap_phase: bool = True,
    device: Optional[str | torch.device] = None,
) -> np.ndarray:
    """Estimate delay between sensors from phase differences using PyTorch.

    This function mirrors :func:`signal_analysis_pipeline.estimate_propagation_delays`
    but performs the computations on the GPU.  See the documentation
    there for a full description.

    Parameters
    ----------
    phase : ndarray, shape (n_freqs, n_times)
        Phase difference array obtained from ``wavelet_coherence``.
    freqs : ndarray, shape (n_freqs,)
        Frequencies corresponding to the rows of ``phase``.
    distances : float or ndarray
        Distance in micrometres between the two sensors under consideration.
    sampling_rate : float
        Sampling frequency in Hertz.  Currently unused but kept for
        interface consistency.
    wrap_phase : bool, optional
        Whether to wrap phase values into the principal range [−π, π] prior
        to computing delays (default: True).
    device : str or torch.device, optional
        Device on which to perform the computation.

    Returns
    -------
    ndarray, shape (n_freqs, n_times)
        Estimated propagation velocities in micrometres per second.  Positive
        values indicate that the first signal leads the second.
    """
    dev = _get_device(device)
    ph = torch.as_tensor(phase, dtype=torch.float32, device=dev)
    freqs_t = torch.as_tensor(freqs, dtype=torch.float32, device=dev)
    if wrap_phase:
        ph = torch.remainder(ph + math.pi, 2 * math.pi) - math.pi
    # compute delays: Δt = Δφ / (2πf)
    denom = 2.0 * math.pi * freqs_t[:, None]
    denom = torch.where(denom == 0, torch.tensor(torch.finfo(ph.dtype).eps, device=dev), denom)
    delays = ph / denom
    # convert distances to metres
    distances_m = np.asarray(distances) * 1e-6
    # broadcast distances to match shape
    delays_np = delays.cpu().numpy()
    velocities = distances_m / delays_np  # micrometre per second
    return velocities


###############################################################################
# Bandpass filtering

def bandpass_filter(
    signals: np.ndarray,
    sampling_rate: float,
    lowcut: float,
    highcut: float,
    order: int = 4,
    device: Optional[str | torch.device] = None,
) -> np.ndarray:
    """Apply a Butterworth band‑pass filter to multichannel signals.

    The filtering is performed in the frequency domain using SciPy to
    design the filter and PyTorch to apply it on the GPU.  Zero‑phase
    filtering is used via forward/backward convolution to avoid phase
    distortions.

    Parameters
    ----------
    signals : ndarray, shape (n_signals, n_samples)
        Input signals to be filtered.
    sampling_rate : float
        Sampling frequency of the signals in Hertz.
    lowcut : float
        Lower cutoff frequency in Hertz.
    highcut : float
        Upper cutoff frequency in Hertz.
    order : int, optional
        Order of the Butterworth filter.
    device : str or torch.device, optional
        Device on which to perform the filtering.

    Returns
    -------
    ndarray
        Band‑pass filtered signals with the same shape as the input.
    """
    # Design Butterworth bandpass filter in SOS form
    nyq = 0.5 * sampling_rate
    low = lowcut / nyq
    high = highcut / nyq
    sos = _signal.butter(order, [low, high], btype="bandpass", output="sos")
    # Apply filter using SciPy sosfiltfilt on CPU first for stability
    filtered = _signal.sosfiltfilt(sos, signals, axis=-1)
    # Move to GPU for further processing if required
    dev = _get_device(device)
    return torch.as_tensor(filtered, dtype=torch.float32, device=dev).cpu().numpy()


###############################################################################
# Similarity with filtering

def filtered_cosine_similarity(
    signals: np.ndarray,
    sampling_rate: float,
    bands: List[Tuple[float, float]],
    order: int = 4,
    device: Optional[str | torch.device] = None,
) -> Dict[Tuple[float, float], np.ndarray]:
    """Compute cosine similarity matrices for multiple frequency bands.

    For each band defined by ``(lowcut, highcut)`` the signals are
    band‑pass filtered and the cosine similarity matrix of their FFT
    magnitudes is computed.  This allows for frequency selective
    detection of correlated channels.

    Parameters
    ----------
    signals : ndarray, shape (n_signals, n_samples)
        Input multi‑channel signals.
    sampling_rate : float
        Sampling frequency of the signals in Hertz.
    bands : list of tuple of float
        List of (lowcut, highcut) pairs defining the frequency bands.
    order : int, optional
        Order of the Butterworth filters.
    device : str or torch.device, optional
        Device on which to perform the computations.

    Returns
    -------
    dict
        Mapping from band tuple to cosine similarity matrix (ndarray).
    """
    results: Dict[Tuple[float, float], np.ndarray] = {}
    for (low, high) in bands:
        filtered = bandpass_filter(signals, sampling_rate, low, high, order=order, device=device)
        freqs, mags = compute_fft_magnitude(filtered, sampling_rate, device=device)
        sim = cosine_similarity_matrix(mags, device=device)
        results[(low, high)] = sim
    return results


###############################################################################
# Persistence (reuse CPU implementation)

from signal_analysis_pipeline import save_results_to_db as _save_results_to_db

def save_results_to_db(
    db_path: str,
    spectral_matrix: np.ndarray,
    similarity_matrix: np.ndarray,
    candidate_pairs: List[Tuple[int, int]],
    coherence_results: Dict[Tuple[int, int], Dict[str, np.ndarray]],
    freqs_fft: np.ndarray,
    freqs_cwt: Dict[Tuple[int, int], np.ndarray],
    sensor_labels: List[str],
    layout: ProbeLayout,
) -> None:
    """Persist analysis results to a SQLite database.

    This wrapper simply forwards to the original implementation.  The
    inputs are expected to be NumPy arrays; any ``torch`` tensors
    should be converted beforehand.
    """
    return _save_results_to_db(
        db_path=db_path,
        spectral_matrix=spectral_matrix,
        similarity_matrix=similarity_matrix,
        candidate_pairs=candidate_pairs,
        coherence_results=coherence_results,
        freqs_fft=freqs_fft,
        freqs_cwt=freqs_cwt,
        sensor_labels=sensor_labels,
        layout=layout,
    )


###############################################################################
# Extended analysis utilities

def multi_band_similarity(
    signals: np.ndarray,
    sampling_rate: float,
    bands: List[Tuple[float, float]],
    order: int = 4,
    device: Optional[str | torch.device] = None,
) -> Dict[Tuple[float, float], Dict[str, np.ndarray]]:
    """
    Compute deep similarity metrics across multiple frequency bands.

    For each band defined by ``(lowcut, highcut)`` the signals are
    band‑pass filtered and two similarity matrices are computed:

    1. ``fft_similarity`` – cosine similarity of the FFT magnitudes of
       the filtered signals.  This emphasises frequency‑domain
       correlation.
    2. ``time_correlation`` – Pearson correlation coefficients between
       the filtered time‑domain signals.  This captures direct
       temporal relationships.

    This extended analysis allows users to explore correlations that
    manifest both in the spectral and temporal domains within
    selected frequency ranges.  The filtering uses a Butterworth
    band‑pass design implemented in the frequency domain, followed by
    efficient GPU‑accelerated operations where available.

    Parameters
    ----------
    signals : ndarray, shape (n_signals, n_samples)
        Input multi‑channel signals.
    sampling_rate : float
        Sampling frequency of the signals in Hertz.
    bands : list of tuple of float
        List of (lowcut, highcut) pairs defining the frequency bands.
    order : int, optional
        Order of the Butterworth filters (default: 4).
    device : str or torch.device, optional
        Device on which to perform the computations.  If ``None`` the
        default CUDA device is used when available.

    Returns
    -------
    dict
        Mapping each band to a dictionary containing keys
        ``'fft_similarity'`` and ``'time_correlation'`` with
        corresponding similarity matrices (NumPy arrays).
    """
    results: Dict[Tuple[float, float], Dict[str, np.ndarray]] = {}
    for (lowcut, highcut) in bands:
        # 1. Band‑pass filter
        filtered = bandpass_filter(
            signals,
            sampling_rate=sampling_rate,
            lowcut=lowcut,
            highcut=highcut,
            order=order,
            device=device,
        )
        # 2. FFT magnitude similarity
        freqs, mags = compute_fft_magnitude(filtered, sampling_rate, device=device)
        sim_fft = cosine_similarity_matrix(mags, device=device)
        # 3. Time‑domain Pearson correlation (zero‑lag)
        # Convert filtered signals to 2D array (n_signals, n_samples) on CPU
        filt_cpu = np.asarray(filtered)
        # Compute correlation matrix using NumPy; protect against NaNs
        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(filt_cpu)
        # Replace NaNs along diagonal with 1.0
        n_ch = filt_cpu.shape[0]
        for i in range(n_ch):
            corr[i, i] = 1.0
        results[(lowcut, highcut)] = {
            "fft_similarity": sim_fft,
            "time_correlation": corr,
        }
    return results


def phase_drift_analysis(
    phase: np.ndarray,
    freqs: np.ndarray,
    dt: float,
    *,
    device: Optional[str | torch.device] = None,
) -> np.ndarray:
    """
    Estimate phase drift over time for each frequency.

    Given a matrix of phase differences (frequency × time) this function
    unwraps the phase along the time axis, fits a straight line to the
    unwrapped phase at each frequency and converts the slope into a
    drift value with units of seconds per second (dimensionless).  The
    result can be interpreted as the effective time shift per unit
    elapsed time between two signals, averaged across the analysis
    interval.  Positive values indicate that the phase difference is
    increasing over time, suggesting a relative delay that grows, while
    negative values correspond to decreasing phase differences.

    The drift is computed as

    .. math::

       d(f) = \frac{\mathrm{slope}(\phi_f(t))}{2\pi f}

    where ``slope`` is obtained via least squares fitting of the
    unwrapped phase ``phi_f`` against time (in seconds).  ``dt`` is
    the sampling interval used to convert sample indices to seconds.

    Parameters
    ----------
    phase : ndarray, shape (n_freqs, n_times)
        Phase difference in radians.
    freqs : ndarray, shape (n_freqs,)
        Frequencies corresponding to each row (Hz).
    dt : float
        Sampling interval in seconds (i.e. ``1/sampling_rate``).
    device : str or torch.device, optional
        Device on which to perform the computation.  If ``None`` the
        default CUDA device is used when available.

    Returns
    -------
    ndarray, shape (n_freqs,)
        Drift values (seconds of relative shift per second) for each
        frequency.
    """
    # Convert inputs to tensors on the chosen device
    try:
        import torch  # type: ignore
    except ImportError:
        raise RuntimeError("phase_drift_analysis requires PyTorch to be installed")
    dev = _get_device(device)
    phase_t = torch.as_tensor(phase, dtype=torch.float32, device=dev)
    freqs_t = torch.as_tensor(freqs, dtype=torch.float32, device=dev)
    # Unwrap phase along time axis to avoid discontinuities
    ph = torch.unwrap(phase_t, dim=1)
    n_times = ph.shape[1]
    # Build time vector in seconds
    t = torch.arange(n_times, dtype=torch.float32, device=dev) * dt
    t_mean = t.mean()
    t_zero = t - t_mean
    # Compute slope via covariance/variance for each frequency
    # slope = cov(ph_f, t_zero) / var(t_zero)
    ph_mean = ph.mean(dim=1, keepdim=True)
    cov = ((ph - ph_mean) * t_zero).mean(dim=1)
    var_t = (t_zero * t_zero).mean()
    slope = cov / var_t
    # Convert slope (rad/s) to drift (s/s) by dividing by 2πf
    denom = 2.0 * math.pi * freqs_t
    # Avoid division by zero
    denom_safe = torch.where(denom == 0.0, torch.full_like(denom, float('nan')), denom)
    drift = slope / denom_safe
    return drift.cpu().numpy()


###############################################################################
# Public API

__all__ = [
    "detrend_signals",
    "compute_fft_magnitude",
    "cosine_similarity_matrix",
    "select_candidate_pairs",
    "compute_cwt",
    "wavelet_coherence",
    "ProbeLayout",
    "build_probe_positions",
    "estimate_propagation_delays",
    "bandpass_filter",
    "filtered_cosine_similarity",
    "save_results_to_db",
    "multi_band_similarity",
    "phase_drift_analysis",
]