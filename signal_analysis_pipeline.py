"""
signal_analysis_pipeline.py
===========================

This module contains utilities for analysing multi‑channel time series data
recorded from spatially separated probes.  It implements the key steps of
the *Coherent Multiplex* pipeline — spectral similarity, candidate selection
and wavelet coherence — and extends the approach to incorporate known
sensor positions.  In addition, helper functions are provided for data
ingestion, logging, persistence and interactive visualisation.

The design is deliberately modular: each function performs a single task
and can be reused in isolation or as part of a larger processing chain.
Logging and optional progress reporting make it suitable for long‑running
jobs on large datasets.  The code relies only on standard numerical
packages plus ``fcwt`` for the fast continuous wavelet transform; if
``fcwt`` is not available on your system you can replace the call with
another CWT implementation (e.g. ``pycwt``) but note that performance
may differ.

Usage example::

    from signal_analysis_pipeline import (
        load_csv_data,
        detrend_signals,
        compute_fft_magnitude,
        cosine_similarity_matrix,
        select_candidate_pairs,
        compute_cwt,
        wavelet_coherence,
        build_probe_positions,
        compute_spatial_distances,
        save_results_to_db,
    )

    # Load data (each column is a channel, each row a timestamp)
    df = load_csv_data("my_data.csv", channels=["1_Blue Ave. (V)", "1_White Ave. (V)"])

    # Detrend and normalise
    signals = detrend_signals(df.values)

    # Compute FFT magnitudes for each channel
    freqs, mags = compute_fft_magnitude(signals, sampling_rate=1000.0)

    # Build similarity graph
    sim = cosine_similarity_matrix(mags)
    pairs = select_candidate_pairs(sim, threshold=0.8)

    # Compute coherence on candidate pairs
    for (i, j) in pairs:
        freqs_cwt, coeffs_i = compute_cwt(signals[i], sampling_rate=1000.0,
                                          f_min=0.1, f_max=50.0, n_freqs=128)
        _,       coeffs_j = compute_cwt(signals[j], sampling_rate=1000.0,
                                          f_min=0.1, f_max=50.0, n_freqs=128)
        coh, freqs_cwt = wavelet_coherence(coeffs_i, coeffs_j)
        # ... further analysis ...

The functions defined here are intended to operate on NumPy arrays and
Pandas data structures.  Plotting utilities using Plotly and Matplotlib
are provided in the companion file ``visualisation_tools.py``.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from scipy.signal import detrend as _detrend
from fcwt import cwt

# Attempt to import fcwt; fall back to None if unavailable.  Users can
# substitute another CWT implementation in compute_cwt().
try:
    import fcwt  # type: ignore
    _HAVE_FCWT = True
except ImportError:
    fcwt = None  # type: ignore
    _HAVE_FCWT = False


def setup_logging(log_path: str, level: int = logging.INFO) -> logging.Logger:
    """Configure a logger to write messages to ``log_path``.

    Parameters
    ----------
    log_path : str
        Path to the log file.  The parent directory will be created if
        necessary.
    level : int, optional
        Logging level to use (default: ``logging.INFO``).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    # Create file handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    # Avoid duplicate handlers if called multiple times
    if not logger.handlers:
        logger.addHandler(fh)
    return logger


def load_csv_data(
    file_path: str,
    channels: Optional[List[str]] = None,
    time_column: Optional[str] = None,
    sep: str = ",",
    header: int | None = 0,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Load a CSV file containing time series data.

    The CSV is expected to have time increasing down the rows and channels
    across the columns.  If ``time_column`` is given, the specified column
    will be removed from the returned frame and used only for indexing.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    channels : list of str, optional
        Subset of column names to return.  If omitted, all columns are
        returned.
    time_column : str, optional
        Name of the column containing timestamps.  If provided this column
        is dropped from the output DataFrame and stored as the index.
    sep : str, optional
        Column separator (default ",").
    header : int or None, optional
        Row number to use as the header (0-indexed).  Pass ``None`` to
        indicate that the CSV does not have a header row.
    read_csv_kwargs : dict
        Additional keyword arguments passed directly to ``pandas.read_csv``.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing only the requested channels, indexed by time
        if ``time_column`` was provided.
    """
    df = pd.read_csv(file_path, sep=sep, header=header, **read_csv_kwargs)
    if time_column is not None and time_column in df.columns:
        time_idx = df.pop(time_column)
        df.index = pd.to_datetime(time_idx)
    if channels is not None:
        missing = [c for c in channels if c not in df.columns]
        if missing:
            raise ValueError(f"Channels {missing} not found in file")
        df = df[channels]
        
    # Fill NaNs in numeric columns using the mean of the last up to 5 previous measurements.
    # # If there are no previous valid samples, fill with 0.
    # num_cols = df.select_dtypes(include=[np.number]).columns
    # for col in num_cols:
    #     s = df[col]
    #     # rolling mean of the previous up to 5 samples (shift to exclude current row)
    #     prev_mean = s.shift(1).rolling(window=5, min_periods=1).mean()
    #     # replace NaNs with the computed previous mean
    #     s_filled = s.where(s.notna(), prev_mean)
    #     # any remaining NaNs (e.g., at start with no previous data) set to 0
    #     s_filled = s_filled.fillna(0.0)
    #     df[col] = s_filled
    return df


def detrend_signals(signals: np.ndarray, axis: int = -1) -> np.ndarray:
    """Remove linear trends from each time series.

    This function accepts a 2D array where rows correspond to different
    channels and columns correspond to time samples.  The linear trend is
    removed along the specified axis using ``scipy.signal.detrend``.

    Parameters
    ----------
    signals : ndarray, shape (n_signals, n_samples)
        Input time series data.
    axis : int, optional
        Axis along which to detrend (default: -1).

    Returns
    -------
    ndarray
        Detrended time series with the same shape as the input.
    """
    return _detrend(signals, axis=axis)


def compute_fft_magnitude(signals: np.ndarray, sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the single‑sided magnitude spectra of multiple time series.

    A real input signal of length ``n`` has a Hermitian Fourier transform,
    so only the first ``n//2 + 1`` bins contain unique information.  This
    function returns the frequencies (in Hz) and corresponding magnitude
    spectra for each input channel.

    Parameters
    ----------
    signals : ndarray, shape (n_signals, n_samples)
        Real‑valued time series data.
    sampling_rate : float
        Sampling frequency of the signals in Hertz.

    Returns
    -------
    freqs : ndarray, shape (n_freqs,)
        Array of frequency bins corresponding to the spectra.
    mags : ndarray, shape (n_signals, n_freqs)
        Magnitude spectra for each signal.
    """
    n_signals, n_samples = signals.shape
    # Compute FFT along the time axis
    fft_vals = np.fft.rfft(signals, axis=1)
    mags = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sampling_rate)
    return freqs, mags


def cosine_similarity_matrix(mags: np.ndarray) -> np.ndarray:
    """Compute the cosine similarity matrix between rows of ``mags``.

    Cosine similarity between vectors ``a`` and ``b`` is defined as
    ``(a·b) / (||a|| * ||b||)``.  This function vectorises the computation
    by normalising each row and taking a dot product with its transpose.

    Parameters
    ----------
    mags : ndarray, shape (n_signals, n_features)
        Matrix of magnitude spectra.

    Returns
    -------
    ndarray, shape (n_signals, n_signals)
        Cosine similarity between each pair of signals.  The diagonal will
        contain ones by construction.
    """
    # Normalise each row
    norms = np.linalg.norm(mags, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    mags_norm = mags / norms
    # Compute similarity matrix as dot product of row vectors
    sim_matrix = mags_norm @ mags_norm.T
    # Clip numerical noise outside [−1, 1]
    return np.clip(sim_matrix, -1.0, 1.0)


def select_candidate_pairs(sim_matrix: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
    """Return list of index pairs whose similarity exceeds a threshold.

    Since the cosine similarity matrix is symmetric and the diagonal is
    trivially 1, we only return pairs (i, j) with i < j and
    ``sim_matrix[i, j] >= threshold``.

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
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                pairs.append((i, j))
    return pairs


def compute_cwt(
    signal: np.ndarray,
    sampling_rate: float,
    f_min: float,
    f_max: float,
    n_freqs: int,
    n_threads: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the continuous wavelet transform using fcwt.

    The fcwt library maps frequency bands to scale parameters internally and
    returns a complex array of coefficients indexed by frequency and time.
    If fcwt is not installed, this function raises an informative error.

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
        Number of frequency bins between ``f_min`` and ``f_max``.
    n_threads : int, optional
        Number of threads passed to fcwt (default: 1).  Increase this on
        multi‑core machines to accelerate the transform.

    Returns
    -------
    freqs : ndarray, shape (n_freqs,)
        Analysis frequencies in Hertz.
    coeffs : ndarray, shape (n_freqs, n_samples)
        Complex wavelet coefficients.

    Raises
    ------
    ImportError
        If fcwt could not be imported.
    """
    # if not _HAVE_FCWT:
    #     raise ImportError(
    #         "fcwt is not available. Please install it or substitute another CWT implementation."
    #     )
        
     # ---- enforce types expected by the fcwt C++ binding ----
    sampling_rate   = int(round(sampling_rate))   # sampling rate must be int
    f_min = float(f_min)
    f_max = float(f_max)
    n_freqs   = int(n_freqs)
    n_threads = int(n_threads)

    # fcwt.cwt returns frequencies ascending from f_min to f_max
    freqs, coeffs = cwt(
        signal.astype(float),
        sampling_rate,
        f_min,
        f_max,
        n_freqs,
        nthreads=n_threads,
    )
    return freqs, coeffs


def _smooth_spectrum(data: np.ndarray, sigma_time: int = 2, sigma_freq: int = 2) -> np.ndarray:
    """Smooth a time–frequency spectrum using a simple convolution.

    Rather than pulling in heavy dependencies for Gaussian filtering, this
    helper uses a separable box filter as an approximation.  Increasing
    ``sigma_time`` and ``sigma_freq`` increases the degree of smoothing.

    Parameters
    ----------
    data : ndarray, shape (n_freqs, n_times)
        Input matrix to be smoothed.
    sigma_time : int, optional
        Half‑window size along the time axis (default: 2 samples).
    sigma_freq : int, optional
        Half‑window size along the frequency axis (default: 2 samples).

    Returns
    -------
    ndarray
        Smoothed data array of the same shape.
    """
    # Simple moving average along both axes
    if sigma_time <= 0 and sigma_freq <= 0:
        return data
    # Convolve along frequency axis
    if sigma_freq > 0:
        kernel_f = np.ones(2 * sigma_freq + 1) / (2 * sigma_freq + 1)
        data = np.apply_along_axis(lambda m: np.convolve(m, kernel_f, mode="same"), 0, data)
    # Convolve along time axis
    if sigma_time > 0:
        kernel_t = np.ones(2 * sigma_time + 1) / (2 * sigma_time + 1)
        data = np.apply_along_axis(lambda m: np.convolve(m, kernel_t, mode="same"), 1, data)
    return data


def wavelet_coherence(
    coeffs1: np.ndarray,
    coeffs2: np.ndarray,
    smooth_time: int = 2,
    smooth_freq: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the magnitude squared wavelet coherence between two signals.

    Given complex wavelet coefficient matrices ``coeffs1`` and ``coeffs2``
    (both of shape ``(n_freqs, n_times)``), this function computes the
    coherence matrix as

    .. math::

       R(t, f) = \frac{|S_{12}(t, f)|^2}{S_1(t, f) \, S_2(t, f)},

    where ``S_1`` and ``S_2`` are smoothed auto‑spectra and ``S_12`` is
    the smoothed cross‑spectrum.  The smoothing is performed using a
    simple separable moving average; adjust ``smooth_time`` and
    ``smooth_freq`` to control the smoothing along the time and frequency
    axes respectively.

    Parameters
    ----------
    coeffs1, coeffs2 : ndarray, shape (n_freqs, n_times)
        Complex wavelet coefficients of the two signals to compare.
    smooth_time : int, optional
        Smoothing half‑window in samples along the time axis (default: 2).
    smooth_freq : int, optional
        Smoothing half‑window in samples along the frequency axis (default: 2).

    Returns
    -------
    coherence : ndarray, shape (n_freqs, n_times)
        Magnitude squared wavelet coherence values between 0 and 1.
    phase : ndarray, shape (n_freqs, n_times)
        Phase difference (in radians) between the two signals at each time
        and frequency.
    """
    # Auto spectra
    S1 = np.abs(coeffs1) ** 2
    S2 = np.abs(coeffs2) ** 2
    # Cross spectrum
    S12 = coeffs1 * np.conj(coeffs2)
    # Smooth spectra and cross spectrum
    S1_smooth = _smooth_spectrum(S1, sigma_time=smooth_time, sigma_freq=smooth_freq)
    S2_smooth = _smooth_spectrum(S2, sigma_time=smooth_time, sigma_freq=smooth_freq)
    S12_smooth = _smooth_spectrum(np.abs(S12) ** 2, sigma_time=smooth_time, sigma_freq=smooth_freq)
    # Compute coherence
    denom = S1_smooth * S2_smooth
    # Avoid division by zero
    denom[denom == 0] = np.finfo(float).eps
    coherence = S12_smooth / denom
    # Clip to [0, 1] range for interpretability
    coherence = np.clip(coherence, 0.0, 1.0)
    # Compute phase difference
    phase = np.angle(S12)
    return coherence, phase


@dataclass
class ProbeLayout:
    """Container for sensor coordinates.

    A probe layout describes the physical arrangement of electrodes in a
    multi‑probe experiment.  Each probe has a vertical position along the
    y‑axis and houses two contact electrodes (A and B) separated along the
    x‑axis by a constant pitch.  Coordinates are stored in micrometres.
    """
    y_positions_um: np.ndarray
    pitch_um: int
    x_origin: float = 0.0
    z_origin: float = 0.0

    def sensor_positions(self) -> np.ndarray:
        """Return the 3D coordinates of all sensors.

        The coordinate system places the origin at (x_origin, 0, z_origin).
        The y positions correspond to the centres of the probes.  Each
        probe contains two sensors at x positions ``pitch_um`` and
        ``2*pitch_um`` relative to the origin.  The z‑coordinate is fixed
        and can be used for 3D visualisations.

        Returns
        -------
        ndarray of shape (n_probes*2, 3)
            Positions of sensors in micrometres.  Sensors are ordered
            (probe0_A, probe0_B, probe1_A, probe1_B, ...).
        """
        positions: List[Tuple[float, float, float]] = []
        for y in self.y_positions_um:
            # Sensor A
            positions.append((self.x_origin + 1 * self.pitch_um, y, self.z_origin))
            # Sensor B
            positions.append((self.x_origin + 2 * self.pitch_um, y, self.z_origin))
        return np.array(positions, dtype=float)


def build_probe_positions(
    probe_y_positions_cm: Iterable[float], pitch_mm: float = 5.0
) -> ProbeLayout:
    """Construct a ``ProbeLayout`` from human friendly units.

    Parameters
    ----------
    probe_y_positions_cm : iterable of float
        Distances of probes from the origin along the y‑axis in centimetres.
    pitch_mm : float, optional
        Lateral separation between sensors A and B in millimetres (default 5 mm).

    Returns
    -------
    ProbeLayout
        Layout object containing sensor positions in micrometres.
    """
    UM_PER_CM = 10_000
    UM_PER_MM = 1_000
    y_positions_um = np.array(list(probe_y_positions_cm)) * UM_PER_CM
    pitch_um = int(pitch_mm * UM_PER_MM)
    return ProbeLayout(y_positions_um=y_positions_um, pitch_um=pitch_um)


def compute_spatial_distances(layout: ProbeLayout) -> np.ndarray:
    """Compute pairwise Euclidean distances between all sensors.

    Parameters
    ----------
    layout : ProbeLayout
        Layout specifying the sensor coordinates.

    Returns
    -------
    ndarray of shape (n_sensors, n_sensors)
        Matrix of distances in micrometres.  The matrix is symmetric with
        zeros on the diagonal.
    """
    coords = layout.sensor_positions()
    n = coords.shape[0]
    dist_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            dist_matrix[i, j] = dist_matrix[j, i] = dist
    return dist_matrix


def estimate_propagation_delays(
    phase: np.ndarray,
    freqs: np.ndarray,
    distances: float | np.ndarray,
    sampling_rate: float,
    wrap_phase: bool = True,
) -> np.ndarray:
    """Estimate delay between sensors from phase differences.

    This function provides a simplistic estimate of propagation time
    between sensors by relating the phase difference at each frequency
    to a time delay via ``Δt = Δφ / (2π f)``.  The result is then
    normalised by the distance to give a velocity estimate.  Note that
    this assumes a linear phase model and should be interpreted with
    care.  Users may choose to wrap phase differences into [−π, π] or
    leave them unwrapped depending on the analysis context.

    Parameters
    ----------
    phase : ndarray, shape (n_freqs, n_times)
        Phase difference array obtained from ``wavelet_coherence``.
    freqs : ndarray, shape (n_freqs,)
        Frequencies corresponding to the rows of ``phase``.
    distances : float or ndarray
        Distance in micrometres between the two sensors under consideration.
    sampling_rate : float
        Sampling frequency used to compute time axis.  Required to
        convert sample indices into seconds if later used for further
        analysis.
    wrap_phase : bool, optional
        Whether to wrap phase values into the principal range [−π, π] prior
        to computing delays (default: True).

    Returns
    -------
    ndarray, shape (n_freqs, n_times)
        Estimated propagation velocities in micrometres per second.  Positive
        values indicate that the first signal leads the second.
    """
    # Optionally wrap phase to [−π, π]
    ph = phase.copy()
    if wrap_phase:
        ph = (ph + np.pi) % (2 * np.pi) - np.pi
    # Compute delay at each time/frequency: Δt = Δφ/(2πf)
    # Avoid division by zero for DC component
    denom = 2 * np.pi * freqs[:, None]
    denom[denom == 0] = np.finfo(float).eps
    delays = ph / denom  # in seconds
    # Convert distances to metres for velocity calculation
    # Distances provided in micrometres; 1e-6 metre per micrometre
    distances_m = np.asarray(distances) * 1e-6
    # If distances is a scalar, broadcast to the shape of delays
    if np.isscalar(distances_m):
        distances_m = float(distances_m)
    velocities = distances_m / delays
    return velocities


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

    This function creates a SQLite database containing multiple tables:

    - **sensors**: sensor id, label, and physical coordinates
    - **spectral_features**: magnitude spectra for each sensor (flattened)
    - **similarity_matrix**: pairwise cosine similarity values
    - **candidate_pairs**: list of pairs exceeding the threshold
    - **coherence**: for each candidate pair, stores coherence and phase
      matrices as blobs along with corresponding frequency vectors

    Parameters
    ----------
    db_path : str
        Path to the output SQLite database.  The file will be created
        or overwritten.
    spectral_matrix : ndarray, shape (n_signals, n_freqs)
        Magnitude spectra computed from the FFT.
    similarity_matrix : ndarray, shape (n_signals, n_signals)
        Pairwise cosine similarity matrix.
    candidate_pairs : list of tuple of int
        List of index pairs whose similarity exceeds the threshold.
    coherence_results : dict
        Mapping from candidate pair to a dict containing keys
        'coherence', 'phase' and optionally other metrics.  Each value
        must be an ndarray with shape (n_freqs, n_times).
    freqs_fft : ndarray
        Frequency vector used for the FFT magnitude spectra.
    freqs_cwt : dict
        Mapping from candidate pair to an ndarray of frequencies used
        in the wavelet coherence analysis.
    sensor_labels : list of str
        Names or labels of the sensors, in the order corresponding to
        rows of the input data.
    layout : ProbeLayout
        Layout object containing sensor coordinates.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Create tables
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sensors (
            id INTEGER PRIMARY KEY,
            label TEXT,
            x REAL,
            y REAL,
            z REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS spectral_features (
            sensor_id INTEGER,
            freqs BLOB,
            magnitudes BLOB,
            FOREIGN KEY(sensor_id) REFERENCES sensors(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS similarity_matrix (
            sensor_i INTEGER,
            sensor_j INTEGER,
            similarity REAL,
            FOREIGN KEY(sensor_i) REFERENCES sensors(id),
            FOREIGN KEY(sensor_j) REFERENCES sensors(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS candidate_pairs (
            sensor_i INTEGER,
            sensor_j INTEGER,
            FOREIGN KEY(sensor_i) REFERENCES sensors(id),
            FOREIGN KEY(sensor_j) REFERENCES sensors(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS coherence (
            sensor_i INTEGER,
            sensor_j INTEGER,
            freqs BLOB,
            coherence BLOB,
            phase BLOB,
            FOREIGN KEY(sensor_i) REFERENCES sensors(id),
            FOREIGN KEY(sensor_j) REFERENCES sensors(id)
        )
        """
    )
    conn.commit()
    # Insert sensors
    coords = layout.sensor_positions()
    for idx, (label, coord) in enumerate(zip(sensor_labels, coords)):
        cur.execute(
            "INSERT OR REPLACE INTO sensors (id, label, x, y, z) VALUES (?, ?, ?, ?, ?)",
            (idx, label, float(coord[0]), float(coord[1]), float(coord[2])),
        )
    # Insert spectral features
    for idx, mag in enumerate(spectral_matrix):
        cur.execute(
            "INSERT INTO spectral_features (sensor_id, freqs, magnitudes) VALUES (?, ?, ?)",
            (idx, freqs_fft.tobytes(), mag.tobytes()),
        )
    # Insert similarity matrix entries
    n = similarity_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            cur.execute(
                "INSERT INTO similarity_matrix (sensor_i, sensor_j, similarity) VALUES (?, ?, ?)",
                (i, j, float(similarity_matrix[i, j])),
            )
    # Insert candidate pairs
    for (i, j) in candidate_pairs:
        cur.execute(
            "INSERT INTO candidate_pairs (sensor_i, sensor_j) VALUES (?, ?)", (i, j)
        )
    # Insert coherence matrices
    for (i, j), results in coherence_results.items():
        coh = results.get("coherence")
        ph = results.get("phase")
        freqs_pair = freqs_cwt[(i, j)]
        cur.execute(
            "INSERT INTO coherence (sensor_i, sensor_j, freqs, coherence, phase) VALUES (?, ?, ?, ?, ?)",
            (i, j, freqs_pair.tobytes(), coh.tobytes(), ph.tobytes()),
        )
    conn.commit()
    conn.close()


__all__ = [
    "setup_logging",
    "load_csv_data",
    "detrend_signals",
    "compute_fft_magnitude",
    "cosine_similarity_matrix",
    "select_candidate_pairs",
    "compute_cwt",
    "wavelet_coherence",
    "ProbeLayout",
    "build_probe_positions",
    "compute_spatial_distances",
    "estimate_propagation_delays",
    "save_results_to_db",
]