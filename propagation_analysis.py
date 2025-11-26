# propagation_analysis.py

import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List

from GPU_Acc.signal_analysis_pipeline import ProbeLayout


@dataclass
class PropagationResult:
    """Container for propagation analysis for a single pair."""
    pair: Tuple[int, int]                  # (ch_i, ch_j)
    freqs: np.ndarray                      # (n_freqs,)
    times: np.ndarray                      # (n_times,)
    coherence: np.ndarray                  # (n_freqs, n_times)
    phase: np.ndarray                      # (n_freqs, n_times)
    delay: np.ndarray                      # (n_freqs, n_times)
    velocity: np.ndarray                   # (n_freqs, n_times)
    mask: np.ndarray                       # bool mask (n_freqs, n_times) where valid
    distance_m: float
    direction_label: str                   # e.g. "upward", "downward", "same_level"


def phase_to_delay(
    phase: np.ndarray,
    freqs: np.ndarray,
    *,
    device: Optional[str] = None,
    use_torch: bool = False,
) -> np.ndarray:
    """
    Convert phase differences to time delays: Δt = phase / (2π f).

    This helper supports both NumPy and PyTorch backends.  When
    ``use_torch`` is ``True`` the computation will be performed on
    the specified ``device`` using PyTorch tensors, which can be
    advantageous for very large phase matrices on a CUDA‑enabled GPU.

    Parameters
    ----------
    phase : ndarray, shape (n_freqs, n_times)
        Phase in radians, typically in [-π, π].  If ``use_torch`` is
        set the array will be converted to a ``torch.Tensor`` on
        ``device``.
    freqs : ndarray, shape (n_freqs,)
        Frequencies in Hz.
    device : str, optional
        Device specifier for PyTorch (e.g. ``'cuda'`` or ``'cpu'``).
        Ignored when ``use_torch`` is ``False``.
    use_torch : bool, optional
        If ``True`` perform the division on the GPU via PyTorch.

    Returns
    -------
    delay : ndarray, shape (n_freqs, n_times)
        Time delays in seconds.  NaNs may appear when frequencies are
        zero.
    """
    # Always operate on copies to avoid modifying inputs in place
    phase_arr = np.asarray(phase)
    freqs_arr = np.asarray(freqs)
    if not use_torch:
        # Avoid division by zero at f=0 by using NaN
        freqs_safe = freqs_arr.copy()
        freqs_safe[freqs_safe == 0] = np.nan
        return phase_arr / (2.0 * np.pi * freqs_safe[:, None])
    # PyTorch backend
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for use_torch=True") from exc
    # Resolve device
    dev = torch.device(device) if device is not None else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    # Convert to tensors
    ph = torch.as_tensor(phase_arr, dtype=torch.float32, device=dev)
    fr = torch.as_tensor(freqs_arr, dtype=torch.float32, device=dev)
    # Replace zeros with NaN to avoid division by zero; PyTorch NaN will propagate
    fr_safe = torch.where(fr == 0.0, torch.full_like(fr, float('nan')), fr)
    # Reshape frequencies for broadcasting
    denom = 2.0 * math.pi * fr_safe[:, None]
    delay = ph / denom
    return delay.cpu().numpy()


def classify_direction(
    layout: ProbeLayout,
    ch_i: int,
    ch_j: int,
) -> str:
    """
    Classify vertical direction between two sensors based on their y coordinate.

    Returns one of: "upward", "downward", "same_level".
    """
    # Obtain sensor positions.  ProbeLayout defines a ``sensor_positions``
    # method that returns an array of shape (n_sensors, 3).  We avoid
    # relying on a non‑existent ``positions`` attribute for forward
    # compatibility.
    pos = layout.sensor_positions()
    yi = pos[ch_i, 1]
    yj = pos[ch_j, 1]

    if np.isclose(yi, yj):
        return "same_level"
    elif yj > yi:
        # j is "above" i (depending on your axis convention)
        return "upward"
    else:
        return "downward"


def compute_propagation_for_pair(
    ch_i: int,
    ch_j: int,
    freqs_cwt: np.ndarray,
    coherence: np.ndarray,
    phase: np.ndarray,
    distance_m: float,
    layout: ProbeLayout,
    coherence_threshold: float = 0.5,
    t_axis: Optional[np.ndarray] = None,
    *,
    device: Optional[str] = None,
    use_torch: bool = False,
) -> PropagationResult:
    """
    Compute time delays and velocities for a single channel pair.

    This function extends the basic NumPy implementation by optionally
    performing the calculations on a GPU via PyTorch.  When
    ``use_torch`` is ``True`` the phase–to–delay conversion and the
    subsequent velocity calculation are executed on the specified
    ``device``.  This can significantly accelerate analyses when
    processing hundreds of frequencies and thousands of time points.

    Parameters
    ----------
    ch_i, ch_j : int
        Indices of channels in CHANNELS list.
    freqs_cwt : ndarray, shape (n_freqs,)
        CWT / coherence frequency axis in Hz.
    coherence : ndarray, shape (n_freqs, n_times)
        Coherence magnitude (0..1).
    phase : ndarray, shape (n_freqs, n_times)
        Phase difference in radians.
    distance_m : float
        Distance between the two sensors in meters.
    layout : ProbeLayout
        Sensor layout with positions and labels.
    coherence_threshold : float, optional
        Only compute velocities where coherence >= threshold.
    t_axis : ndarray, optional
        Time axis in seconds, shape (n_times,). If None, indices are used.
    device : str, optional
        PyTorch device specifier (e.g. 'cuda' or 'cpu') used when
        ``use_torch`` is ``True``.
    use_torch : bool, optional
        If ``True``, use PyTorch to accelerate delay and velocity
        calculation on the given ``device``.

    Returns
    -------
    PropagationResult
        Dataclass containing the delay, velocity and masks for the pair.
    """
    # Convert inputs to arrays
    coherence = np.asarray(coherence)
    phase = np.asarray(phase)
    freqs_cwt = np.asarray(freqs_cwt)
    # 1) Δt from phase
    delay = phase_to_delay(phase, freqs_cwt, device=device, use_torch=use_torch)
    # 2) Velocity: v = d / Δt
    if use_torch:
        try:
            import torch  # type: ignore
        except ImportError:
            # fall back to NumPy if torch unavailable
            use_torch = False
    if use_torch:
        dev = torch.device(device) if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        d = torch.tensor(distance_m, dtype=torch.float32, device=dev)
        delays_t = torch.as_tensor(delay, dtype=torch.float32, device=dev)
        with torch.no_grad():
            velocity_t = d / delays_t
        velocity = velocity_t.cpu().numpy()
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            velocity = distance_m / delay
    # 3) Mask: where coherence is high and delay not NaN/inf
    mask = (
        (coherence >= coherence_threshold)
        & np.isfinite(delay)
        & (np.abs(delay) > 0.0)
    )
    # 4) Direction label (geometric)
    direction_label = classify_direction(layout, ch_i, ch_j)
    # 5) Time axis
    if t_axis is None:
        times = np.arange(coherence.shape[1], dtype=float)
    else:
        times = np.asarray(t_axis)
    return PropagationResult(
        pair=(ch_i, ch_j),
        freqs=freqs_cwt,
        times=times,
        coherence=coherence,
        phase=phase,
        delay=delay,
        velocity=velocity,
        mask=mask,
        distance_m=distance_m,
        direction_label=direction_label,
    )


def dominant_velocity_chirp(
    result: PropagationResult,
    f_min: float,
    f_max: float,
) -> Dict[str, np.ndarray]:
    """
    Track dominant velocity (and frequency) over time within a band.

    For each time point, find the frequency in [f_min, f_max] with:
        - highest coherence, and
        - valid velocity mask.

    Returns
    -------
    dict with keys:
        'time'     : (n_times,)
        'freq'     : (n_times,)
        'velocity' : (n_times,)
        'delay'    : (n_times,)
    """
    freqs = result.freqs
    coh = result.coherence
    vel = result.velocity
    delay = result.delay
    mask = result.mask

    band_mask = (freqs >= f_min) & (freqs <= f_max)
    if not np.any(band_mask):
        raise ValueError(f"No frequencies in band [{f_min}, {f_max}]")

    freqs_band = freqs[band_mask]
    coh_band = coh[band_mask, :]
    vel_band = vel[band_mask, :]
    delay_band = delay[band_mask, :]
    mask_band = mask[band_mask, :]

    n_times = coh_band.shape[1]
    dom_freq = np.full(n_times, np.nan)
    dom_vel = np.full(n_times, np.nan)
    dom_delay = np.full(n_times, np.nan)

    for t in range(n_times):
        valid = mask_band[:, t]
        if not np.any(valid):
            continue
        # pick index of max coherence within valid band
        idx = np.argmax(coh_band[valid, t])
        # map back to band indices
        valid_indices = np.where(valid)[0]
        f_idx = valid_indices[idx]

        dom_freq[t] = freqs_band[f_idx]
        dom_vel[t] = vel_band[f_idx, t]
        dom_delay[t] = delay_band[f_idx, t]

    return {
        "time": result.times,
        "freq": dom_freq,
        "velocity": dom_vel,
        "delay": dom_delay,
    }
