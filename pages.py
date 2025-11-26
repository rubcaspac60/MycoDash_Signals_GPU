"""
Streamlit page implementations.

Each function defined in this module corresponds to a page in the
Streamlit application.  Pages share data via ``st.session_state`` so
that expensive computations need not be repeated when navigating
between pages.

The pages are deliberately simple: they expose only the essential
parameters for the analysis and display the results with basic plots.
You can extend or customise the pages to suit your own datasets and
visualisation preferences.  See ``app.py`` for how pages are wired
together.
"""

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import streamlit as st
from typing import List, Tuple, Dict

import plotly.express as px

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
    multi_band_similarity,
    phase_drift_analysis,
)


###############################################################################
# Page helpers


def _ensure_data() -> bool:
    """Check that signals and metadata have been loaded into session state."""
    return all(k in st.session_state for k in ("signals", "sampling_rate", "channel_names"))


###############################################################################
# Page implementations


def upload_page() -> None:
    """Page for uploading a CSV file and selecting channels to analyse."""
    st.header("Step 1: Upload Data")
    st.markdown(
        "Upload a CSV file containing time series data.  Each column "
        "should correspond to a sensor.  Optionally specify a time index column "
        "to be ignored.  Once loaded, the selected channels will be stored in "
        "the session for downstream analysis."
    )
    file = st.file_uploader("CSV file", type=["csv"])
    if file is None:
        st.info("Awaiting CSV upload…")
        return
    # Read the CSV into a DataFrame
    try:
        df = pd.read_csv(file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return
    st.write("Preview of data:")
    st.dataframe(df.head())
    # Select channels
    columns = list(df.columns)
    time_col = st.selectbox("Time column (optional)", options=["None"] + columns, index=0)
    available = [c for c in columns if c != time_col]
    selected = st.multiselect("Select channels to analyse", options=available, default=available)
    sampling_rate = st.number_input("Sampling rate (Hz)", min_value=0.1, value=1000.0, step=1.0)
    if st.button("Load signals"):
        if not selected:
            st.warning("Please select at least one channel.")
            return
        signals = df[selected].to_numpy().T  # shape (n_channels, n_samples)
        # optionally drop time column
        st.session_state["signals"] = signals.astype(np.float32)
        st.session_state["sampling_rate"] = float(sampling_rate)
        st.session_state["channel_names"] = selected
        st.success(f"Loaded {signals.shape[0]} channels with {signals.shape[1]} samples.")


def analysis_page() -> None:
    """Page for computing spectral features, similarity and candidate pairs."""
    st.header("Step 2: Spectral Analysis")
    if not _ensure_data():
        st.info("Please upload data on the 'Data Upload' page first.")
        return
    signals = st.session_state["signals"]
    fs = st.session_state["sampling_rate"]
    # Detrend signals
    if st.button("Detrend signals", key="detrend"):
        with st.spinner("Removing trends…"):
            signals = detrend_signals(signals)
            st.session_state["signals"] = signals
        st.success("Detrending complete.")
    # Compute FFT and similarity
    if st.button("Compute spectral similarity", key="fft"):
        with st.spinner("Computing FFT magnitudes…"):
            freqs_fft, mags = compute_fft_magnitude(signals, sampling_rate=fs)
            st.session_state["freqs_fft"] = freqs_fft
            st.session_state["mags"] = mags
        with st.spinner("Computing cosine similarity…"):
            sim = cosine_similarity_matrix(mags)
            st.session_state["sim_matrix"] = sim
        st.success("Spectral analysis complete.")
    # Display similarity matrix
    if "sim_matrix" in st.session_state:
        sim = st.session_state["sim_matrix"]
        fig = px.imshow(sim, text_auto=True, aspect="auto", color_continuous_scale="Viridis")
        fig.update_layout(title="Cosine similarity between channels", xaxis_title="Channel", yaxis_title="Channel")
        st.plotly_chart(fig, use_container_width=True)
        threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.8, 0.01)
        pairs = select_candidate_pairs(sim, threshold=threshold)
        st.session_state["candidate_pairs"] = pairs
        st.write(f"Selected {len(pairs)} candidate pairs above threshold.")
        if pairs:
            st.table(pd.DataFrame(pairs, columns=["Channel i", "Channel j"]))


def coherence_page() -> None:
    """Page for computing and visualising wavelet coherence on candidate pairs."""
    st.header("Step 3: Wavelet Coherence")
    if not _ensure_data():
        st.info("Please upload data and run spectral analysis first.")
        return
    if "candidate_pairs" not in st.session_state or not st.session_state["candidate_pairs"]:
        st.info("No candidate pairs found.  Adjust the similarity threshold on the previous page.")
        return
    signals = st.session_state["signals"]
    fs = st.session_state["sampling_rate"]
    pairs = st.session_state["candidate_pairs"]
    # Analysis parameters
    st.subheader("CWT parameters")
    f_min = st.number_input("Minimum frequency (Hz)", min_value=0.1, value=1.0, step=0.1)
    f_max = st.number_input("Maximum frequency (Hz)", min_value=f_min + 0.1, value=50.0, step=0.5)
    n_freqs = st.number_input("Number of frequencies", min_value=8, max_value=256, value=32, step=1)
    if st.button("Compute CWT and coherence", key="cwt"):
        coherence_results: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
        freqs_cwt_dict: Dict[Tuple[int, int], np.ndarray] = {}
        with st.spinner("Computing wavelet transforms on GPU…"):
            for (i, j) in pairs:
                freqs_cwt, coeffs_i = compute_cwt(signals[i], fs, f_min, f_max, int(n_freqs))
                _,         coeffs_j = compute_cwt(signals[j], fs, f_min, f_max, int(n_freqs))
                coh, phase = wavelet_coherence(coeffs_i, coeffs_j)
                coherence_results[(i, j)] = {"coherence": coh, "phase": phase}
                freqs_cwt_dict[(i, j)] = freqs_cwt
        st.session_state["coherence_results"] = coherence_results
        st.session_state["freqs_cwt_dict"] = freqs_cwt_dict
        st.success("Wavelet coherence computed for all pairs.")
    # Visualise coherence for a selected pair
    if "coherence_results" in st.session_state:
        pair_options = [f"{i}–{j}" for (i, j) in pairs]
        selected_label = st.selectbox("Select a pair to visualise", options=pair_options)
        if selected_label:
            idx = pair_options.index(selected_label)
            pair = pairs[idx]
            coh = st.session_state["coherence_results"][pair]["coherence"]
            freqs_cwt = st.session_state["freqs_cwt_dict"][pair]
            # Plot heatmap
            fig = px.imshow(
                coh,
                aspect="auto",
                origin="lower",
                x=np.arange(coh.shape[1]),
                y=freqs_cwt,
                color_continuous_scale="Viridis",
            )
            fig.update_layout(
                title=f"Wavelet coherence for channels {pair}",
                xaxis_title="Time (samples)",
                yaxis_title="Frequency (Hz)"
            )
            st.plotly_chart(fig, use_container_width=True)


def propagation_page() -> None:
    """Page for estimating propagation velocities from wavelet phases."""
    st.header("Step 4: Propagation Analysis")
    if not _ensure_data():
        st.info("Please upload data and run previous analyses first.")
        return
    if "coherence_results" not in st.session_state or not st.session_state["coherence_results"]:
        st.info("No coherence results available.  Compute coherence on the previous page.")
        return
    # Assume a simple vertical probe layout for demonstration
    st.subheader("Sensor geometry")
    n_channels = st.session_state["signals"].shape[0]
    default_positions = [float(i) * 1.0 for i in range(n_channels)]  # 1 cm apart
    positions_str = st.text_input(
        "Enter vertical positions for each sensor (cm, comma separated)",
        value=", ".join(str(p) for p in default_positions)
    )
    try:
        positions_cm = [float(v.strip()) for v in positions_str.split(",") if v.strip()]
    except Exception:
        st.error("Invalid positions.  Please enter comma separated numbers.")
        return
    if len(positions_cm) != n_channels:
        st.error(f"Please provide exactly {n_channels} positions.")
        return
    layout = build_probe_positions(positions_cm)
    # Select pair to analyse
    pairs = st.session_state.get("candidate_pairs", [])
    pair_options = [f"{i}–{j}" for (i, j) in pairs]
    if not pair_options:
        st.info("No candidate pairs available.")
        return
    selected_label = st.selectbox("Select a pair", options=pair_options)
    idx = pair_options.index(selected_label)
    pair = pairs[idx]
    # Retrieve coherence result
    coh = st.session_state["coherence_results"][pair]["coherence"]
    phase = st.session_state["coherence_results"][pair]["phase"]
    freqs_cwt = st.session_state["freqs_cwt_dict"][pair]
    fs = st.session_state["sampling_rate"]
    # Compute distance between sensors in micrometres
    coords = layout.sensor_positions()
    i, j = pair
    dist_um = np.linalg.norm(coords[i] - coords[j])
    # Estimate velocities
    velocities = estimate_propagation_delays(phase, freqs_cwt, dist_um, sampling_rate=fs)
    # Plot dominant velocity over time
    dom_vel = np.zeros(velocities.shape[1])
    for t in range(velocities.shape[1]):
        # choose frequency with highest coherence
        freq_idx = np.argmax(coh[:, t])
        dom_vel[t] = velocities[freq_idx, t]
    fig = px.line(y=dom_vel, x=np.arange(len(dom_vel)))
    fig.update_layout(
        title=f"Dominant propagation velocity for channels {pair}",
        xaxis_title="Time (samples)",
        yaxis_title="Velocity (μm/s)"
    )
    st.plotly_chart(fig, use_container_width=True)


def band_similarity_page() -> None:
    """Page for computing multi‑band similarity metrics."""
    st.header("Step 5: Frequency Band Similarity")
    if not _ensure_data():
        st.info("Please upload data on the 'Data Upload' page first.")
        return
    signals = st.session_state["signals"]
    fs = st.session_state["sampling_rate"]
    # Specify bands
    st.markdown(
        "Define one or more frequency bands for analysis.  Enter values in Hz as `low, high` and separate multiple bands by semicolons."
    )
    band_input = st.text_input("Bands", value="0.5,4;4,8;8,12;12,30;30,80")
    try:
        band_list: List[Tuple[float, float]] = []
        bands_str = [b.strip() for b in band_input.split(";") if b.strip()]
        for b in bands_str:
            low_str, high_str = [s.strip() for s in b.split(",")]
            low = float(low_str)
            high = float(high_str)
            if low >= high:
                raise ValueError(f"Low cutoff {low} must be < high cutoff {high}")
            band_list.append((low, high))
    except Exception as e:
        st.error(f"Invalid band specification: {e}")
        return
    if st.button("Compute band similarity"):
        with st.spinner("Computing multi‑band similarity matrices…"):
            sim_results = multi_band_similarity(signals, sampling_rate=fs, bands=band_list)
        st.session_state["band_similarity_results"] = sim_results
        st.success("Band similarity computation complete.")
    # Display results if available
    if "band_similarity_results" in st.session_state:
        results: Dict[Tuple[float, float], Dict[str, np.ndarray]] = st.session_state["band_similarity_results"]
        for band, matrices in results.items():
            low, high = band
            st.subheader(f"Band {low}–{high} Hz")
            fft_sim = matrices["fft_similarity"]
            time_corr = matrices["time_correlation"]
            # Plot FFT similarity
            fig_fft = px.imshow(
                fft_sim,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Viridis",
            )
            fig_fft.update_layout(
                title=f"FFT cosine similarity ({low}–{high} Hz)",
                xaxis_title="Channel",
                yaxis_title="Channel",
            )
            st.plotly_chart(fig_fft, use_container_width=True)
            # Plot time‑domain correlation
            fig_corr = px.imshow(
                time_corr,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1.0,
                zmax=1.0,
            )
            fig_corr.update_layout(
                title=f"Time‑domain correlation ({low}–{high} Hz)",
                xaxis_title="Channel",
                yaxis_title="Channel",
            )
            st.plotly_chart(fig_corr, use_container_width=True)


def drift_analysis_page() -> None:
    """Page for computing phase drift across time for a selected pair."""
    st.header("Step 6: Drift Analysis")
    if not _ensure_data():
        st.info("Please upload data and perform coherence analysis first.")
        return
    # Need coherence results to compute drift
    if "coherence_results" not in st.session_state or not st.session_state["coherence_results"]:
        st.info("No coherence results available.  Compute coherence on the 'Coherence' page.")
        return
    # List pairs and select one
    pairs = st.session_state.get("candidate_pairs", [])
    pair_options = [f"{i}–{j}" for (i, j) in pairs]
    if not pair_options:
        st.info("No candidate pairs available.")
        return
    selected_label = st.selectbox("Select a pair for drift analysis", options=pair_options)
    idx = pair_options.index(selected_label)
    pair = pairs[idx]
    # Retrieve phase and frequency vectors
    phase = st.session_state["coherence_results"][pair]["phase"]
    freqs_cwt = st.session_state["freqs_cwt_dict"][pair]
    fs = st.session_state["sampling_rate"]
    dt = 1.0 / fs
    # Compute drift (seconds per second) for each frequency
    with st.spinner("Computing phase drift…"):
        drift_vals = phase_drift_analysis(phase, freqs_cwt, dt)
    # Plot drift as a function of frequency
    fig = px.line(x=freqs_cwt, y=drift_vals)
    fig.update_layout(
        title=f"Phase drift for channels {pair}",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Drift (s/s)",
    )
    st.plotly_chart(fig, use_container_width=True)
    # Provide summary statistics
    st.write("Mean drift:", float(np.nanmean(drift_vals)))
    st.write("Std drift:", float(np.nanstd(drift_vals)))