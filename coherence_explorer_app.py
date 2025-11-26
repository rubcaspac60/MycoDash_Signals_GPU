"""
coherence_explorer_app.py

Interactive coherence explorer for multi-probe fungal signals.

Features:
- Similarity matrix heatmap (click to select pair)
- Peak slider to move through manually defined peaks
- Wavelet coherence (time–frequency heatmap) for selected pair & peak
- Raw time-series for the selected pair in the selected peak window
- 3D spatial layout of probes with selected pair highlighted
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any

import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State

from GPU_Acc.signal_analysis_pipeline import (
    load_csv_data,
    detrend_signals,
    compute_fft_magnitude,
    cosine_similarity_matrix,
    compute_cwt,
    wavelet_coherence,
    build_probe_positions,
    compute_spatial_distances,
)

# ---------------------------------------------------------------------
# 1) USER CONFIGURATION
# ---------------------------------------------------------------------

DATA_PATH = "Data/ExperimentalBarrel_29_09_2023_csv"
SAMPLING_RATE = 1000.0  # Hz

CHANNELS = [
    "1_Blue Ave. (V)", "1_White Ave. (V)",
    "2_Blue Ave. (V)", "2_White Ave. (V)",
    "3_Blue Ave. (V)", "3_White Ave. (V)",
]

# Peaks as (sample_start, sample_end)
PEAKS = [
    {"section": "colonization", "peak": "PW_Off", "time": (2168, 2661)},
    {"section": "colonization", "peak": "1",      "time": (2760, 2825)},
    {"section": "colonization", "peak": "2",      "time": (3020, 3034)},
    {"section": "colonization", "peak": "3",      "time": (3038, 3094)},
    {"section": "colonization", "peak": "2a",     "time": (3020, 3094)},
    {"section": "colonization", "peak": "4",      "time": (3739, 3833)},
    {"section": "colonization", "peak": "5",      "time": (4297, 4361)},
    {"section": "colonization", "peak": "6",      "time": (6212, 6274)},
    {"section": "colonization", "peak": "7",      "time": (6559, 6580)},
    {"section": "colonization", "peak": "8",      "time": (7365, 7460)},
    {"section": "colonization", "peak": "9",      "time": (8420, 8438)},
]

# CWT frequency range
CWT_F_MIN = 0.1
CWT_F_MAX = 50.0
CWT_N_FREQS = 128

# ---------------------------------------------------------------------
# 2) LOAD DATA & PRECOMPUTE GLOBALS
# ---------------------------------------------------------------------

# Load CSV and get signals as (n_samples, n_channels)
df = load_csv_data(DATA_PATH, channels=CHANNELS)
raw_signals = df[CHANNELS].values  # (n_samples, n_channels)
# Detrend
signals = detrend_signals(raw_signals)  # same shape

# For spectral similarity, use channels-first arrangement
signals_ch_first = signals.T  # (n_channels, n_samples)

freqs_fft, mags = compute_fft_magnitude(signals_ch_first, sampling_rate=SAMPLING_RATE)
sim_matrix = cosine_similarity_matrix(mags)  # (n_channels, n_channels)

# Build probe layout and distances
# Your build_probe_positions likely accepts probe_y_positions + pitch_mm,
# but if you already have a layout builder wrapper, adapt this call:
probe_y_positions_cm = [5, 15, 25]
layout, layout_labels = build_probe_positions(probe_y_positions_cm, pitch_mm=5.0)
distances = compute_spatial_distances(layout)

# Map channel labels -> layout indices (assumes same naming)
label_to_layout_idx = {lbl: idx for idx, lbl in enumerate(layout.labels)}


# ---------------------------------------------------------------------
# 3) HELPER FUNCTIONS TO BUILD FIGURES
# ---------------------------------------------------------------------

def make_similarity_heatmap() -> go.Figure:
    """Plot the similarity matrix with CHANNELS on both axes."""
    fig = px.imshow(
        sim_matrix,
        x=CHANNELS,
        y=CHANNELS,
        color_continuous_scale="Viridis",
        zmin=0,
        zmax=1,
        aspect="auto",
    )
    fig.update_layout(
        title="Spectral Cosine Similarity (click a cell to select a pair)",
        xaxis_title="Channel",
        yaxis_title="Channel",
    )
    return fig


def get_peak_window(peak_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return time vector and windowed signals for the given peak index.

    Returns
    -------
    t_win : ndarray, shape (n_samples_win,)
        Time in seconds.
    sig_win_ch_first : ndarray, shape (n_channels, n_samples_win)
        Detrended signals for that window, channels-first.
    """
    pk = PEAKS[peak_idx]
    s0, s1 = pk["time"]
    sig_win = signals[s0:s1, :]          # (n_samples_win, n_channels)
    sig_win_ch_first = sig_win.T         # (n_channels, n_samples_win)
    n_samples_win = sig_win.shape[0]
    t_win = np.arange(n_samples_win) / SAMPLING_RATE
    return t_win, sig_win_ch_first


def make_timeseries_figure(peak_idx: int, ch_i: int, ch_j: int) -> go.Figure:
    t_win, sig_win = get_peak_window(peak_idx)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t_win,
        y=sig_win[ch_i],
        mode="lines",
        name=CHANNELS[ch_i],
    ))
    fig.add_trace(go.Scatter(
        x=t_win,
        y=sig_win[ch_j],
        mode="lines",
        name=CHANNELS[ch_j],
    ))
    pk = PEAKS[peak_idx]
    fig.update_layout(
        title=f"Raw signals in peak '{pk['peak']}' "
              f"({CHANNELS[ch_i]} vs {CHANNELS[ch_j]})",
        xaxis_title="Time (s)",
        yaxis_title="Voltage (a.u.)",
        template="plotly_white",
    )
    return fig


def make_coherence_figure(peak_idx: int, ch_i: int, ch_j: int) -> go.Figure:
    """Compute and plot wavelet coherence for selected pair + peak."""
    _, sig_win = get_peak_window(peak_idx)  # (n_channels, n_samples_win)
    sig_i = sig_win[ch_i]
    sig_j = sig_win[ch_j]

    freqs_cwt, coeffs_i = compute_cwt(
        sig_i,
        sampling_rate=SAMPLING_RATE,
        f_min=CWT_F_MIN,
        f_max=CWT_F_MAX,
        n_freqs=CWT_N_FREQS,
    )
    _, coeffs_j = compute_cwt(
        sig_j,
        sampling_rate=SAMPLING_RATE,
        f_min=CWT_F_MIN,
        f_max=CWT_F_MAX,
        n_freqs=CWT_N_FREQS,
    )

    coherence, phase = wavelet_coherence(coeffs_i, coeffs_j)  # (n_freqs, n_times)
    n_times = coherence.shape[1]
    t_coh = np.linspace(0, coherence.shape[1] / SAMPLING_RATE, n_times)

    fig = go.Figure(data=go.Heatmap(
        x=t_coh,
        y=freqs_cwt,
        z=coherence,
        colorscale="Viridis",
        colorbar=dict(title="Coherence"),
    ))
    pk = PEAKS[peak_idx]
    fig.update_layout(
        title=f"Wavelet coherence in peak '{pk['peak']}' "
              f"({CHANNELS[ch_i]} vs {CHANNELS[ch_j]})",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
    )
    return fig


def make_3d_layout_figure(ch_i: int, ch_j: int) -> go.Figure:
    """
    3D spatial layout: highlight selected pair, show all probes.

    Assumes layout has attributes .positions (Nx3) and .labels.
    """
    positions = layout.positions  # ndarray (n_sensors, 3)
    xs = positions[:, 0]
    ys = positions[:, 1]
    zs = positions[:, 2]

    fig = go.Figure()

    # All points
    fig.add_trace(go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers+text",
        marker=dict(size=6),
        text=layout.labels,
        textposition="top center",
        name="Probes",
    ))

    # Highlight the selected pair with bigger markers + connecting line
    # Map CHANNELS indices to layout indices (assuming your labels match)
    ch_i_label = CHANNELS[ch_i]
    ch_j_label = CHANNELS[ch_j]

    if ch_i_label in label_to_layout_idx and ch_j_label in label_to_layout_idx:
        i_idx = label_to_layout_idx[ch_i_label]
        j_idx = label_to_layout_idx[ch_j_label]

        xi, yi, zi = positions[i_idx]
        xj, yj, zj = positions[j_idx]

        # Highlighted markers
        fig.add_trace(go.Scatter3d(
            x=[xi, xj],
            y=[yi, yj],
            z=[zi, zj],
            mode="markers",
            marker=dict(size=10, color="red"),
            name="Selected pair",
        ))

        # Connecting line
        fig.add_trace(go.Scatter3d(
            x=[xi, xj],
            y=[yi, yj],
            z=[zi, zj],
            mode="lines",
            line=dict(width=5, color="red"),
            name="Connection",
        ))

    fig.update_layout(
        title="3D probe layout (selected pair highlighted)",
        scene=dict(
            xaxis_title="X (µm)",
            yaxis_title="Y (µm)",
            zaxis_title="Z (µm)",
        ),
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------
# 4) DASH APP
# ---------------------------------------------------------------------

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H2("Interactive Coherence Explorer"),

        html.Div(
            [
                html.Div(
                    [
                        html.H4("1. Similarity matrix"),
                        dcc.Graph(
                            id="similarity-heatmap",
                            figure=make_similarity_heatmap(),
                        ),
                        html.Div(
                            "Click on a cell (i,j) to choose a channel pair.",
                            style={"fontSize": "0.9em", "color": "#555"},
                        ),
                        html.Br(),
                        html.Label("2. Peak selection"),
                        dcc.Slider(
                            id="peak-slider",
                            min=0,
                            max=len(PEAKS) - 1,
                            step=1,
                            value=0,
                            marks={i: pk["peak"] for i, pk in enumerate(PEAKS)},
                        ),
                    ],
                    style={"width": "45%", "display": "inline-block", "verticalAlign": "top"},
                ),
                html.Div(
                    [
                        html.H4("3. 3D probe layout"),
                        dcc.Graph(id="layout-3d"),
                    ],
                    style={"width": "54%", "display": "inline-block", "verticalAlign": "top"},
                ),
            ]
        ),

        html.Hr(),
        html.Div(
            [
                html.Div(
                    [
                        html.H4("4. Raw signals in selected peak"),
                        dcc.Graph(id="timeseries-plot"),
                    ],
                    style={"width": "49%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        html.H4("5. Wavelet coherence (time–frequency)"),
                        dcc.Graph(id="coherence-plot"),
                    ],
                    style={"width": "49%", "display": "inline-block"},
                ),
            ]
        ),

        # Hidden store for currently-selected pair (fallback if no click yet)
        dcc.Store(id="selected-pair", data={"i": 0, "j": 1}),
    ]
)


# ---------------------------------------------------------------------
# 5) CALLBACKS
# ---------------------------------------------------------------------

@app.callback(
    Output("selected-pair", "data"),
    Input("similarity-heatmap", "clickData"),
    State("selected-pair", "data"),
)
def update_selected_pair(clickData, current_pair):
    """
    Update selected pair (i,j) when user clicks on similarity heatmap.
    If no click, keep the current pair.
    """
    if clickData is None:
        return current_pair

    # clickData structure: clickData["points"][0]["x"], ["y"]
    pt = clickData["points"][0]
    x_label = pt["x"]
    y_label = pt["y"]

    try:
        i = CHANNELS.index(y_label)  # rows = y
        j = CHANNELS.index(x_label)  # cols = x
    except ValueError:
        return current_pair

    return {"i": i, "j": j}


@app.callback(
    Output("timeseries-plot", "figure"),
    Output("coherence-plot", "figure"),
    Output("layout-3d", "figure"),
    Input("peak-slider", "value"),
    Input("selected-pair", "data"),
)
def update_plots(peak_idx, selected_pair):
    i = selected_pair["i"]
    j = selected_pair["j"]

    fig_ts = make_timeseries_figure(peak_idx, i, j)
    fig_coh = make_coherence_figure(peak_idx, i, j)
    fig_layout = make_3d_layout_figure(i, j)

    return fig_ts, fig_coh, fig_layout


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------

if __name__ == "__main__":
    app.run_server(debug=True)
