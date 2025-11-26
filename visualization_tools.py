"""
visualization_tools.py
======================

This module contains a collection of helper functions for visualising
the outputs of the signal and propagation analysis pipelines.  The
goal is to provide intuitive 2D and 3D representations of the time–
frequency coherence, phase, delays and propagation velocities, and
to accelerate rendering where possible by exploiting GPU resources
through the use of ``cupy`` and ``plotly``.  Most functions accept
either NumPy arrays or PyTorch tensors as input and internally
convert them as necessary for plotting.

Key features
------------

* **Heatmaps and surfaces:** Functions such as :func:`plot_coherence_heatmap`
  and :func:`plot_velocity_surface` display coherence or velocity
  matrices as either 2D heatmaps or 3D surfaces using Matplotlib or
  Plotly.  Plotly is particularly suitable for interactive 3D
  visualisations in the browser.
* **Animations:** The :func:`animate_velocity` routine can export a
  sequence of frames showing the evolution of velocity or delay over
  time for a given frequency band.  GIFs and MP4 videos can be
  produced directly from the function via ``matplotlib.animation`` or
  ``imageio``.
* **GPU support:** When ``cupy`` is available the underlying array
  operations (e.g. smoothing or interpolation) can be carried out on
  the GPU.  Plotting itself is done on the CPU but by reducing the
  computational burden prior to visualisation the overall pipeline can
  remain responsive for very large datasets.

Users are encouraged to integrate these functions into their
Streamlit applications.  Plotly figures can be passed directly to
``st.plotly_chart`` while Matplotlib figures can be rendered via
``st.pyplot``.  Animations can be written to temporary files and
served through ``st.video`` or ``st.image``.

Note
----
This file defines convenience functions but does not depend on the
rest of the project; it can be imported independently wherever
visualisation is needed.  All dependencies are optional: ``matplotlib``,
``plotly``, and ``imageio`` must be installed separately.  GPU
acceleration requires ``cupy`` which is not included by default.  If
``cupy`` is not available the functions will fall back to standard
NumPy.
"""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Any, Iterable, Optional, Tuple, Dict

try:
    import cupy as cp  # type: ignore
    _HAVE_CUPY = True
except ImportError:
    cp = None  # type: ignore
    _HAVE_CUPY = False

try:
    import plotly.graph_objs as go  # type: ignore
    _HAVE_PLOTLY = True
except ImportError:
    go = None  # type: ignore
    _HAVE_PLOTLY = False

try:
    import imageio  # type: ignore
    _HAVE_IMAGEIO = True
except ImportError:
    imageio = None  # type: ignore
    _HAVE_IMAGEIO = False


def _as_ndarray(x: Any) -> np.ndarray:
    """Convert a PyTorch tensor or cupy array to a NumPy array."""
    # handle torch
    try:
        import torch  # type: ignore
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    # handle cupy
    if _HAVE_CUPY and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def plot_coherence_heatmap(
    coherence: Any,
    freqs: Iterable[float],
    times: Iterable[float],
    title: str = "Wavelet Coherence",
    cmap: str = "viridis",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 4),
) -> plt.Axes:
    """Render a 2D heatmap of coherence values.

    Parameters
    ----------
    coherence : array-like, shape (n_freqs, n_times)
        Coherence matrix (values between 0 and 1).
    freqs : iterable of float
        Frequency axis corresponding to the rows.
    times : iterable of float
        Time axis corresponding to the columns.
    title : str, optional
        Title of the plot.
    cmap : str, optional
        Colour map used for the heatmap.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.  If ``None`` a new figure and axes
        will be created.
    figsize : tuple, optional
        Size of the figure in inches when ``ax`` is ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the heatmap.
    """
    C = _as_ndarray(coherence)
    F = np.asarray(freqs)
    T = np.asarray(times)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        C,
        aspect="auto",
        origin="lower",
        extent=[T[0], T[-1], F[0], F[-1]],
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Coherence")
    return ax


def plot_velocity_surface(
    velocity: Any,
    freqs: Iterable[float],
    times: Iterable[float],
    title: str = "Propagation Velocity",
    use_plotly: bool = False,
    **kwargs,
) -> Any:
    """Plot a 3D surface of propagation velocities.

    Either Matplotlib or Plotly will be used depending on the value of
    ``use_plotly``.  Plotly surfaces are interactive and better suited
    for web applications such as Streamlit.  Matplotlib surfaces are
    static but may be preferable in headless environments.

    Parameters
    ----------
    velocity : array-like, shape (n_freqs, n_times)
        Velocity matrix (in micrometres per second).
    freqs : iterable of float
        Frequency axis corresponding to the rows.
    times : iterable of float
        Time axis corresponding to the columns.
    title : str, optional
        Plot title.
    use_plotly : bool, optional
        Whether to generate a Plotly surface.  Defaults to ``False``.

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objs.Figure
        The created figure object.
    """
    V = _as_ndarray(velocity)
    F = np.asarray(freqs)
    T = np.asarray(times)
    if use_plotly and _HAVE_PLOTLY:
        fig = go.Figure(
            data=[
                go.Surface(
                    z=V,
                    x=T,
                    y=F,
                    colorscale="Viridis",
                    colorbar=dict(title="Velocity (µm/s)"),
                )
            ],
            layout=go.Layout(
                title=title,
                scene=dict(
                    xaxis_title="Time (s)",
                    yaxis_title="Frequency (Hz)",
                    zaxis_title="Velocity (µm/s)",
                ),
            ),
        )
        return fig
    # Fallback to Matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    T_grid, F_grid = np.meshgrid(T, F)
    surf = ax.plot_surface(
        T_grid,
        F_grid,
        V,
        cmap="viridis",
        **kwargs,
    )
    fig.colorbar(surf, shrink=0.5, aspect=10, label="Velocity (µm/s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_zlabel("Velocity (µm/s)")
    ax.set_title(title)
    return fig


def animate_velocity(
    velocity: Any,
    freqs: Iterable[float],
    times: Iterable[float],
    filename: str,
    fps: int = 10,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    **kwargs,
) -> None:
    """Create an animation of propagation velocity over time.

    Each frame of the animation corresponds to a single time point and
    shows the velocity as a function of frequency.  The result is
    saved either as a GIF or MP4 depending on the file extension of
    ``filename``.  Internally, either ``imageio`` or
    ``matplotlib.animation`` is used.  If ``imageio`` is not
    available, the fallback uses Matplotlib and may not support MP4
    output.

    Parameters
    ----------
    velocity : array-like, shape (n_freqs, n_times)
        Velocity matrix.
    freqs : iterable of float
        Frequency axis.
    times : iterable of float
        Time axis.
    filename : str
        Destination file path.  The extension determines the format.
    fps : int, optional
        Frames per second of the resulting animation.
    vmin, vmax : float, optional
        Colour limits.  If omitted, the min and max of ``velocity`` are used.
    kwargs : dict
        Additional keyword arguments passed to the underlying writer.
    """
    V = _as_ndarray(velocity)
    F = np.asarray(freqs)
    T = np.asarray(times)
    n_times = V.shape[1]
    if vmin is None:
        vmin = np.nanmin(V)
    if vmax is None:
        vmax = np.nanmax(V)
    # Determine if imageio is available and if file extension is gif/mp4
    ext = filename.split(".")[-1].lower()
    use_imageio = _HAVE_IMAGEIO and ext in {"gif", "mp4", "mov"}
    if use_imageio:
        frames = []
        for t_idx in range(n_times):
            fig, ax = plt.subplots(figsize=(6, 3))
            im = ax.plot(F, V[:, t_idx])
            ax.set_ylim(vmin, vmax)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Velocity (µm/s)")
            ax.set_title(f"Time = {T[t_idx]:.2f} s")
            fig.tight_layout()
            # Draw canvas and convert to array
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
            plt.close(fig)
        imageio.mimsave(filename, frames, fps=fps, **kwargs)
        return
    # Fallback: use matplotlib.animation
    import matplotlib.animation as animation  # type: ignore
    fig, ax = plt.subplots(figsize=(6, 3))
    line, = ax.plot(F, V[:, 0])
    ax.set_ylim(vmin, vmax)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Velocity (µm/s)")
    ax.set_title(f"Time = {T[0]:.2f} s")
    def update(frame_idx: int):
        line.set_ydata(V[:, frame_idx])
        ax.set_title(f"Time = {T[frame_idx]:.2f} s")
        return line,
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_times,
        interval=1000.0 / fps,
        blit=True,
    )
    if ext in {"gif", "mp4"}:
        Writer = animation.writers.get(ext, None)
        if Writer is None:
            Writer = animation.FFMpegWriter
        writer = Writer(fps=fps, **kwargs)
        anim.save(filename, writer=writer)
    else:
        # default to gif in fallback
        anim.save(filename, writer="imagemagick", fps=fps)
    plt.close(fig)


def plot_sensor_positions_3d(
    positions: Any,
    values: Optional[Any] = None,
    labels: Optional[Iterable[str]] = None,
    title: str = "Sensor positions",
    use_plotly: bool = True,
    **kwargs,
) -> Any:
    """
    Visualise 3D sensor positions as a scatter plot.

    This helper accepts an array of sensor coordinates of shape
    ``(n_sensors, 3)`` and optionally a vector of values to colour each
    marker.  When ``use_plotly`` is ``True`` the plot will be
    interactive and suitable for embedding in web applications such
    as Streamlit.  Otherwise a Matplotlib figure will be returned.

    Parameters
    ----------
    positions : array-like, shape (n_sensors, 3)
        Coordinates of sensors.  The axes correspond to X, Y, Z in
        micrometres or other units.
    values : array-like, optional
        Values to map to marker colour (e.g. velocity magnitude).  If
        provided, must have length ``n_sensors``.
    labels : iterable of str, optional
        Labels for each sensor.  Used for hover information in the
        interactive plot.
    title : str, optional
        Title of the plot.
    use_plotly : bool, optional
        Whether to create a Plotly figure.  If ``False`` a static
        Matplotlib figure is returned.
    kwargs : dict
        Additional keyword arguments passed to Plotly ``Scatter3d`` or
        Matplotlib.

    Returns
    -------
    figure
        Plotly or Matplotlib figure object.
    """
    P = _as_ndarray(positions)
    if values is not None:
        V = _as_ndarray(values)
        assert V.shape[0] == P.shape[0], "values must match positions"
    else:
        V = None
    if labels is not None:
        labels_list = list(labels)
    else:
        labels_list = [str(i) for i in range(P.shape[0])]
    # Use Plotly for interactive visualisation
    if use_plotly and _HAVE_PLOTLY:
        import plotly.graph_objs as go  # type: ignore
        marker_kwargs = {
            "size": 6,
            "opacity": 0.8,
        }
        if V is not None:
            marker_kwargs["color"] = V
            marker_kwargs["colorscale"] = "Viridis"
            marker_kwargs["colorbar"] = dict(title="Value")
        trace = go.Scatter3d(
            x=P[:, 0],
            y=P[:, 1],
            z=P[:, 2],
            mode="markers+text",
            marker=marker_kwargs,
            text=labels_list,
            hovertext=labels_list,
            hoverinfo="text",
        )
        layout = go.Layout(
            title=title,
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
            ),
        )
        fig = go.Figure(data=[trace], layout=layout)
        return fig
    # Fallback to Matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if V is not None:
        p = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=V, cmap="viridis", **kwargs)
        fig.colorbar(p, ax=ax, label="Value")
    else:
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], **kwargs)
    for i, txt in enumerate(labels_list):
        ax.text(P[i, 0], P[i, 1], P[i, 2], txt)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    return fig


__all__ = [
    "plot_coherence_heatmap",
    "plot_velocity_surface",
    "animate_velocity",
    "plot_sensor_positions_3d",
]