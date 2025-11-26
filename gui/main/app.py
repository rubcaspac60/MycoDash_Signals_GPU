from __future__ import annotations

import dearpygui.dearpygui as dpg

from .state import app_state
from .theme import apply_brand_theme


STATUS_TEXT_TAG = "status_text"


def _log_placeholder(action: str) -> None:
    """Temporary logger for menu actions until handlers are wired."""

    dpg.log_info(f"{action} clicked (handler pending)")


def _build_menu_bar() -> None:
    with dpg.viewport_menu_bar():
        with dpg.menu(label="File"):
            dpg.add_menu_item(label="Open Dataset", callback=lambda: _log_placeholder("Open Dataset"))
            dpg.add_menu_item(label="Quit", callback=lambda: dpg.stop_dearpygui())
        with dpg.menu(label="View"):
            dpg.add_menu_item(label="GPU Status", callback=lambda: _log_placeholder("GPU Status"))
            dpg.add_menu_item(label="Cache", callback=lambda: _log_placeholder("Cache Inspector"))
        with dpg.menu(label="Help"):
            dpg.add_menu_item(label="About", callback=lambda: _log_placeholder("About"))


def _build_main_window() -> None:
    with dpg.window(tag="primary_window", label="MycoDash Signals (GPU)", width=1280, height=800):
        dpg.add_text("Status: Ready", tag=STATUS_TEXT_TAG)
        with dpg.tab_bar():
            dpg.add_tab(label="Data Manager", tag="tab_data")
            dpg.add_tab(label="Time-Series Viewer", tag="tab_timeseries")
            dpg.add_tab(label="Spectral / Fitting", tag="tab_spectral")
            dpg.add_tab(label="3D Layout", tag="tab_layout")
            dpg.add_tab(label="Labels", tag="tab_labels")


def _initialize_state() -> None:
    app_state.gpu_available = dpg.is_viewport_ok()


def run_app() -> None:
    """Launch the DearPyGui application with the base layout and theme."""

    dpg.create_context()
    dpg.configure_app(docking=True, docking_space=True)
    dpg.create_viewport(title="MycoDash Signals (GPU)", width=1280, height=800)

    _build_menu_bar()
    _build_main_window()
    apply_brand_theme()
    _initialize_state()

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary_window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    run_app()
