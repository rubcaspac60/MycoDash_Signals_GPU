"""Main application bootstrap and shared state for the DearPyGui app."""

from .app import run_app
from .state import AppState, app_state
from .theme import apply_brand_theme, BRAND_COLORS

__all__ = [
    "run_app",
    "AppState",
    "app_state",
    "apply_brand_theme",
    "BRAND_COLORS",
]
