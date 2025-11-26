from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AppState:
    """Runtime state shared across the DearPyGui application."""

    experiment_path: Optional[str] = None
    selected_channels: List[int] = field(default_factory=list)
    gpu_available: bool = False
    message: str = "Ready"


app_state = AppState()
