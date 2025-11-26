# MycoDash Signals (GPU)

GPU-accelerated, DearPyGui-based tools for managing, inspecting, and labeling large multi-channel time-series recordings from fungal/mycelial sensor arrays. This repository supersedes the earlier Streamlit prototype with a desktop-focused workflow designed for CUDA-enabled analysis pipelines and long-duration datasets.

## Key capabilities
- **GPU signal processing**: PyTorch/CUDA transforms for detrending, filtering, FFT/STFT, wavelets, coherence, and propagation metrics (see `signal_analysis_pipeline_torch.py`).
- **Desktop GUI (DearPyGui)**: Docking layout with entry tabs for data management, time-series navigation, spectral/fitting views, 3D sensor layout editing, and labeling workflows (see `dpg_app.py` and `gui/main/`).
- **Chunked, large-file handling**: Suitable for multi-week recordings with MHz–sub-Hz sampling via chunked loading and GPU-first computation.
- **Extensible analysis**: Building blocks for peak detection, section labeling, and future SpikeInterface integration.

## Repository layout
- `dpg_app.py` / `gui/main/` — DearPyGui application entry point, menu bar, brand theme, and shared state scaffolding.
- `signal_analysis_pipeline_torch.py` — Primary CUDA-accelerated signal analysis pipeline (FFT/STFT/CWT, coherence, propagation).
- `signal_analysis_pipeline.py` — CPU reference implementation.
- `propagation_analysis.py`, `visualization_tools.py` — Higher-level utilities for propagation metrics and plotting.
- `coherence_explorer_app.py`, `app.py`, `pages.py` — Legacy Streamlit apps (kept for reference while migrating to DearPyGui).
- `extended_tutorial.ipynb` (and related notebooks) — Example analyses and walkthroughs of the GPU pipeline.

## Requirements
- **Python**: 3.9–3.11 recommended.
- **GPU stack**: CUDA 12.x drivers with a compatible PyTorch build. Verify availability:
  ```bash
  python - <<'PY'
  import torch
  print('CUDA available:', torch.cuda.is_available())
  print('Device count:', torch.cuda.device_count())
  PY
  ```
- **Core Python packages**:
  - `torch` (CUDA build), `numpy`, `scipy`, `matplotlib`
  - `dearpygui` (for the desktop GUI)
  - Optional: `plotly`, `pandas`, `h5py` depending on your data sources

## Installation
1. **Clone and enter the repository**
   ```bash
   git clone https://github.com/rubcaspac60/MycoDash_Signals_GPU.git
   cd MycoDash_Signals_GPU
   ```
2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # on Windows use: .venv\Scripts\activate
   ```
3. **Install CUDA-enabled PyTorch** (pick the wheel that matches your driver)
   ```bash
   pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
   ```
   Adjust `cu121` to your local CUDA version (e.g., `cu118`, `cu124`).
4. **Install GUI and scientific dependencies**
   ```bash
   pip install dearpygui numpy scipy matplotlib
   ```
   Add any optional packages (e.g., `plotly`, `pandas`, `h5py`) as needed for your datasets.

## Running the DearPyGui desktop app
```bash
python dpg_app.py
```
This launches the docking-ready viewport with placeholder tabs for data management, time-series viewing, spectral/fitting, 3D layout editing, and labeling. Future milestones will wire these tabs into the GPU pipeline.

## Quick testing routines
- **Bytecode check for GUI entry points** (fast syntax validation):
  ```bash
  python -m compileall gui dpg_app.py
  ```
- **GPU availability check** (verify PyTorch sees CUDA):
  ```bash
  python - <<'PY'
  import torch
  print('CUDA available:', torch.cuda.is_available())
  print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
  PY
  ```

## Tutorials and walkthroughs
- **Desktop app tutorial**: see [`docs/desktop_app_tutorial.md`](docs/desktop_app_tutorial.md) for installation, launch steps, and first-use guidance specific to the DearPyGui application.
- **GPU analysis notebooks**: open `extended_tutorial.ipynb` (and related notebooks) for end-to-end examples of the CUDA signal analysis pipeline.

## Roadmap (high-level)
- Wire DearPyGui tabs to chunked data loaders and GPU transforms.
- Add interactive 3D sensor layout editor with import/export.
- Integrate spectral fitting, peak detection, and labeling with a persistent database backend.
- Provide SpikeInterface-compatible adapters for data ingestion and sorting.
