# DearPyGui Desktop App Tutorial

This guide walks through installing the GPU-ready dependencies, launching the DearPyGui desktop shell, and performing a first sanity check of the GUI and GPU runtime. The app currently provides a docking viewport with placeholder tabs for data management, time-series navigation, spectral/fitting, 3D layout editing, and labeling. Later milestones will wire these views to the CUDA pipelines.

## 1. Prerequisites
- Python 3.9–3.11
- CUDA 12.x drivers and a compatible GPU
- Ability to install CUDA-enabled PyTorch wheels from https://download.pytorch.org/whl

## 2. Installation steps
1) **Clone and enter the repository**
```bash
git clone https://github.com/rubcaspac60/MycoDash_Signals_GPU.git
cd MycoDash_Signals_GPU
```
2) **Create and activate a virtual environment (recommended)**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```
3) **Install CUDA-enabled PyTorch**
Choose the wheel that matches your driver/CUDA runtime. For CUDA 12.1:
```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```
4) **Install GUI + scientific dependencies**
```bash
pip install dearpygui numpy scipy matplotlib
```
Add optional packages as needed: `plotly`, `pandas`, `h5py`.

## 3. Launching the app
With the environment activated, run:
```bash
python dpg_app.py
```
You should see a native window titled **"MycoDash Signals (GPU)"** with a menu bar and tabs for Data Manager, Time-Series Viewer, Spectral / Fitting, 3D Layout, and Labels. Docking is enabled by default so you can rearrange panels as features are added.

## 4. Verifying GPU availability
Before running GPU-heavy routines, confirm PyTorch can see your GPU:
```bash
python - <<'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('Device 0:', torch.cuda.get_device_name(0))
PY
```
Expect `CUDA available: True` for a functional CUDA install.

## 5. Quick sanity checks / tests
- **Bytecode compilation (fast syntax check):**
  ```bash
  python -m compileall gui dpg_app.py
  ```
- **Headless GUI smoke test (optional):**
  ```bash
  python - <<'PY'
import dearpygui.dearpygui as dpg
from gui.main.app import run_app

# Create and immediately destroy the context to ensure imports succeed.
dpg.create_context()
dpg.destroy_context()
print('DearPyGui imports OK')
PY
  ```

## 6. Next steps
- Load or simulate multi-channel data using `signal_analysis_pipeline_torch.py` to validate your CUDA toolchain.
- Begin wiring the Data Manager tab to chunked loaders and GPU transforms as outlined in the project roadmap.
- Use the labeling and layout tabs (as they evolve) to store metadata alongside your recordings for downstream ML/SpikeInterface tooling.
