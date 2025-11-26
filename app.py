"""
Streamlit application entry point.

This script launches a simple web interface that wraps the GPU‑accelerated
signal analysis pipeline defined in ``signal_analysis_pipeline_torch.py``.
Users can upload their own CSV files, configure analysis parameters and
interactively explore the results (spectral similarity, wavelet
coherence, propagation velocities) without writing any code.  The
heavy computations run on the GPU when available and the outputs are
cached to avoid recomputation.

To start the app run:

```bash
streamlit run app.py
```

See ``pages.py`` for the individual page implementations.
"""

import streamlit as st

from pages import upload_page, analysis_page, coherence_page, propagation_page, band_similarity_page, drift_analysis_page


PAGE_NAMES = {
    "Data Upload": upload_page,
    "Analysis": analysis_page,
    "Coherence": coherence_page,
    "Propagation": propagation_page,
    "Band Similarity": band_similarity_page,
    "Drift Analysis": drift_analysis_page,
}


def main() -> None:
    st.set_page_config(page_title="GPU Signal Analysis", layout="wide")
    st.sidebar.title("Navigation")
    page_name = st.sidebar.selectbox("Select a page", list(PAGE_NAMES.keys()))
    st.sidebar.markdown("---")
    # Render selected page
    PAGE_NAMES[page_name]()


if __name__ == "__main__":
    main()