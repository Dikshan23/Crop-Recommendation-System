from pathlib import Path

import pandas as pd
import streamlit as st

from admin.shared import require_admin_page
from src.train import train


def render():
    require_admin_page("Retrain Model")

    st.title("🔄 Retrain Model")
    st.warning("Choose a data source before training. You can use the bundled dataset, upload a CSV/XLSX file")

    source = st.radio(
        "Training source",
        ["Bundled dataset", "Upload CSV/XLSX",],
        horizontal=True,
    )

    training_input = None

    if source == "Upload CSV/XLSX":
        uploaded_file = st.file_uploader("Upload training file", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            suffix = Path(uploaded_file.name).suffix.lower()
            if suffix == ".csv":
                training_input = pd.read_csv(uploaded_file)
            else:
                training_input = pd.read_excel(uploaded_file)

    if st.button("🚀 Start Retraining Now", type="primary"):
        with st.spinner("Training new model..."):
            try:
                train(training_input)
                st.success("✅ Model retrained and saved successfully!")
            except Exception as e:
                st.error(f"Training failed: {e}")