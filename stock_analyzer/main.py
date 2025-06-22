import streamlit as st
import importlib

PAGES = {
    "Strategy Analyzer": "app",
    "Fundamental Analysis": "fundamental_analysis",
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

module = importlib.import_module(PAGES[selection])
if hasattr(module, "main"):
    module.main()