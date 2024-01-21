import streamlit as st
import os


def page_config():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    st.set_page_config(
        page_title="Data App Demo",
        layout="centered",
        initial_sidebar_state="expanded",
    )