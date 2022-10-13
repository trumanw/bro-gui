from PIL import Image

import streamlit as st

from utils.path import get_file_path, get_dir_name, data_str
from multiapp import MultiApp

img = Image.open(
    get_file_path(
        "logo.png",
        dir_path=f"{get_dir_name(__file__)}/{data_str}",
    ),
)
st.session_state.logo = img
st.set_page_config(page_title="bro-gui", page_icon=img, layout="wide")
app = MultiApp()

from apps import home, hydrogenation_trial, oxidation_trial
app.add_app("THF Hydrogenation Trial", hydrogenation_trial.app)
app.add_app("HMF Oxidation Trial", oxidation_trial.app)
app.add_app("Home Page", home.app)
app.run()