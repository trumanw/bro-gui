from PIL import Image

import streamlit as st

from utils.path import get_file_path, get_dir_name, data_str
from multiapp import MultiApp

img = Image.open(
    get_file_path(
        "rascore_logo.png",
        dir_path=f"{get_dir_name(__file__)}/{data_str}",
    ),
)
st.set_page_config(page_title="bro-gui", page_icon=img, layout="wide")
app = MultiApp()

from apps import home, trial
app.add_app("Start Trial", trial.app)
app.add_app("Home Page", home.app)
app.run()