import streamlit as st
from PIL import Image

from utils.path import get_file_path, get_dir_name, util_str, data_str

class MultiApp:
    def __init__(self):
        self.apps = []
    
    def add_app(self, title, function):
        self.apps.append({"title": title, "function": function})

    def run(self):
        img = Image.open(
            get_file_path(
                "rascore_logo.png",
                dir_path=f"{get_dir_name(__file__)}/{data_str}",
            ),
        )

        st.set_page_config(page_title="bro-gui", page_icon=img, layout="wide")

        st.sidebar.markdown("## Main Menu")
        app = st.sidebar.selectbox(
            "Select Page", self.apps, format_func=lambda app: app["title"]
        )
        st.sidebar.markdown("---")
        app["function"]()

app = MultiApp()

# from pages.dashboard_page import dashboard_page
# app.add_app("Dashboard", dashboard_page)

from pages.experiment_page import experiment_page
app.add_app("Experiment", experiment_page)

# from pages.history_page import history_page
# app.add_app("History", history_page)

app.run()