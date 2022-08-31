import yaml
from PIL import Image

import streamlit as st
import streamlit_authenticator as stauth

from utils.path import get_file_path, get_dir_name, util_str, data_str
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
app.add_app("Home Page", home.app)
app.add_app("Start Trial", trial.app)

# Authentication 
with open('.streamlit/config.yaml') as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    authenticator.logout('Logout', 'sidebar')
    st.write(f'Welcome *{name}*')
    app.run()

elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')