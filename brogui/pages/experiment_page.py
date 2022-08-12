from inspect import Parameter
from sqlite3 import paramstyle
import streamlit as st
from PIL import Image
import pandas as pd

from utils.variables import AF_OPTIONS, DOE_OPTIONS
from utils.variables import OPTION_PARAMS, VariableFactory

def experiment_page():

    # left_col, right_col = st.columns(2)

    st.markdown("# Experiment")

    st.sidebar.markdown("## Step 1. Create a new experiment")
    # exp_id = 
    exp_name = st.sidebar.text_input('Experiment Name')
    exp_desc = st.sidebar.text_input('Goal Description', value='MOO')

    Y_vector = st.multiselect(
        'Goal Dimensions',
        ['STY(space-time-yield)', 'E-Factor'],
        ['STY(space-time-yield)', 'E-Factor'])
    Y_dims = len(Y_vector)
    X_vector = st.multiselect(
        'Input Dimensions', OPTION_PARAMS, OPTION_PARAMS
    )
    X_dims = len(X_vector)

    acqucision_function_select = st.sidebar.selectbox(
        'Acqucision Function', (AF_OPTIONS),
        index=1
    )
    initial_sampling_select = st.sidebar.selectbox(
        'Initial Sampling', (DOE_OPTIONS), 
        index=1
    )
    
    # Init experiment description table
    key_items = ['Exp Name', 'Goal', 'Output(Y) dimension', 'Input(X) dimension', 'Acqucision function', 'Initial sampling']
    descriptions = [exp_name, exp_desc, str(Y_dims), str(X_dims), acqucision_function_select, initial_sampling_select]
    exp_desc_tb = pd.DataFrame({'Key Items': key_items, "Description": descriptions})

    # Init parameters description table
    params = [VariableFactory(i) for i in X_vector]
    parameter_names = [f"{i.symbol} - {i.parameter}({i.unit})" for i in params]
    parameter_types = [f"{i.parameter_type}" for i in params]
    parameter_values = [f"{str(i.parameter_range)}" for i in params]
    param_desc_tb = pd.DataFrame({
        'Parameter': parameter_names, 
        'Type': parameter_types, 
        'Values': parameter_values})

    exp_create_btn = st.sidebar.button('Create')
    exp_create_valid = True
    if exp_create_btn:
        # validation
        if '' == exp_name:
            exp_create_valid = False
            st.sidebar.error('Invalid experiment name.')

        if 0 == Y_dims:
            exp_create_valid = False
            st.sidebar.error('Goal Dimensions should be integer not {Y_dims}.')
        
        if 0 == X_dims:
            exp_create_valid = False
            st.sidebar.error('Input Dimensions should be integer not {X_dims}.')

        if exp_create_valid:
            st.markdown("## Create a new experiment:")
            st.table(exp_desc_tb)
            st.markdown("## Parameters and scopes:")
            st.table(param_desc_tb)

    st.sidebar.markdown("---")

    st.sidebar.markdown("## Step 2. Setup parameters")
    
    st.sidebar.markdown("---")

    st.sidebar.markdown("## Step 3. ")
    st.sidebar.markdown("---")

    