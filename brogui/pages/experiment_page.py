import streamlit as st
import pandas as pd

from nextorch import plotting, bo, doe

from utils.variables import AF_OPTIONS, DOE_OPTIONS
from utils.variables import OPTION_PARAMS, VariableFactory

def experiment_page():
    
    st.markdown("# Experiment")
    
    st.sidebar.markdown("## Setup Experiment")
    
    # exp_id = 
    exp_name = st.sidebar.text_input('Experiment Name', value="demo") #FIXME: generate a random unique expid
    exp_desc = st.sidebar.text_input('Goal Description', value='MOO')

    Y_vector = st.sidebar.multiselect(
        'Goal Dimensions',
        ['STY(space-time-yield)', 'E-Factor'],
        ['STY(space-time-yield)', 'E-Factor'])
    Y_dims = len(Y_vector)
    X_vector = st.sidebar.multiselect(
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
    n_initial = st.sidebar.text_input('Initial Sampling Num', value='10')
    X_initial_plot = st.sidebar.multiselect(
        'Initial Plotting Dimensions', OPTION_PARAMS, OPTION_PARAMS
    )

    # cache exp info into session states
    st.session_state.exp_name = exp_name
    st.session_state.exp_desc = exp_desc
    st.session_state.Y_vector = Y_vector
    st.session_state.X_vector = X_vector
    st.session_state.Y_dims = Y_dims
    st.session_state.X_dims = X_dims
    st.session_state.acqucision_function_select = acqucision_function_select
    st.session_state.initial_sampling_select = initial_sampling_select
    st.session_state.n_initial = n_initial
    st.session_state.X_initial_plot = X_initial_plot
    render()

    st.sidebar.markdown("## Generate Initial Samples")
    init_sampling_btn = st.sidebar.button('Sample')
    if init_sampling_btn:
        X_init_lhc = doe.latin_hypercube(n_dim = X_dims, n_points = int(n_initial), seed= 1)
        # Compare the 3 sampling plans
        params = [VariableFactory(i) for i in st.session_state.X_initial_plot]
        X_ranges = [i.parameter_range for i in params]

        from utils.plotting import sampling_3d
        sampling_3d([X_init_lhc],
                    X_names = st.session_state.X_initial_plot,
                    X_ranges = X_ranges,
                    design_names = ['LHC'],
                    )

def render():
    render_exp_info(
        st.session_state.exp_name, 
        st.session_state.exp_desc, 
        str(st.session_state.Y_dims), 
        str(st.session_state.X_dims), 
        st.session_state.acqucision_function_select, 
        st.session_state.initial_sampling_select, 
        st.session_state.n_initial)
    render_params_info(st.session_state.X_vector)

def render_exp_info(
    exp_name, exp_desc, 
    Y_dims, X_dims, 
    acqucision_function_select, 
    initial_sampling_select, 
    n_initial):

    key_items = [
        'Exp Name', 'Goal', 'Output(Y) dimension', 
        'Input(X) dimension', 'Acqucision function', 
        'Initial sampling function', 'Initial sampling num']
    descriptions = [
        exp_name, exp_desc, 
        str(Y_dims), str(X_dims), 
        acqucision_function_select, 
        initial_sampling_select, n_initial]
    exp_desc_tb = pd.DataFrame({'Key Items': key_items, "Description": descriptions})

    st.markdown('### Experiment Info')
    st.table(exp_desc_tb)

def render_params_info(X_vector):
    params = [VariableFactory(i) for i in X_vector]
    parameter_names = [f"{i.symbol} - {i.parameter}({i.unit})" for i in params]
    parameter_types = [f"{i.parameter_type}" for i in params]
    parameter_values = [f"{str(i.parameter_range)}" for i in params]
    param_desc_tb = pd.DataFrame({
        'Parameter': parameter_names, 
        'Type': parameter_types, 
        'Values': parameter_values})

    st.markdown('### Parameter Info')
    st.table(param_desc_tb)
