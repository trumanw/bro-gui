from re import A
import streamlit as st
import pandas as pd
import numpy as np
from nextorch import bo, doe
from st_aggrid import AgGrid

from utils.variables import AF_OPTIONS, DOE_OPTIONS
from utils.variables import OPTION_PARAMS, VariableFactory

def experiment_page():
    st.markdown("# Human-in-the-loop(HITL) Experiment")
    
    st.sidebar.markdown("## Setup Experiment")
    
    # exp_id = 
    exp_name = st.sidebar.text_input('Experiment Name', value="demo") #FIXME: generate a random unique expid
    st.session_state.exp_name = exp_name

    exp_desc = st.sidebar.text_input('Goal Description', value='MOO')
    st.session_state.exp_desc = exp_desc

    Y_vector = st.sidebar.multiselect(
        'Goal Dimensions',
        ['STY(space-time-yield)', 'E-Factor'],
        ['STY(space-time-yield)', 'E-Factor'])
    Y_dims = len(Y_vector)
    X_vector = st.sidebar.multiselect(
        'Input Dimensions', OPTION_PARAMS, OPTION_PARAMS
    )
    X_dims = len(X_vector)
    st.session_state.Y_vector = Y_vector
    st.session_state.X_vector = X_vector
    st.session_state.Y_dims = Y_dims
    st.session_state.X_dims = X_dims

    acqucision_function_select = st.sidebar.selectbox(
        'Acqucision Function', (AF_OPTIONS),
        index=1
    )
    st.session_state.acqucision_function_select = acqucision_function_select

    initial_sampling_select = st.sidebar.selectbox(
        'Initial Sampling', (DOE_OPTIONS), 
        index=1
    )
    st.session_state.initial_sampling_select = initial_sampling_select
    
    n_initial = st.sidebar.text_input('Initial Sampling Num', value='5')
    st.session_state.n_initial = int(n_initial)

    n_batch = st.sidebar.text_input('Batch Sampling Num', value='5')
    st.session_state.n_batch = int(n_batch)

    # generate initial samples
    st.sidebar.markdown("## Step 1. Generate Initial Samples")
    init_sampling_btn = st.sidebar.button('Sample')
    if init_sampling_btn:
        X_init_lhc = doe.latin_hypercube(
            n_dim = st.session_state.X_dims, 
            n_points = st.session_state.n_initial)

        # show initial sampled variables in a table
        params = [VariableFactory(i) for i in st.session_state.X_vector]
        X_ranges = [i.parameter_range for i in params]
        X_ranges_vec = np.array([i[1] - i[0] for i in X_ranges])
        X_ranges_lower = np.array([i[0] for i in X_ranges])
        X_col_names = [f"{i.symbol} ({i.unit})" for i in params]

        new_X_init_lhc = []
        for row in X_init_lhc:
            new_row = row * X_ranges_vec + X_ranges_lower
            new_X_init_lhc.append(new_row)
    
        rescale_X_init_lhc = pd.DataFrame(
            data=np.array(new_X_init_lhc), columns=X_col_names
        )

        # add "Trial Number", "Trial Type" into the initial df
        rescale_X_init_lhc['Trial Type'] = 'init'
        rescale_X_init_lhc['Trial Index'] = 0
        for target_col in Y_vector:
            rescale_X_init_lhc[target_col] = np.nan
        new_cols_names_ordered = ['Trial Index', 'Trial Type'] + X_col_names + Y_vector
        rescale_X_init_lhc = rescale_X_init_lhc[new_cols_names_ordered]
        
        # cache trials table
        st.session_state.trials = rescale_X_init_lhc
        # create a new experiment
        exp_lhc = bo.EHVIMOOExperiment(st.session_state.exp_name)
        st.session_state.exp = exp_lhc
        st.session_state.trial_index = 0

    render()

def render():
    render_exp_info()
    render_params_info()
    render_initial_params()
    render_trials_table()

def render_trials_table():
    if 'trials' in st.session_state:
        st.markdown('### Interative Trials Table')
        st.session_state.ag_grid = AgGrid(st.session_state.trials, editable=True)

def render_initial_params():
    st.sidebar.markdown('## Step 2. Initialize Experiment')
    bo_start_btn = st.sidebar.button('Start')

    params = [VariableFactory(i) for i in st.session_state.X_vector]
    X_ranges = [i.parameter_range for i in params]
    X_units = [f"{i.unit}" for i in params]
    X_col_names = [f"{i.symbol} ({i.unit})" for i in params]
    Y_col_names = st.session_state.Y_vector

    if bo_start_btn:
        st.session_state.trials = st.session_state.ag_grid['data']

        # get Y_initial and X_initial
        X_initial_input = st.session_state.trials[X_col_names].to_numpy()
        Y_initial_input = st.session_state.trials[Y_col_names].to_numpy()
        st.session_state.exp.input_data(
            X_initial_input, Y_initial_input,
            X_ranges = X_ranges, X_names = X_units, 
            unit_flag = True
        )
        #FIXME valid to the range of the target values
        ref_point = [10.0, 10.0]
        st.session_state.exp.set_ref_point(ref_point)
        st.session_state.exp.set_optim_specs(maximize=True)

        # start run trial-0 and generate samples for trial-1
        trial_no = st.session_state.trial_index + 1
        X_new, X_new_real, acq_func = st.session_state.exp.generate_next_point(
                n_candidates=st.session_state.n_batch)
        for row in X_new_real:
            row_dict = dict(zip(['Trial Index', 'Trial Type'] + X_col_names, [trial_no, "BO"] + row.tolist()))
            st.session_state.trials = st.session_state.trials.append(row_dict, ignore_index=True)
        
        # update last X_new, X_new_real, trial_index
        st.session_state.last_X_new = X_new
        st.session_state.last_X_new_real = X_new_real
        st.session_state.trial_index = trial_no

    st.sidebar.markdown('## Step 3. Human-in-the-loop')
    bo_next_btn = st.sidebar.button('Explore')
    if bo_next_btn:
        st.session_state.trials = st.session_state.ag_grid['data']

        # get incremental X_real and Y_real
        X_initial_input = st.session_state.trials[X_col_names].to_numpy()
        Y_initial_input = st.session_state.trials[
            st.session_state.trials['Trial Index'] == (st.session_state.trial_index)][Y_col_names].to_numpy()

        # run trial to update surrogate model
        st.session_state.exp.run_trial(st.session_state.last_X_new, st.session_state.last_X_new_real, Y_initial_input)

        # generate the first batch 
        trial_no = st.session_state.trial_index + 1
        X_new, X_new_real, acq_func = st.session_state.exp.generate_next_point(
                n_candidates=st.session_state.n_batch)
        for row in X_new_real:
            row_dict = dict(zip(['Trial Index', 'Trial Type'] + X_col_names, [trial_no, "BO"] + row.tolist()))
            st.session_state.trials = st.session_state.trials.append(row_dict, ignore_index=True)

        # update last X_new, X_new_real, trial_index
        st.session_state.last_X_new = X_new
        st.session_state.last_X_new_real = X_new_real
        st.session_state.trial_index = trial_no

def render_exp_info():

    key_items = [
        'Exp Name', 'Goal', 'Output(Y) dimension', 
        'Input(X) dimension', 'Acqucision function', 
        'Initial sampling function', 
        'Initial sampling num', 'Batch sampling num']
    descriptions = [
        st.session_state.exp_name, 
        st.session_state.exp_desc, 
        str(st.session_state.Y_dims), 
        str(st.session_state.X_dims), 
        st.session_state.acqucision_function_select, 
        st.session_state.initial_sampling_select, 
        str(st.session_state.n_initial),
        str(st.session_state.n_batch)]
    exp_desc_tb = pd.DataFrame({'Key Items': key_items, "Description": descriptions})

    st.markdown('### Experiment Info')
    st.table(exp_desc_tb)

def render_params_info():
    params = [VariableFactory(i) for i in st.session_state.X_vector]
    parameter_names = [f"{i.symbol} - {i.parameter}({i.unit})" for i in params]
    parameter_types = [f"{i.parameter_type}" for i in params]
    parameter_values = [f"{str(i.parameter_range)}" for i in params]
    param_desc_tb = pd.DataFrame({
        'Parameter': parameter_names, 
        'Type': parameter_types, 
        'Values': parameter_values})

    st.markdown('### Parameter Info')
    st.table(param_desc_tb)