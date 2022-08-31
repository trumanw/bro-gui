from re import A
import streamlit as st
import pandas as pd
import numpy as np
from nextorch import bo, doe, io
from nextorch.parameter import ParameterSpace
from nextorch.utils import encode_to_real_ParameterSpace
from st_aggrid import AgGrid

from utils.variables import AF_OPTIONS, DOE_OPTIONS,\
    TRIALS_TABLE_COLUMN_TYPE, TRIALS_TABLE_COLUMN_INDEX,\
    TRIALS_TYPE_INIT, TRIALS_TYPE_BO, INFO_TABLE_HEIGHT,\
    TRIALS_TABLE_HEIGTH
from utils.variables import OPTION_PARAMS, VariableFactory
from utils.plotting import pareto_front

def app():
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
        index=6
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

    # (optional) Initialize from saved trials csv file
    st.sidebar.markdown("## Step 0. (Optional) Continue a Trial")
    trial_csvfile = st.sidebar.file_uploader("Load from .csv file :")
    if trial_csvfile is not None:
        trail_df = pd.read_csv(trial_csvfile)
        st.info("Load trial from file")

    # generate initial samples
    st.sidebar.markdown("## Step 1. New a Trial")
    init_sampling_btn = st.sidebar.button('Sample')
    if init_sampling_btn:
        # X_init_lhc_original = doe.latin_hypercube(
        X_init_lhc_original = doe.randomized_design(
            n_dim = st.session_state.X_dims, 
            n_points = st.session_state.n_initial)

        # show initial sampled variables in a table
        params = [VariableFactory(i) for i in st.session_state.X_vector]
        bo_params = [i.parameter() for i in params]
        parameter_space = ParameterSpace(bo_params)

        real_X_init_lhc = encode_to_real_ParameterSpace(X_init_lhc_original, parameter_space)
    
        X_col_names = [f"{i.symbol} ({i.unit})" for i in params]
        rescale_X_init_lhc = pd.DataFrame(
            data=np.array(real_X_init_lhc), columns=X_col_names
        )

        # add "Trial Number", "Trial Type" into the initial df
        rescale_X_init_lhc[TRIALS_TABLE_COLUMN_INDEX] = 0
        rescale_X_init_lhc[TRIALS_TABLE_COLUMN_TYPE] = TRIALS_TYPE_INIT 
        for target_col in Y_vector:
            rescale_X_init_lhc[target_col] = np.nan
        new_cols_names_ordered = [TRIALS_TABLE_COLUMN_INDEX, TRIALS_TABLE_COLUMN_TYPE] + X_col_names + Y_vector
        rescale_X_init_lhc = rescale_X_init_lhc[new_cols_names_ordered]
        
        # cache trials table
        st.session_state.trials = rescale_X_init_lhc
        st.session_state.fixed_trials_headers = [TRIALS_TABLE_COLUMN_INDEX, TRIALS_TABLE_COLUMN_TYPE]
        st.session_state.X_trials_headers = X_col_names
        st.session_state.Y_trials_headers = Y_vector
        # create a new experiment
        exp_lhc = bo.EHVIMOOExperiment(st.session_state.exp_name)
        exp_lhc.define_space(bo_params)
        st.session_state.exp = exp_lhc
        st.session_state.trial_index = 0

    render()

def render():
    render_exp_and_params()
    render_explore_widget()
    render_trials_table()
    render_pareto_front()

def render_trials_table():
    if 'trials' in st.session_state:
        st.markdown('----')
        st.markdown('### Interative Trials Table')
        df = st.session_state.trials
        fixed_trials_headers = st.session_state.fixed_trials_headers
        X_trials_headers = st.session_state.X_trials_headers
        Y_trials_headers = st.session_state.Y_trials_headers

        grid_options = {
            "defaultColDef": {
                "minWidth": 5,
                "editable": False,
                "filter": True,
                "resizable": True,
                "sortable": True
            },
            "columnDefs": [{
                    "headerName": col_name,
                    "field": col_name,
                    "editable": False,
                    "type": ["numericColumn", "numberColumnFilter"]
                } for col_name in fixed_trials_headers]+ \
                [{
                    "headerName": col_name,
                    "field": col_name,
                    "editable": True,
                    "type": ["numericColumn"]
                } for col_name in X_trials_headers] + \
                [{
                    "headerName": col_name,
                    "field": col_name,
                    "editable": True,
                    "type": ["numericColumn", "numberColumnFilter"]
                } for col_name in Y_trials_headers],
        }
        st.session_state.ag_grid = AgGrid(
            df, 
            theme="streamlit", 
            gridOptions=grid_options, 
            height=TRIALS_TABLE_HEIGTH,
            fit_columns_on_grid_load=True, 
            reload_data=False)
        
        # get current state of the traisl table
        df = st.session_state.ag_grid['data']
        csv = df.to_csv(index=False).encode('utf-8')
        add_trial_save_button(csv)
        
        st.markdown('----')

def render_pareto_front():
    st.sidebar.markdown('## Step 3. Visualize Pareto Front')

    bo_plot_btn = st.sidebar.button('Plot')
    if bo_plot_btn:
        if 'exp' in st.session_state and hasattr(st.session_state.exp, 'Y_real'):
            st.markdown('### Visualize Paretor Front')
            Y_real_opts, X_real_opts = st.session_state.exp.get_optim()
            Y_col_names = st.session_state.Y_vector

            col_1, col_2 = st.columns([1, 1])
            only_pareto_fig = pareto_front(Y_real_opts[:, 0], Y_real_opts[:, 1], Y_names=Y_col_names, fill=False)
            all_samples_fig = pareto_front(st.session_state.exp.Y_real[:, 0], st.session_state.exp.Y_real[:, 1], Y_names=Y_col_names, fill=False)
            col_1.write('Only pareto front')
            col_1.pyplot(only_pareto_fig)
            col_2.write('All sampled points')
            col_2.pyplot(all_samples_fig)

def add_trial_save_button(csv):
    st.download_button(
            "Export",
            csv,
            "file.csv",
            "text/csv",
            key='download-csv'
            )

def render_explore_widget():
    st.sidebar.markdown('## Step 2. Human-in-the-loop')

    bo_next_btn = st.sidebar.button('Explore')
    if bo_next_btn:
        params = [VariableFactory(i) for i in st.session_state.X_vector]
        X_col_names = [f"{i.symbol} ({i.unit})" for i in params]
        Y_col_names = st.session_state.Y_vector
        X_ranges = [i.parameter_range for i in params]
        X_units = [f"{i.unit}" for i in params]

        # get current state of the traisl table
        df = st.session_state.ag_grid['data']
        # convert all the values from str to float
        for col in X_col_names + Y_col_names:
            df[col] = df[col].astype(float)

        # collect unique trials types
        trials_types = df[TRIALS_TABLE_COLUMN_TYPE].unique()
        if len(trials_types) > 1:   # human-in-the-loop
            Y_initial_input = df[df[TRIALS_TABLE_COLUMN_INDEX] == (st.session_state.trial_index)][Y_col_names].to_numpy()

            # run trial to update surrogate model
            st.session_state.exp.run_trial(
                st.session_state.last_X_new, 
                st.session_state.last_X_new_real, 
                Y_initial_input)

        else:   # initialize experiment
            X_initial_input = df[X_col_names].to_numpy()
            Y_initial_input = df[Y_col_names].to_numpy()
            st.session_state.exp.input_data(
                X_initial_input, Y_initial_input,
                X_ranges = X_ranges, X_names = X_units, 
                unit_flag = True)

            #FIXME valid to the range of the target values
            ref_point = [10.0, 10.0]
            st.session_state.exp.set_ref_point(ref_point)
            st.session_state.exp.set_optim_specs(maximize=True)

        # update trials table
        trial_no = st.session_state.trial_index + 1
        X_new, X_new_real, acq_func = st.session_state.exp.generate_next_point(
                n_candidates=st.session_state.n_batch)

        # update input AgGrid data back to the session_state.trials
        st.session_state.trials = df
        for row in X_new_real:
            row_dict = dict(zip([TRIALS_TABLE_COLUMN_INDEX, TRIALS_TABLE_COLUMN_TYPE] + X_col_names, [trial_no, TRIALS_TYPE_BO] + row.tolist()))
            row_df = pd.DataFrame([row_dict])
            st.session_state.trials = pd.concat([st.session_state.trials, row_df])
        
        # update last X_new, X_new_real, trial_index
        st.session_state.last_X_new = X_new
        st.session_state.last_X_new_real = X_new_real
        st.session_state.trial_index = trial_no

def render_exp_and_params():
    st.markdown('----')
    col1, col2 = st.columns([0.5, 1])
    
    # render exp info
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
    with col1:
        st.markdown('### Experiment Info')
        # st.table(exp_desc_tb)
        grid_options = {
            "defaultColDef": {
                "minWidth": 5,
                "editable": False,
                "filter": True,
                "resizable": True,
                "sortable": True
            },
            "columnDefs": [{
                "headerName": col_name,
                "field": col_name,
                "editable": False
            } for col_name in ["Key Items", "Description"]]
        }
        AgGrid(
            exp_desc_tb, 
            theme="streamlit", 
            gridOptions=grid_options, 
            height=INFO_TABLE_HEIGHT,
            fit_columns_on_grid_load=True, 
            reload_data=True)

    # render params info
    params = [VariableFactory(i) for i in st.session_state.X_vector]
    parameter_names = [f"{i.symbol} - {i.parameter_name}({i.unit})" for i in params]
    parameter_types = [f"{i.parameter_type}" for i in params]
    parameter_values = [f"{str(i.parameter_range)}" for i in params]
    parameter_interval = [f"{i.interval}" for i in params]

    param_desc_tb = pd.DataFrame({
        'Parameter': parameter_names, 
        'Type': parameter_types, 
        'Values': parameter_values,
        'Interval': parameter_interval})
    with col2:
        st.markdown('### Parameter Info')
        # st.table(param_desc_tb)
        grid_options = {
            "defaultColDef": {
                "minWidth": 5,
                "editable": False,
                "filter": True,
                "resizable": True,
                "sortable": True
            },
            "columnDefs": [{
                "headerName": col_name,
                "field": col_name,
                "editable": False
            } for col_name in ["Parameter", "Type", "Values", "Interval"]]
        }
        AgGrid(
            param_desc_tb, 
            theme="streamlit", 
            height=INFO_TABLE_HEIGHT,
            gridOptions=grid_options, 
            fit_columns_on_grid_load=True, 
            reload_data=True)