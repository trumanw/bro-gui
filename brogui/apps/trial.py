from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from st_aggrid import AgGrid
from nextorch import bo, doe, io
from nextorch.parameter import ParameterSpace
from nextorch.utils import encode_to_real_ParameterSpace

from utils.variables import AF_OPTIONS, DOE_OPTIONS,\
    TRIALS_TABLE_COLUMN_TYPE, TRIALS_TABLE_COLUMN_INDEX,\
    TRIALS_TYPE_INIT, TRIALS_TYPE_BO, INFO_TABLE_HEIGHT,\
    TRIALS_TABLE_HEIGTH
from utils.variables import OPTION_PARAMS, VariableFactory, TrialState
from utils.plotting import pareto_front
from fs.larkapi import LarkSheetSession

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
        ['STY(space-time-yield)', 'E-Factor', 'Productivity (mol/h)'],
        ['E-Factor', 'Productivity (mol/h)'])
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
    
    n_initial = st.sidebar.text_input('Initial Sampling Num', value='4')
    st.session_state.n_initial = int(n_initial)

    n_batch = st.sidebar.text_input('Batch Sampling Num', value='2')
    st.session_state.n_batch = int(n_batch)

    # show initial sampled variables in a table
    params = [VariableFactory(i) for i in st.session_state.X_vector]
    bo_params = [i.parameter() for i in params]
    parameter_space = ParameterSpace(bo_params)
    X_col_names = [f"{i.symbol} ({i.unit})" for i in params]
    Y_col_names = st.session_state.Y_vector
    X_ranges = [i.parameter_range for i in params]
    X_units = [f"{i.unit}" for i in params]

    exp = bo.EHVIMOOExperiment(st.session_state.exp_name)
    trial_df = None
    trial_no = None

    # (optional) Restore a trial from Feishu Spreadsheet
    st.sidebar.markdown("## Restore Trial From Feishu")
    with st.sidebar.form("feishu_sheet_form"):
        fs_sheet_token = st.text_input("Feishu spreadsheet token")
        fs_sheet_index = st.number_input("Feishu spreadsheet index", min_value=1)
        st.session_state.feishu_sheet_token = fs_sheet_token
        st.session_state.feishu_sheet_index = fs_sheet_index - 1 # user input index starts from 0
        st.session_state.fs = LarkSheetSession() 
        fs_sheet_sync = st.form_submit_button("Sync")

    if fs_sheet_sync:
        remote_df, remote_sheet_id, resp_code, resp_error = st.session_state.fs.load_trials_from_remote(
            st.session_state.feishu_sheet_token, st.session_state.feishu_sheet_index)
        trial_no = max(remote_df["Trial Index"].astype(int).tolist())
        # remove rows including NaN values
        nonan_trial_df = remote_df.dropna(axis=0, how='any')
        X_restore_input = nonan_trial_df[X_col_names].astype(float).to_numpy()
        Y_restore_input = nonan_trial_df[Y_col_names].astype(float).to_numpy()

        # restore bo.Experiment instance from file
        exp.define_space(bo_params)
        exp.input_data(
            X_restore_input, Y_restore_input,
            X_ranges = X_ranges, X_names = X_units,
            unit_flag = True)

        #FIXME init ref_point
        ref_point = [10.0, 10.0]
        exp.set_ref_point(ref_point)
        exp.set_optim_specs(maximize=True)

        st.session_state.exp = exp
        st.session_state.trials = remote_df
        st.session_state.feishu_sheet_id = remote_sheet_id
        st.session_state.trial_state = TrialState.RESTORED
        st.session_state.trial_index = trial_no
        st.session_state.fixed_trials_headers = [TRIALS_TABLE_COLUMN_INDEX, TRIALS_TABLE_COLUMN_TYPE]
        st.session_state.X_trials_headers = X_col_names
        st.session_state.Y_trials_headers = Y_vector

    # (optional) Initialize from saved trials csv file
    st.sidebar.markdown("## Step 0. (Optional) Continue a Trial")
    trial_csvfile = st.sidebar.file_uploader("Restore trial from file:")
    if trial_csvfile is not None and 'trials' not in st.session_state:
        trial_df = pd.read_csv(trial_csvfile)
        trial_no = max(trial_df["Trial Index"].tolist())
        # remove rows including NaN values
        nonan_trial_df = trial_df.dropna(axis=0, how='any')
        X_restore_input = nonan_trial_df[X_col_names].to_numpy()
        Y_restore_input = nonan_trial_df[Y_col_names].to_numpy()

        # restore bo.Experiment instance from file
        exp.define_space(bo_params)
        exp.input_data(
            X_restore_input, Y_restore_input,
            X_ranges = X_ranges, X_names = X_units,
            unit_flag = True)

        #FIXME init ref_point
        ref_point = [10.0, 10.0]
        exp.set_ref_point(ref_point)
        exp.set_optim_specs(maximize=True)

        st.session_state.exp = exp
        st.session_state.trials = trial_df
        st.session_state.trial_state = TrialState.RESTORED
        st.session_state.trial_index = trial_no 
        st.session_state.fixed_trials_headers = [TRIALS_TABLE_COLUMN_INDEX, TRIALS_TABLE_COLUMN_TYPE]
        st.session_state.X_trials_headers = X_col_names
        st.session_state.Y_trials_headers = Y_vector

    # generate initial samples
    st.sidebar.markdown("## Step 1. New a Trial")
    init_sampling_btn = st.sidebar.button('New')
    if init_sampling_btn:
        # X_init_lhc_original = doe.latin_hypercube(
        X_init_lhc_original = doe.randomized_design(
            n_dim = st.session_state.X_dims, 
            n_points = st.session_state.n_initial)

        real_X_init_lhc = encode_to_real_ParameterSpace(X_init_lhc_original, parameter_space)
    
        trial_df = pd.DataFrame(
            data=np.array(real_X_init_lhc), columns=X_col_names
        )
        trial_no = 0 

        # add "Trial Number", "Trial Type" into the initial df
        trial_df[TRIALS_TABLE_COLUMN_INDEX] = 0
        trial_df[TRIALS_TABLE_COLUMN_TYPE] = TRIALS_TYPE_INIT 
        for target_col in Y_vector:
            trial_df[target_col] = np.nan
        new_cols_names_ordered = [TRIALS_TABLE_COLUMN_INDEX, TRIALS_TABLE_COLUMN_TYPE] + X_col_names + Y_vector
        trial_df = trial_df[new_cols_names_ordered]
        X_restore_input = trial_df[X_col_names].to_numpy()
        Y_restore_input = trial_df[Y_col_names].to_numpy()
        
        # create a new experiment
        exp.define_space(bo_params)

        st.session_state.exp = exp
        st.session_state.trials = trial_df
        st.session_state.trial_state = TrialState.INITIALIZED
        st.session_state.trial_index = trial_no 
        st.session_state.fixed_trials_headers = [TRIALS_TABLE_COLUMN_INDEX, TRIALS_TABLE_COLUMN_TYPE]
        st.session_state.X_trials_headers = X_col_names
        st.session_state.Y_trials_headers = Y_vector

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

        # convert Object type back to float
        for col_name in X_trials_headers+Y_trials_headers:
            df[col_name] = df[col_name].astype(float)
            
        add_trial_save_button(csv)
        add_trial_upload_button(df)
        
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
    exp_name = st.session_state.exp_name
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    trial_csvfile_name = f"{exp_name}-T-{now}.csv"
    st.download_button(
            "Export",
            csv,
            trial_csvfile_name,
            "text/csv",
            key='download-csv'
            )

def add_trial_upload_button(csv):
    upload_to_feishu_sheet = st.button("Upload")
    if "fs" in st.session_state and upload_to_feishu_sheet:
        fs_session = st.session_state["fs"]
        if "feishu_sheet_token" in st.session_state and \
            "feishu_sheet_id" in st.session_state:
            sheet_token = st.session_state.feishu_sheet_token
            sheet_id = st.session_state.feishu_sheet_id
            upload_state, resp_code, resp_error = fs_session.save_trials_to_remote(
                sheet_token, sheet_id, csv.values.tolist())
            if not upload_state:
                st.error(f"failed to upload due to {resp_code}:{resp_error}")
        else:
            st.error("failed to get Feishu spreadsheetToken and sheetId.")   

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
        
        if TrialState.RESTORED == st.session_state.trial_state:
            st.session_state.trial_state = TrialState.INLOOP
        elif TrialState.INITIALIZED == st.session_state.trial_state:
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
            st.session_state.trial_state = TrialState.INLOOP
        elif TrialState.INLOOP == st.session_state.trial_state:
            Y_initial_input = df[df[TRIALS_TABLE_COLUMN_INDEX] == (st.session_state.trial_index)][Y_col_names].to_numpy()
            #FIXME: check no-NaN in the Y_initial_input

            # run trial to update surrogate model
            st.session_state.exp.run_trial(
                st.session_state.last_X_new, 
                st.session_state.last_X_new_real, 
                Y_initial_input)

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