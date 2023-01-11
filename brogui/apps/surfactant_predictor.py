from builtins import bytes
from datetime import datetime
import base64
import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, JsCode, GridOptionsBuilder
from rdkit import Chem
from rdkit.Chem import Draw

from predictor import calc_cmc, calc_kft, calc_sft

TABLE_HEIGHT = 600

def app():
    st.session_state.fixed_cols = ["Input SMILES", "Canonical SMILES"]
    st.session_state.pred_cols = ["LogCMC(uM)", "SFT(mN/m)", "Krafft Point(°C)", "pKa"]
    st.session_state.last_upload_file = None
    if "pred_df" not in st.session_state:
        st.session_state.pred_df = pd.DataFrame(columns=st.session_state.fixed_cols+st.session_state.pred_cols)

    st.markdown("# Surfactant Properties Prediction")

    st.sidebar.markdown("## Add Molecules")
    new_smi = st.sidebar.text_input("Input SMILES", value="C1=C(OC(=C1)C=O)CO")
  
    add_smi_btn = st.sidebar.button("Add")
    if add_smi_btn:
        try: 
            canon_smi = Chem.CanonSmiles(new_smi) 

            # get base64 encoded molecule image data
            mol = Chem.MolFromSmiles(canon_smi)
            d2d = Draw.MolDraw2DSVG(300, 150)
            d2d.DrawMolecule(mol)
            d2d.FinishDrawing()
            text = d2d.GetDrawingText()
            imtext = base64.b64encode(bytes(text, 'utf-8')).decode('utf8')

            new_row = {
                'Input SMILES': new_smi,
                'Canonical SMILES': canon_smi,
                '2D Image': imtext,
                'LogCMC(uM)': calc_cmc(canon_smi),
                'SFT(mN/m)': calc_sft(canon_smi),
                'Krafft Point(°C)': calc_kft(canon_smi)
            }

            st.session_state.pred_df = st.session_state.pred_df.append(new_row, ignore_index=True)
        except Exception as exc:
            print(exc)
 
    render()

    st.sidebar.markdown("## Table Action")
    add_export_button(
        st.session_state.pred_df.to_csv(index=False).encode('utf-8'))

    # add_upload_button()

def render():
    render_pred_aggrid()

def render_pred_aggrid():
    if 'pred_df' in st.session_state:
        st.markdown("----")
        st.markdown("### Property Prediction Table")

        df = st.session_state.pred_df
        go = GridOptionsBuilder.from_dataframe(df)
        go.configure_default_column(
            editable = False,
            filterable = True,
            resizable = True,
            sortable = True
        )
        go.configure_columns(
            [{
                "headerName": col_name,
                "field": col_name,
                "editable": False,
                "type": ["numericColumn", "numberColumnFilter", "centerAligned"]
            } for col_name in st.session_state.fixed_cols] + \
            [{
                "headerName": col_name,
                "field": col_name,
                "editable": False,
                "type": ["numericColumn", "numberColumnFilter"]
            } for col_name in st.session_state.pred_cols]
        )
 
        go.configure_column(
            '2D Image', 
            cellRenderer=image_render,
            pinned=True,
            autoHeight=True,
            minWidth=322
            )

        grid_options = go.build()
        st.session_state.ag_grid = AgGrid(
            df, 
            theme="streamlit",
            height=TABLE_HEIGHT,
            gridOptions=grid_options,
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,
            reload_data=True)

def add_export_button(csv):
    exp_name = "property_table"
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    trial_csvfile_name = f"{exp_name}-T-{now}.csv"
    st.sidebar.download_button(
            "Export",
            csv,
            trial_csvfile_name,
            "text/csv",
            key='download-csv'
            )

def add_upload_button(): 
    csvfile = st.sidebar.file_uploader("Upload")
    if csvfile != st.session_state.last_upload_file:
        if csvfile is not None:
            st.session_state.pred_df = pd.read_csv(csvfile)
            st.session_state.last_upload_file = csvfile

image_render = JsCode("""function (params) {
    this.params = params;
    this.img_render = document.createElement('div');
    
    this.img_render.innerHTML = `
        <img src="data:image/svg+xml;base64,` + this.params.data["2D Image"] + `"/>`;
    
    return this.img_render;
    }""")