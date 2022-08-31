import streamlit as st

def app():

    left_col, right_col = st.columns(2)

    right_col.markdown("# Dashboard")
    right_col.markdown("### Bayesian Reaction Optimization(BRO) experiments dashboard")

    st.sidebar.markdown("## Database-Related Links")

    st.sidebar.markdown("## Community-Related Links")

    st.markdown("---")

    left_col.markdown(
        """
        ### Usage

        To the left, is a dropdown main menu for navigating to 
        each page in the *Rascore* database:

        - **Home Page:** We are here!
        - **Database Overview:** Overview of the *Rascore* database, molecular annotations, 
        and RAS conformational classification.
        - **Search PDB:** Search for individual PDB entries containing RAS structures.
        - **Explore Conformations:** Explore RAS SW1 and SW2 conformations found in the PDB by nucleotide state.
        - **Analyze Mutations:** Analyze the structural impact of RAS mutations by comparing WT and mutated structures.
        - **Compare Inhibitors:** Compare inhibitor-bound RAS structures by compound binding site and chemical substructure.
        - **Query Database:** Query the *Rascore* database by conformations and molecular annotations.
        - **Classify Structures:** Conformationally classify and annotate the molecular contents of uploaded RAS structures.
        """
    )