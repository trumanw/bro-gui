# requirements: scipy, numpy, pandas, rdkit, sklearn, IPython, PIL, matplotlib
from pathlib import Path
import pickle
from typing import Union, Optional

import torch
import numpy as np
import pandas as pd
import dgl
from dgllife.utils import CanonicalAtomFeaturizer, mol_to_bigraph
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
lg = RDLogger.logger()
lg.setLevel(4)

from .utils import chemtools as ct
from .utils import vectools

SCALER = {
    'cmc': np.array([[1.167560e+03, 8.904600e+00, 3.045968e+02, 7.414142e-01, 5.643559e+00,
                    4.441616e+00, 7.522936e-01, 6.284404e-01]], dtype=np.float32),
    'sft': np.array([[9.0393103e+02, 1.8097099e+01, 2.7015399e+02, 8.2159954e-01, 9.1092854e+00,
                    3.8424213e+00, 6.9902915e-01, 4.7076923e-01]], dtype=np.float32)
}

def pkl_load(one_path):
    with open(one_path, 'rb') as f:
        return pickle.load(f)

def numpy_2_fp(array: np.array,
               dtype: str = 'UIntSparseIntVect'):
    if dtype == 'UIntSparseIntVect':
        fp = DataStructs.cDataStructs.UIntSparseIntVect(len(array))
    elif dtype == 'ExplicitBitVect':
        fp = DataStructs.cDataStructs.ExplicitBitVect(len(array))
    for ix, value in enumerate(array):
        fp[ix] = int(value)
    return fp

#################### set deault params ###################
ECFP_BITS = 1024
ECFP_RADIUS = 2

# set data path
REPO_DIR = Path(__file__).parent.parent / Path('data/models/REPODATA')

ct.FPWEIGHT_D['SASCORE'] = tuple(
    np.load(REPO_DIR / 'SA_fpscores.npz').values()
)

ct.FPWEIGHT_D['SCSCORE'] = tuple(
    np.load(REPO_DIR / 'SC_fpscores.npz').values()
)

################## demo ###################

# load data
fp_db = {
    'cmc': None,
    'sft': None
}

with open(REPO_DIR / 'CMC.pkl', 'rb') as f:
    fp_db['cmc'] = pickle.load(f)

with open(REPO_DIR / 'surface_tension.pkl', 'rb') as f:
    fp_db['sft'] = pickle.load(f)

def calc_descriptor(smiles : Union[str, list], 
                    mode : str = 'cmc'
                   ) -> pd.DataFrame:
    """
    Args: 
        smiles: SMILES representation of molecule
        mode: which reference fingerprint need to load, allowable values are [cmc, sft]

    Returns:
        out_df (pd.DataFrame): calculated features of molecules
        
    """
    usedcols = ['SMILES', 'MW', 'LOGP', 'MR', 'QED', 'SASCORE', 'SCSCORE']
    
    if isinstance(smiles, str):
        smiles = [smiles]
    
    smi_recorder = ct.SmilesRecorder()
    smi_df = smi_recorder.fit(smiles, calc_fpscore=True)
    
    if smi_df.empty:
        return smi_df
    
    out_df = smi_df.loc[:, usedcols]

    # get LIBSIM_MORGAN, get LIBSIM_PHARM

    out_df['mol'] = out_df.SMILES.apply(lambda x: Chem.MolFromSmiles(x))
    # Optional: 删除rdkit不能识别的分子
    out_df.dropna(inplace=True)
    
    out_df['pharm_fp'] = out_df.mol.apply(lambda x: numpy_2_fp(vectools.mol_pharmfp(x)))
    out_df['ecfp_fp'] = out_df.mol.apply(lambda x: AllChem.GetHashedMorganFingerprint(x, ECFP_RADIUS, nBits=ECFP_BITS))

    # DataStructs.DiceSimilarity(fp, mean_mogran_fp) for ECFP
    # DataStructs.TanimotoSimilarity(fp, mean_pharm_fp) for PHARM_FP
    
    out_df['LIBSIM_MORGAN'] = out_df['ecfp_fp'].apply(
        lambda x: DataStructs.DiceSimilarity(x, fp_db[mode]['mean_mogran_fp']))
    out_df['LIBSIM_PHARM'] = out_df['pharm_fp'].apply(
        lambda x: DataStructs.TanimotoSimilarity(x, fp_db[mode]['mean_pharm_fp']))

    # drop useless columns: mol, pharm_fp, ecfp_fp
    out_df.drop(columns=['mol', 'pharm_fp', 'ecfp_fp'], inplace=True)

    # save data
    # out_df.to_csv('xxx_features.csv', index=None, na_rep='')
    return out_df

def calc_feature(smiles : str, 
                 mode : str ='cmc'
                ) -> tuple:
    """
    Args:
        smiles (str): Molecule smiles
        mode (str): which library used to compare. Allowed values are: cmc,  for CMC prediction; sft, for static surface tension prediction
    
    Returns:
        mol_graph (dgl.DGLGraph): Graph representation of molecule
        phys_feature (torch.tensor): Physichemical features of molecule
    """
   
    smi_df = calc_descriptor([smiles], mode)
    if smi_df.empty:
        return None, None
    
    phys_feature = np.array(calc_descriptor(smiles, mode).values.tolist()[0][1:], dtype=np.float32) / SCALER[mode]
    mol_graph = mol2g(smiles)
    
    return mol_graph, torch.tensor(phys_feature)

def mol2g(one_smiles: str, 
          num_virtual_nodes : int =0
         ) -> Optional[dgl.DGLGraph]:
    """
    
    Args:
        one_smiles (str): Molecule SMILES

    Returns:
        g (dgl.DGLGraph): Graph representation of molecule
    """
    m = Chem.MolFromSmiles(one_smiles)
    if not m:
        print('Invalid SMILES, can not be converted.')
        return None
    
    node_enc = CanonicalAtomFeaturizer()
    edge_enc = None
    g = mol_to_bigraph(m, True, node_enc, edge_enc, False)
    if not num_virtual_nodes:
        return g
    else:
        num_real_nodes = g.num_nodes()
        n, d = g.ndata['h'].shape
        real_nodes = list(range(num_real_nodes))
        g.add_nodes(1, {'h': torch.ones(1, d) * 0.02})
        virtual_src = []
        virtual_dst = []
        for count in range(num_virtual_nodes):
            virtual_node = num_real_nodes + count
            virtual_node_copy = [virtual_node] * num_real_nodes
            virtual_src.extend(real_nodes)
            virtual_src.extend(virtual_node_copy)
            virtual_dst.extend(virtual_node_copy)
            virtual_dst.extend(real_nodes)
        g.add_edges(virtual_src, virtual_dst)
        for ek, ev in g.edata.items():
            ev = torch.cat([ev, torch.zeros(g.num_edges(), 1)], dim=1)
            ev[-num_virtual_nodes * num_real_nodes * 2:, -1] = 1
            g.edata[ek] = ev
        return g