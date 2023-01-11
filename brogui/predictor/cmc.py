from pathlib import Path

import torch
import numpy as np

from .features import calc_feature
from .models import CMCModel

cmc_mode_dir = str(Path(__file__).absolute().parent.parent / Path('data/models/cmc_best_model_keku_84.pt'))
cmc_model = CMCModel(in_dim=74, hidden_dim=256, n_classes=1)
cmc_ckpt = torch.load(cmc_mode_dir)
cmc_model.load_state_dict(cmc_ckpt)
cmc_model.eval()

def calc(smiles: str) -> float:
    """
    
    Args:
        smiles (Union[str, list]): 
        log_name (Union[str, Path]):
    
    Returns:
        y_predict (list): 
    """

    y_pred = np.nan
    with torch.no_grad():
        one_graph_feature, one_phys_feature = calc_feature(smiles, 'cmc')
        if one_graph_feature:
            y_pred = cmc_model(one_graph_feature, one_phys_feature).numpy()[0][0]
    
    return y_pred