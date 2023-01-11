from pathlib import Path

import torch
import numpy as np

from .features import mol2g
from .models import KrafftModel

krafft_mode_dir = str(Path(__file__).absolute().parent.parent / Path('data/models/krafft_best_model_keku_20221009.pt'))
krafft_model = KrafftModel(in_dim=74, hidden_dim=256, n_classes=1)
krafft_ckpt = torch.load(krafft_mode_dir)
krafft_model.load_state_dict(krafft_ckpt)
krafft_model.eval()

def calc(smi: str) -> float:
    """
    
    Args:
        smiles (Union[str, list]): 
        log_name (Union[str, Path]):
    
    Returns:
        y_predict (float): 
    """
    
    y_pred = np.nan
    with torch.no_grad():
        g = mol2g(smi)
        if g:
            y_pred = krafft_model(g).numpy()[0][0]
    
    return y_pred*100