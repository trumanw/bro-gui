from pathlib import Path
from typing import Union

import torch
import numpy as np

from .features import calc_feature
from .models import SFTModel

sft_mode_dir = str(Path(__file__).absolute().parent.parent / Path('data/models/sft_model.pt'))
sft_model = SFTModel(in_dim=74, hidden_dim=128, n_classes=1)
sft_ckpt = torch.load(sft_mode_dir)
sft_model.load_state_dict(sft_ckpt)
sft_model.eval()

def calc(smiles: Union[str, list],
           temp : float = 298.15
          ) -> float:
    """
    """
    
    y_pred = np.nan
    with torch.no_grad():
        one_graph_feature, one_phys_feature = calc_feature(smiles, 'sft')
        if one_graph_feature:
            temp = torch.tensor([[temp / 400]])
            y_pred = sft_model(one_graph_feature, temp, one_phys_feature).numpy()[0][0]
    
    return y_pred*100 - 1