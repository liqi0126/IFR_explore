import os
import pandas as pd

import torch


def load_pc(base_dir, pc_path):
    df = pd.read_csv(os.path.join(base_dir, pc_path))
    pc = df[['x', 'y', 'z']].to_numpy()
    pc = torch.from_numpy(pc).float()
    return pc
