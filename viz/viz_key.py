
import sys
import pptk
import pandas as pd
import numpy as np

from gym_sapien.envs.util import visualize_key

df = pd.read_csv(sys.argv[1])
pc = df[['x', 'y', 'z']].to_numpy()
src = df[['src']].to_numpy().squeeze().astype('bool')
dst = df[['tgt']].to_numpy().squeeze().astype('bool')

visualize_key(pc, src, dst)
