"computing initial inducing points for sparse GP using Farthest Point Sampling"

import torch
from dgl.geometry import farthest_point_sampler
import pandas as pd
import numpy as np

def farthest_point_sampling(no_initial_points, path):
    df=pd.read_csv(path)
    array=np.expand_dims(df.iloc[:, 0:3].to_numpy(), axis=0)
    sampled_pc_idx=torch.squeeze(farthest_point_sampler(torch.from_numpy(array) , no_initial_points))
    idx_list= sampled_pc_idx.tolist()
    sampled_pc= df.iloc[:, 0:3].to_numpy()[idx_list][:]
    return sampled_pc

#example
sampled_pc=farthest_point_sampling(60, 'resources/curvature_cc/bun_zipper.csv')