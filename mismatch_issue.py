"just for resolving mismatch issue"

import pandas as pd
from geometric_kernels.spaces.mesh import Mesh

df=pd.read_csv('resources/curvature_cc/bun_zipper.csv')
print("number of vertices in curvature files from cc:", len(df))

df=pd.read_csv('resources/curvature_pc/bun_zipper.csv')
print("number of vertices in curvature files from pc:", len(df))

mesh = Mesh.load_mesh("resources/clouds/bun_zipper.ply")
print("Number of vertices in original ply files:", mesh.num_vertices)