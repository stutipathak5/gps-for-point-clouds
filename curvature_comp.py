import gmsh
import pandas as pd
import numpy as np
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt

gmsh.initialize()
gmsh.open('resources/meshes/bun_zipper_res3.ply')

points=np.reshape(gmsh.model.mesh.getNodes()[1],(-1,3))
cloud = PyntCloud(pd.DataFrame(points, columns=['x', 'y', 'z']))
k_neighbors = cloud.get_neighbors(k=10)
ev = cloud.add_scalar_field("eigen_values", k_neighbors=k_neighbors)
cloud.add_scalar_field("curvature", ev=ev)
ground_truth=cloud.points["curvature(11)"].to_numpy()
X=cloud.points[["x", "y", "z"]].to_numpy()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_axis_off()
ax.scatter(X[:,0], X[:,1], X[:,2], c=ground_truth, s=1)
plt.show()