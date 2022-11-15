import gmsh
import pandas as pd
import numpy as np
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt


def missing_elements(L):
    start, end = L[0], L[-1]
    return sorted(set(range(start, end + 1)).difference(L))


gmsh.initialize()
gmsh.open("resources/meshes/bun_zipper_res3.ply")

points = np.reshape(gmsh.model.mesh.getNodes(includeBoundary=True)[1], (-1, 3))
node_tags = gmsh.model.mesh.getNodes(includeBoundary=True)[0].astype(np.int32)

print("Missing elements in curvature computation:", missing_elements(node_tags))

cloud = PyntCloud(pd.DataFrame(points, columns=["x", "y", "z"]))
k_neighbors = cloud.get_neighbors(k=10)
ev = cloud.add_scalar_field("eigen_values", k_neighbors=k_neighbors)
cloud.add_scalar_field("curvature", ev=ev)
ground_truth = cloud.points["curvature(11)"].to_numpy()
X = cloud.points[["x", "y", "z"]].to_numpy()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_axis_off()
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=ground_truth, s=1)
plt.show()

# Save node tags ground truth curvatures for bun_zipper_res3.ply
np.savetxt("resources/curvatures/bun_zipper_res3_node_tags.csv", node_tags)
np.savetxt("resources/curvatures/bun_zipper_res3.csv", ground_truth)
np.savetxt("resources/curvatures/bun_zipper_res3_coords.csv", points)
