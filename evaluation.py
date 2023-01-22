import sys
from chamferdist import ChamferDistance
import torch
import pytorch3d
import numpy as np
import open3d as o3d
from dgl.geometry import farthest_point_sampler
import matplotlib.pyplot as plt

# sys.path.insert(0, "quadric_mesh_simplification") # TODO - remove if not needed?
from quad_mesh_simplify import simplify_mesh

data = np.load("resources/results/" + "bun_zipper.ply" + ".npz")
org_coords, org_faces, simp_coords, org_curv = (
    torch.from_numpy(data["org_coords"].astype(np.float32)),
    torch.from_numpy(data["org_faces"].astype(np.float32)),
    torch.from_numpy(data["simp_coords"].astype(np.float32)),
    torch.from_numpy(data["org_curv"].astype(np.float32)),
)

target_num_points = simp_coords.size(0)
print(target_num_points)
"Different Methods"

# Random/Uniform
rand_idx = torch.randperm(org_coords.size(0))[
    :target_num_points
]  # randperm gives a tensor with randomly arranged values from 0 to n-1, [:n] selects first 5 values
rand_simp = org_coords[rand_idx]

# Top Curvature Points
tcp_idx = torch.topk(org_curv, target_num_points)[1]
tcp_simp = org_coords[tcp_idx]

# Farthest Point Sampling
fps_idx = torch.squeeze(
    farthest_point_sampler(torch.unsqueeze(org_coords, 0), target_num_points), 0
)
fps_simp = org_coords[fps_idx]

# Quadric Error Metric
qdm_simp_coords, qdm_simp_faces = simplify_mesh(
    org_coords.numpy().astype(np.float64),
    org_faces.numpy().astype(np.uint32),
    target_num_points,
)

# plots
fig = plt.figure(figsize=plt.figaspect(1))

ax = fig.add_subplot(231, projection="3d")
ax.set_axis_off()
ax.scatter(org_coords[:, 0], org_coords[:, 1], org_coords[:, 2], s=1, c=org_curv)

ax = fig.add_subplot(232, projection="3d")
ax.set_axis_off()
ax.scatter(simp_coords[:, 0], simp_coords[:, 1], simp_coords[:, 2], s=1)

ax = fig.add_subplot(233, projection="3d")
ax.set_axis_off()
ax.scatter(rand_simp[:, 0], rand_simp[:, 1], rand_simp[:, 2], s=1)

ax = fig.add_subplot(234, projection="3d")
ax.set_axis_off()
ax.scatter(tcp_simp[:, 0], tcp_simp[:, 1], tcp_simp[:, 2], s=1)

ax = fig.add_subplot(235, projection="3d")
ax.set_axis_off()
ax.scatter(fps_simp[:, 0], fps_simp[:, 1], fps_simp[:, 2], s=1)

ax = fig.add_subplot(236, projection="3d")
ax.set_axis_off()
ax.scatter(qdm_simp_coords[:, 0], qdm_simp_coords[:, 1], qdm_simp_coords[:, 2], s=1)

plt.show()

# TODO Make a class in final_subset_gp

"Evaluation Metrics"

# Point-Cloud Structural Distortion Measure
# TODO

# Chamfer Distance
chamferDist = ChamferDistance()
chamf_dist = chamferDist(
    torch.unsqueeze(org_coords, 0), torch.unsqueeze(simp_coords, 0), bidirectional=True
)

# Normals Consistency (TODO - use normals to get (15) in Potamias paper)
o3d_org_pc = o3d.geometry.PointCloud(bunny_mesh.vertices)
o3d_simp_pc = o3d.geometry.PointCloud(qdm.vertices)
o3d_org_pc.estimate_normals()
o3d_simp_pc.estimate_normals()
print(np.asarray(o3d_simp_pc.normals))
