from pytorch3d.ops import knn_points
from pytorch3d.loss import chamfer_distance
import torch
import numpy as np
import open3d as o3d
from dgl.geometry import farthest_point_sampler
import matplotlib.pyplot as plt
from quad_mesh_simplify import simplify_mesh


"Loading Data"

def get_data(file_name):

    data = np.load("resources/results/"+file_name+".npz")
    org_coords, org_faces, simp_coords, org_curv = (
        torch.from_numpy(data["org_coords"].astype(np.float32)),
        torch.from_numpy(data["org_faces"].astype(np.float32)),
        torch.from_numpy(data["simp_coords"].astype(np.float32)),
        torch.from_numpy(data["org_curv"].astype(np.float32)),
    )
    return org_coords, org_faces, simp_coords, org_curv

file_name = str(input("Enter file name (exp. bun_zipper.ply or something.csv): "))
org_coords, org_faces, simp_coords, org_curv = get_data(file_name)
target_num_points = simp_coords.size(0)
print("Size of simplified cloud: ", target_num_points)


"Different Methods"

# Random/Uniform
rand_idx = torch.randperm(org_coords.size(0))[
    :target_num_points
]
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
print("QEM starts")
qdm_simp_coords, qdm_simp_faces = simplify_mesh(
    org_coords.numpy().astype(np.float64),
    org_faces.numpy().astype(np.uint32),
    target_num_points,
)
qdm_simp_coords = torch.from_numpy(qdm_simp_coords.astype(np.float32))
print("QEM stops")


"Plots"

fig = plt.figure(figsize=plt.figaspect(1))
ax = fig.add_subplot(2,3,1, projection="3d")
ax.set_axis_off()
ax.scatter(org_coords[:, 0], org_coords[:, 1], org_coords[:, 2], s=1, c=org_curv)
k=2
for i in [simp_coords, rand_simp, tcp_simp, fps_simp, qdm_simp_coords]:
    ax = fig.add_subplot(2,3,k, projection="3d")
    ax.set_axis_off()
    ax.scatter(i[:, 0], i[:, 1], i[:, 2], s=1)
    k+=1
plt.show()


"Evaluation Metrics"

def evaluation_metrics(coords):

    # Normals Consistency
    org_pcd = o3d.geometry.PointCloud()
    org_pcd.points = o3d.utility.Vector3dVector(org_coords.numpy())
    org_pcd.estimate_normals()
    # o3d.visualization.draw_geometries([org_pcd], point_show_normal=True)
    org_norm= torch.from_numpy(np.asarray(org_pcd.normals))

    simp_pcd = o3d.geometry.PointCloud()
    simp_pcd.points = o3d.utility.Vector3dVector(coords.numpy())
    simp_pcd.estimate_normals()
    # o3d.visualization.draw_geometries([org_pcd], point_show_normal=True)
    simp_norm=torch.from_numpy(np.asarray(simp_pcd.normals))
    simp_knn = knn_points(p1=torch.unsqueeze(org_coords, 0), p2=torch.unsqueeze(coords, 0)).idx.squeeze()
    simp_knn1 = knn_points(p2=torch.unsqueeze(org_coords, 0), p1=torch.unsqueeze(coords, 0)).idx.squeeze()

    simp_n1=org_norm[simp_knn1]
    simp_n2=simp_norm[simp_knn]

    cs = torch.nn.CosineSimilarity()
    cos_sim1 = cs(simp_n1, simp_norm)
    cos_sim2 = cs(simp_n2, org_norm)

    norm_consis = 1-(torch.sum(cos_sim1)/cos_sim1.size(0)) + \
                  1-(torch.sum(cos_sim2)/cos_sim2.size(0))

    # Chamfer Distance
    chamf_dist = chamfer_distance(
        x = torch.unsqueeze(org_coords, 0), y = torch.unsqueeze(coords, 0),
        x_normals=torch.unsqueeze(org_norm, 0), y_normals=torch.unsqueeze(simp_norm, 0)
    )

    # Point-Cloud Structural Distortion Measure
    # TODO

    # Surface Reconstruction
    # TODO

    return norm_consis, chamf_dist

norm_consis, chamf_dist=evaluation_metrics(simp_coords)
print("Chamfer Distance", "Normals Consistency")
print("Proposed", chamf_dist, norm_consis)
norm_consis, chamf_dist=evaluation_metrics(rand_simp)
print("Random", chamf_dist, norm_consis)
norm_consis, chamf_dist=evaluation_metrics(tcp_simp)
print("TCP", chamf_dist, norm_consis)
norm_consis, chamf_dist=evaluation_metrics(fps_simp)
print("FPS", chamf_dist, norm_consis)
norm_consis, chamf_dist=evaluation_metrics(qdm_simp_coords)
print("QDM", chamf_dist, norm_consis)
