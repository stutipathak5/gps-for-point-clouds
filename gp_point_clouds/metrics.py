import torch
import numpy as np
# import open3d as o3d
from pytorch3d.ops import knn_points
from pytorch3d.loss import chamfer_distance


def compute_all_metrics(orig_coords, simp_coords):

    # Normals Consistency
    org_pcd = o3d.geometry.PointCloud()
    org_pcd.points = o3d.utility.Vector3dVector(orig_coords.numpy())
    org_pcd.estimate_normals()
    # o3d.visualization.draw_geometries([org_pcd], point_show_normal=True)
    org_norm = torch.from_numpy(np.asarray(org_pcd.normals))

    simp_pcd = o3d.geometry.PointCloud()
    simp_pcd.points = o3d.utility.Vector3dVector(simp_coords.numpy())
    simp_pcd.estimate_normals()
    # o3d.visualization.draw_geometries([org_pcd], point_show_normal=True)
    simp_norm = torch.from_numpy(np.asarray(simp_pcd.normals))
    simp_knn = knn_points(
        p1=torch.unsqueeze(orig_coords, 0), p2=torch.unsqueeze(simp_coords, 0)
    ).idx.squeeze()
    simp_knn1 = knn_points(
        p2=torch.unsqueeze(orig_coords, 0), p1=torch.unsqueeze(simp_coords, 0)
    ).idx.squeeze()

    simp_n1 = org_norm[simp_knn1]
    simp_n2 = simp_norm[simp_knn]

    cs = torch.nn.CosineSimilarity()
    cos_sim1 = cs(simp_n1, simp_norm)
    cos_sim2 = cs(simp_n2, org_norm)

    norm_consis = (
        1
        - (torch.sum(cos_sim1) / cos_sim1.size(0))
        + 1
        - (torch.sum(cos_sim2) / cos_sim2.size(0))
    )

    # Chamfer Distance
    chamf_dist = chamfer_distance(
        x=torch.unsqueeze(orig_coords, 0),
        y=torch.unsqueeze(simp_coords, 0),
        x_normals=torch.unsqueeze(org_norm, 0),
        y_normals=torch.unsqueeze(simp_norm, 0),
    )

    # Point-Cloud Structural Distortion Measure
    # TODO

    # Surface Reconstruction
    # TODO


    return norm_consis, chamf_dist
