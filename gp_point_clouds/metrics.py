import torch
import numpy as np
from pytorch3d.ops import knn_points
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures.pointclouds import Pointclouds


def compute_all_metrics(orig_coords, simp_coords):

    # 1. Normals Consistency
    orig_pc = Pointclouds(points=orig_coords[None, :, :].float())
    orig_pc.estimate_normals(assign_to_self=True)
    orig_norm = orig_pc.normals_packed()

    simp_pc = Pointclouds(points=simp_coords[None, :, :].float())
    simp_pc.estimate_normals(assign_to_self=True)
    simp_norm = simp_pc.normals_packed()

    simp_knn = knn_points(
        p1=orig_pc.points_padded(), p2=simp_pc.points_padded()
    ).idx.squeeze()
    simp_knn1 = knn_points(
        p2=orig_pc.points_padded(), p1=simp_pc.points_padded()
    ).idx.squeeze()

    simp_n1 = orig_norm[simp_knn1]
    simp_n2 = simp_norm[simp_knn]

    cs = torch.nn.CosineSimilarity()
    cos_sim1 = cs(simp_n1, simp_norm)
    cos_sim2 = cs(simp_n2, orig_norm)

    norm_consis = (
        1
        - (torch.sum(cos_sim1) / cos_sim1.size(0))
        + 1
        - (torch.sum(cos_sim2) / cos_sim2.size(0))
    )

    # 2. Chamfer Distance
    chamf_dist = chamfer_distance(
        x=torch.unsqueeze(orig_coords, 0).float().to("cpu"),
        y=torch.unsqueeze(simp_coords, 0).float().to("cpu"),
        x_normals=torch.unsqueeze(orig_norm, 0).float().to("cpu"),
        y_normals=torch.unsqueeze(simp_norm, 0).float().to("cpu"),
    )

    # 3. Point-Cloud Structural Distortion Measure
    # TODO

    # 4. Surface Reconstruction
    # TODO

    return norm_consis, chamf_dist
