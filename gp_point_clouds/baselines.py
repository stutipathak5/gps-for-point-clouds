import torch
import numpy as np
from dgl.geometry import farthest_point_sampler
from quad_mesh_simplify import simplify_mesh


def random_simplify(orig_coords, target_num_points):
    """Random/uniform sampling."""
    rand_idx = torch.randperm(orig_coords.size(0))[:target_num_points]
    rand_simp = orig_coords[rand_idx]
    return rand_simp


def top_curvature_simplify(orig_curv, orig_coords, target_num_points):
    """Simplify using top curvature points."""
    tcp_idx = torch.topk(orig_curv, target_num_points)[1]
    tcp_simp = orig_coords[tcp_idx]
    return tcp_simp


def farthest_point_simplify(orig_coords, target_num_points):
    """Simplify using farthest point sampling."""
    fps_idx = torch.squeeze(
        farthest_point_sampler(torch.unsqueeze(orig_coords, 0), target_num_points), 0
    )
    fps_simp = orig_coords[fps_idx]
    return fps_simp


def qem_simplify(orig_coords, orig_faces, target_num_points):
    """Simplify using Quadric Error Metric (QEM).

    Note, this function uses NumPy therefore cannot be used on GPUs.
    """
    qdm_simp_coords, qdm_simp_faces = simplify_mesh(
        orig_coords.numpy().astype(np.float64),
        orig_faces.numpy().astype(np.uint32),
        target_num_points,
    )
    qdm_simp_coords = torch.from_numpy(qdm_simp_coords.astype(np.float32))
    return qdm_simp_coords
