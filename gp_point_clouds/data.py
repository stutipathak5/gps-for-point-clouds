import torch
import numpy as np
from jakteristics import compute_features
from pytorch3d.io import IO
from pytorch3d.structures.pointclouds import Pointclouds


def get_data(file_name, device="cpu"):
    mesh = IO().load_mesh("resources/clouds/" + file_name, device=device)
    coords, faces = mesh.verts_list()[0].double(), mesh.faces_list()[0].double()
    pcd = Pointclouds(points=mesh.verts_list())
    bounding_box = pcd.get_bounding_boxes()
    bb_min, bb_max = np.array(bounding_box[0,:,0]), np.array(bounding_box[0,:,1])
    diag = bb_max - bb_min
    volume = diag[0] * diag[1] * diag[2]
    surface = volume ** (2 / 3)
    surface_per_point = surface / coords.size(0)
    radius = np.sqrt(surface_per_point * 12)

    curv = (
        torch.from_numpy(
            compute_features(
                coords.cpu().numpy(),
                search_radius=radius,
                feature_names=["surface_variation"],
            )
        )
        .double()
        .squeeze(1)
        .to(device)
    )

    return coords, curv, faces
