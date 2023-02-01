import torch
import numpy as np
from jakteristics import compute_features
from pytorch3d.io import IO


def get_data(file_name, device="cpu"):
    mesh = IO().load_mesh("resources/clouds/" + file_name, device=device)
    coords, faces = mesh.verts_list()[0].double(), mesh.faces_list()[0].double()

    # TODO - we need to estimate search_radius using a bounding box in PyTorch3D
    estimated_radius = 1.0

    curv = (
        torch.from_numpy(
            compute_features(
                coords.cpu().numpy(),
                search_radius=estimated_radius,
                feature_names=["surface_variation"],
            )
        )
        .double()
        .squeeze(1)
        .to(device)
    )

    return coords, curv, faces
