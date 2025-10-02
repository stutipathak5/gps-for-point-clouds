import torch
import numpy as np


def get_data_jak(file_name, neigh_size, device="cpu"):

    from pytorch3d.io import IO
    from pytorch3d.structures.pointclouds import Pointclouds
    from jakteristics import compute_features

    mesh = IO().load_mesh("resources/clouds/" + file_name, device=device)
    coords, faces = mesh.verts_list()[0].double(), mesh.faces_list()[0].double()
    pcd = Pointclouds(points=mesh.verts_list())
    bounding_box = pcd.get_bounding_boxes()
    diag = bounding_box[0, :, 1] - bounding_box[0, :, 0]
    volume = diag[0] * diag[1] * diag[2]
    surface = volume ** (2 / 3)
    surface_per_point = surface / coords.size(0)
    radius = torch.sqrt(surface_per_point * neigh_size)
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


def get_data_jak_sf(file_name, neigh_size, device="cpu"):

    from pytorch3d.io import IO
    from pytorch3d.structures.pointclouds import Pointclouds
    from jakteristics import compute_features

    mesh = IO().load_mesh("resources/new/" + file_name, device=device)
    coords, faces = mesh.verts_list()[0].double(), mesh.faces_list()[0].double()
    pcd = Pointclouds(points=mesh.verts_list())
    bounding_box = pcd.get_bounding_boxes()
    diag = bounding_box[0, :, 1] - bounding_box[0, :, 0]
    volume = diag[0] * diag[1] * diag[2]
    surface = volume ** (2 / 3)
    surface_per_point = surface / coords.size(0)
    radius = torch.sqrt(surface_per_point * neigh_size)
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


def get_data_cc(file_name, neigh_size):

    import cloudComPy as cc

    if file_name.endswith(".ply"):
        mesh = cc.loadMesh("resources/clouds/" + file_name)
        faces = mesh.IndexesToNpArray()
        cloud = mesh.getAssociatedCloud()
    else:
        cloud = cc.loadPointCloud("resources/clouds/" + file_name)
    coords = torch.from_numpy(cloud.toNpArrayCopy()).double()
    bounding_box = cloud.getOwnBB()
    bb_min = np.array(bounding_box.minCorner())
    bb_max = np.array(bounding_box.maxCorner())
    diag = bb_max - bb_min
    volume = diag[0] * diag[1] * diag[2]
    surface = volume ** (2 / 3)
    surface_per_point = surface / coords.size(0)
    radius = np.sqrt(surface_per_point * neigh_size)
    # cc.computeNormals([cloud])
    cc.computeCurvature(cc.CurvatureType.NORMAL_CHANGE_RATE, radius, [cloud])
    nsf = cloud.getNumberOfScalarFields()
    # norm = torch.from_numpy(cloud.getScalarField(nsf - 2).toNpArray()).double()
    curv = torch.from_numpy(cloud.getScalarField(nsf - 1).toNpArray()).double()

    return coords, curv, faces
