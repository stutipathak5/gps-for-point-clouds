import torch
import numpy as np
import time


def get_data_jak(file_name, device="cpu"):

    from pytorch3d.io import IO
    from pytorch3d.structures.pointclouds import Pointclouds
    from jakteristics import compute_features

    mesh = IO().load_mesh("resources/clouds/" + file_name, device=device)
    coords, faces = mesh.verts_list()[0].double(), mesh.faces_list()[0].double()
    pcd = Pointclouds(points=mesh.verts_list())
    bounding_box = pcd.get_bounding_boxes()
    # bb_min, bb_max = torch.tensor(bounding_box[0,:,0], device=device), torch.tensor(bounding_box[0,:,1], device=device)
    diag = bounding_box[0,:,1] - bounding_box[0,:,0]
    volume = diag[0] * diag[1] * diag[2]
    surface = volume ** (2 / 3)
    surface_per_point = surface / coords.size(0)
    radius = torch.sqrt(surface_per_point * 12)
    t1=time.time()
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
    t2 = time.time()
    print("time jak", t2-t1)

    return coords, curv, faces



def get_data_cc(file_name):

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
    radius = np.sqrt(surface_per_point * 12)
    # cc.computeNormals([cloud])
    t1 = time.time()
    cc.computeCurvature(
        cc.CurvatureType.NORMAL_CHANGE_RATE, radius, [cloud]
    )  # compute curvature as a scalar field
    t2 = time.time()
    nsf = cloud.getNumberOfScalarFields()
    # norm = torch.from_numpy(cloud.getScalarField(nsf - 2).toNpArray()).double()
    curv = torch.from_numpy(cloud.getScalarField(nsf - 1).toNpArray()).double()
    print("time cc", t2-t1)

    return coords, curv, faces, volume, radius, surface


