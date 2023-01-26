import torch
import numpy as np
import cloudComPy as cc


def get_data(file_name):
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
    cc.computeCurvature(
        cc.CurvatureType.NORMAL_CHANGE_RATE, radius, [cloud]
    )  # compute curvature as a scalar field
    nsf = cloud.getNumberOfScalarFields()
    # norm = torch.from_numpy(cloud.getScalarField(nsf - 2).toNpArray()).double()
    curv = torch.from_numpy(cloud.getScalarField(nsf - 1).toNpArray()).double()

    return coords, curv, faces, volume, radius, surface
