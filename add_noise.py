import torch
import numpy as np
from pytorch3d.io import IO
from pytorch3d.structures.pointclouds import Pointclouds
from jakteristics import compute_features
import matplotlib.pyplot as plt
import open3d as o3d


# file_name = "armadillo"

# mesh = IO().load_mesh("resources/clouds/" + file_name + ".ply")
# coords, faces = mesh.verts_list()[0].double(), mesh.faces_list()[0].double()
# pcd = Pointclouds(points=mesh.verts_list())
# bounding_box = pcd.get_bounding_boxes()
# diag = bounding_box[0, :, 1] - bounding_box[0, :, 0]

# noise_bb=0.05
# coords_noise = coords.numpy() + np.random.normal(0, noise_bb*diag, coords.numpy().shape)


# fig = plt.figure(figsize=plt.figaspect(2 / 2))

# ax = fig.add_subplot(122, projection="3d")
# ax.set_axis_off()
# ax.scatter(coords_noise[:, 0], coords_noise[:, 1], coords_noise[:, 2], s=0.5)
# plt.show()



# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(coords_noise)
# o3d.io.write_point_cloud("resources/clouds/" + file_name + "_noisy_"+ str(noise_bb) + ".ply", pcd)


# import torch
# from pytorch3d.loss import chamfer_distance


# x=0.02

# mesh1 = IO().load_mesh("resources/clouds/armadillo_noisy_"+ str(x) +".ply")
# pcd1 = Pointclouds(points=mesh1.verts_list())
# mesh2 = IO().load_mesh("resources/results/armadillo_noisy_"+ str(x) +"_0.04.ply")
# pcd2 = Pointclouds(points=mesh2.verts_list())

# # Calculate Chamfer distance
# chamfer_dist, _ = chamfer_distance(pcd1, pcd2)

# print(x)

# print(chamfer_dist)



import torch
from pytorch3d.loss import chamfer_distance

mesh1 = IO().load_mesh("resources/clouds/bun_zipper.ply")
pcd1 = Pointclouds(points=mesh1.verts_list())
mesh2 = IO().load_mesh("resources/results/bun_zipper_7189.ply")
pcd2 = Pointclouds(points=mesh2.verts_list())

chamfer_dist, _ = chamfer_distance(pcd1, pcd2)

print(chamfer_dist)
