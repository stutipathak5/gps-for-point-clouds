import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from gp_point_clouds.algorithm import SubsetAlgorithm

# GPU initialisation (if available)
if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.enabled = True
else:
    device = "cpu"

print("Device:", device, "\n")

# Done both
# TODO introduce simplication ratio concept
# TODO assumed that target_num_points is a multiple of initial_set_size: CORRECT THIS
simp_ratio = float(input("Enter desired simplification ratio (exp. 0.01): ") or 0.01)
file_name = str(input("Enter file name (exp. armadillo): ") or "bun_zipper_res3")

total_start = time.time()

# Control flow for CloudComPy (mainly for Tom as can't install it!)
cc_flag = str(input("Use CloudComPy? Answer y or n: "))
if cc_flag == "y":
    print("Using CloudComPy....")
    import cloudComPy as cc
    from gp_point_clouds.data import get_data

    cc.initCC()

    # Get original point cloud using CloudComPy
    coords, curv, faces, volume, radius, surface = get_data(file_name)
    save_faces = True

else:
    print("Not using CloudComPy, using pre-saved curvature computations...")

    # Get original point cloud and curvatures from saved files
    data = np.loadtxt(
        "resources/curvature_cc/" + file_name + ".csv", delimiter=",", skiprows=1
    )
    coords = torch.from_numpy(data[:, :3]).double()
    curv = torch.from_numpy(data[:, 3]).double()
    save_faces = False

original_data_size = curv.shape[0]
target_num_points = int(original_data_size * simp_ratio)
initial_set_size = int(
    target_num_points / 4
)  # TODO the bigger point cloud is the bigger sh0uld be this factor 4 here, figure this out!
# initial_set_size = int(radius*1000)

if original_data_size > 15000:
    random_cloud_size = 15000
else:
    random_cloud_size = original_data_size

opt_subset_size = 100
n_iter = 50

# Initialise and run algorithm
alg = SubsetAlgorithm(
    coords,
    curv,
    target_num_points,
    random_cloud_size,
    opt_subset_size,
    n_iter,
    initial_set_size,
    device,
)
simp_coords, simp_loop_time = alg.run()
total_time = time.time() - total_start

# Plotting
fig = plt.figure(figsize=plt.figaspect(2 / 2))

# ax = fig.add_subplot(121, projection='3d')
# ax.set_axis_off()
# ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=1, c=curv)

ax = fig.add_subplot(122, projection="3d")
ax.set_axis_off()
ax.scatter(simp_coords[:, 0], simp_coords[:, 1], simp_coords[:, 2], s=1)

plt.title(
    "Time Taken for Simplification Loop: " + str(simp_loop_time) + "s"
    "\n"
    "Total Time Taken: " + str(total_time) + "s" + "\n"
    "Size of simplified cloud: " + str(target_num_points) + "\n"
    "Size of randomly selected cloud: " + str(random_cloud_size) + "\n"
    "Size of subset of original cloud used for hyperparameter estimation: "
    + str(opt_subset_size)
    + "\n"
    "Number of times hyperparameters are optimized: " + str(n_iter) + "\n"
    "Initial size of simplified cloud: " + str(initial_set_size) + "\n"
    "original point cloud size: " + str(original_data_size)
)
plt.show()

if save_faces:
    np.savez(
        "resources/results/" + file_name + ".npz",
        org_coords=coords.numpy(),
        org_faces=faces,
        simp_coords=simp_coords,
        org_curv=curv.numpy(),
    )
else:
    # NOTE - can't use evaluation metrics etc. if faces not saved to results file
    np.savez(
        "resources/results/" + file_name + ".npz",
        org_coords=coords.numpy(),
        simp_coords=simp_coords,
        org_curv=curv.numpy(),
    )
