import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from gp_point_clouds.algorithm import SubsetAlgorithm
from gp_point_clouds.data import get_data

# GPU initialisation (if available)
if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.enabled = True
else:
    device = "cpu"

print("Device:", device, "\n")

# Give user option to use simplification ratio or raw parameters
mode = int(input("Enter 0 for simp_ratio mode and 1 for parameters mode: ") or 0)

file_name = str(
    input("Enter file name (exp. bun_zipper_res3.ply): ") or "bun_zipper_res3.ply"
)

total_start1 = time.time()
# Get original point cloud
coords, curv, faces = get_data(file_name, device=device)
original_data_size = curv.shape[0]
total_stop1 = time.time()
print("Original point cloud size (decide simp_ratio/params accordingly)", original_data_size)


if mode == 0:

    simp_ratio = float(
        input("Enter desired simplification ratio (exp. 0.01): ") or 0.01
    )

else:

    target_num_points = int(
        input("Enter desired size of simplified cloud (exp. 5000): ") or 5000
    )
    random_cloud_size = int(
        input("Enter desired size of randomly selected cloud (exp. 20000 (max)): ")
        or 20000
    )
    opt_subset_size = int(
        input(
            "Enter size of subset of original cloud to be used for hyperparameter estimation(exp. 200): "
        )
        or 200
    )
    n_iter = int(
        input("Enter number of times hyperparameters need to be optimized (exp. 100): ")
        or 100
    )
    initial_set_size = int(
        input("Enter initial size of simplified cloud (exp. 1000): ") or 1000
    )

total_start2 = time.time()
if mode == 0:

    if original_data_size > 15000:
        random_cloud_size = 15000
    else:
        random_cloud_size = original_data_size

    target_num_points = int(original_data_size * simp_ratio)
    initial_set_size = int(target_num_points / 3)
    # initial_set_size = int(radius*1000)
    opt_subset_size = 100
    n_iter = 100

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
total_stop2 = time.time()
total_time = total_stop2-total_start2+total_stop1-total_start1

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

np.savez(
    "resources/results/" + file_name + ".npz",
    org_coords=coords.cpu().numpy(),
    org_faces=faces.cpu().numpy(),
    simp_coords=simp_coords,
    org_curv=curv.cpu().numpy(),
)
