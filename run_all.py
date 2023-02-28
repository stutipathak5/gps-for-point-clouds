import os
import time
import json
import torch
import pathlib
import numpy as np
from gp_point_clouds.algorithm import SubsetAlgorithm
from gp_point_clouds.data import get_data_jak, get_data_cc
from gp_point_clouds.baselines import (
    random_simplify,
    top_curvature_simplify,
)


"cloud name: number of vertices (for reference from largest to smallest)"
# lucy: 1,40,27,872
# statuette: 49,99,996
# asian_dragon: 36,09,600
# manuscript: 21,55,617
# happy_vrip: 5,43,652
# dragon_vrip: 4,37,645
# armadillo: 1,72,974
# bun_zipper: 35,947


mode = 0  # 0 for simp ratio mode and 1 for param mode

curv_mode = "jak"  # Backend for curvature computation; use 'cc' for CloudComPy
neigh_size = 25  # neighbourhood size for curvature computation
max_random_cloud_size = 40000  # Max. random cloud size (using ~45k on A100 GPU)
opt_subset_size = 200
n_iter = 100

if mode == 0:
    # Define clouds to simplify and desired simplification ratios for each.
    clouds = {
        "lucy.ply": [0.001, 0.002],
    }
    #     "armadillo.ply": [],
    #     "bun_zipper.ply": [],
    #     "asian_dragon.ply": [],
    #     "happy_vrip.ply": [],
    #     "dragon_vrip.ply": [],
    #     "manuscript.ply": [],
    #     "statuette.ply": [],
    # }

else:
    # Define clouds to simplify and desired target sizes for each.
    clouds = {"bun_zipper.ply": [3594], "armadillo.ply": [17297]}
    initial_set_sizes = [[1200], [2000]]

# GPU initialisation (if available)
if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.enabled = True
else:
    device = "cpu"
print("Device:", device, "\n")

# device = "cpu"   # comment for gpu use


# Loop through all clouds for all specified ratios
i = 1
for cloud in clouds:
    j = 1
    for simp_ratio in clouds[cloud]:
        print("Simplification no. " + str(j))
        # 1. Run algorithm and store results

        # Get original point cloud
        curv_start = time.time()
        if curv_mode == "cc":
            coords, curv, faces = get_data_cc(cloud, neigh_size)
        elif curv_mode == "jak":
            coords, curv, faces = get_data_jak(cloud, neigh_size, device=device)
        else:
            raise NotImplementedError("Select valid curvature backend from [jak, cc]")
        original_data_size = curv.shape[0]
        curv_time = time.time() - curv_start

        simp_start = time.time()
        # Define algorithm parameters
        if original_data_size > max_random_cloud_size:
            random_cloud_size = max_random_cloud_size
        else:
            random_cloud_size = original_data_size

        if mode == 0:
            target_num_points = int(original_data_size * simp_ratio)
            initial_set_size = int(target_num_points / 3)
        else:
            target_num_points = simp_ratio
            initial_set_size = initial_set_sizes[i - 1][j - 1]
        j += 1
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
        simp_time = time.time() - simp_start
        total_time = simp_time + curv_time

        # Output results and run info (timings, sizes, etc.)
        save_dir = os.path.join(
            "resources/results", cloud.replace(".ply", ""), str(simp_ratio)
        )
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

        run_info = {
            "simp_loop_time": simp_loop_time,
            "total_time": total_time,
            "original_cloud_size": original_data_size,
            "simp_cloud_size": target_num_points,
            "random_cloud_size": random_cloud_size,
            "initial_set_size": initial_set_size,
            "hp_opt_subset_size": opt_subset_size,
            "hp_opt_n_iter": n_iter,
        }

        with open(
            os.path.join(save_dir, "gp_run_info.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(run_info, f, ensure_ascii=False, indent=4)

        np.savetxt(
            os.path.join(save_dir, "gp_simp.xyz"),
            simp_coords,
            delimiter=" ",
        )
        np.savez(
            os.path.join(save_dir, "gp_simp.npz"),
            orig_coords=coords.cpu().numpy(),
            orig_faces=faces.cpu().numpy(),
            simp_coords=simp_coords,
            orig_curv=curv.cpu().numpy(),
        )

        # 2. Compute random and TCP baselines (others are in other_simp_methods.py)

        # Compute different baseline simplification methods
        rand_simp = random_simplify(coords.to(device), target_num_points).to("cpu")
        tcp_simp = top_curvature_simplify(
            curv.to(device), coords.to(device), target_num_points
        ).to("cpu")

        np.savetxt(
            os.path.join(save_dir, "rand_simp.xyz"),
            rand_simp.numpy(),
            delimiter=" ",
        )
        np.savetxt(
            os.path.join(save_dir, "tcp_simp.xyz"),
            tcp_simp.numpy(),
            delimiter=" ",
        )
    i += 1
