import os
import time
import json
import torch
import pathlib
import numpy as np
from gp_point_clouds.algorithm import SubsetAlgorithm
from gp_point_clouds.data import get_data_jak, get_data_cc
from gp_point_clouds.metrics import compute_all_metrics
from gp_point_clouds.baselines import (
    random_simplify,
    top_curvature_simplify,
    farthest_point_simplify,
    qem_simplify,
)



mode = 1             # 0 for simp ratio mode and 1 for param mode


curv_mode = "jak"  # Backend for curvature computation; use 'cc' for CloudComPy
neigh_size = 30
max_random_cloud_size = 25000  # Max. random cloud size (using ~45k on A100 GPU)
opt_subset_size = 300
n_iter = 100

if mode == 0:
    # Define clouds to simplify and desired simplification ratios for each.
    clouds = {"bun_zipper.ply": [0.1]}  # , "lucy.ply": [0.001, 0.002]}
else:
    # Define clouds to simplify and desired target sizes for each.
    clouds = {"bun_zipper.ply": [3000, 6000]}  # , "lucy.ply": [15000, 20000]}
    initial_set_sizes = [1000, 2000]

# GPU initialisation (if available)
if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.enabled = True
else:
    device = "cpu"
print("Device:", device, "\n")

device = "cpu"   # comment for gpu use


# Loop through all clouds for all specified ratios
for cloud in clouds:
    i = 1
    for simp_ratio in clouds[cloud]:
        print("Simplification no. " + str(i))
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
            initial_set_size = initial_set_sizes[i-1]
        i += 1
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

        # 2. Evaluate metrics and benchmark against baselines

        # Compute different baseline simplification methods
        rand_simp = random_simplify(coords.to(device), target_num_points).to("cpu")
        tcp_simp = top_curvature_simplify(
            curv.to(device), coords.to(device), target_num_points
        ).to("cpu")
        fps_simp = farthest_point_simplify(coords.to(device), target_num_points).to(
            "cpu"
        )
        qdm_simp = qem_simplify(coords.cpu(), faces.cpu(), target_num_points)

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
        np.savetxt(
            os.path.join(save_dir, "fps_simp.xyz"),
            fps_simp.numpy(),
            delimiter=" ",
        )
        np.savetxt(
            os.path.join(save_dir, "qdm_simp.xyz"),
            qdm_simp.numpy(),
            delimiter=" ",
        )

        # Compute metrics for GP simplified cloud and baselines
        norm_consis_proposed, chamf_dist_proposed = compute_all_metrics(
            coords, torch.tensor(simp_coords, dtype=torch.float, device=device)
        )
        norm_consis_rand, chamf_dist_rand = compute_all_metrics(
            coords, rand_simp.to(device)
        )
        norm_consis_tcp, chamf_dist_tcp = compute_all_metrics(
            coords, tcp_simp.to(device)
        )
        norm_consis_fps, chamf_dist_fps = compute_all_metrics(
            coords, fps_simp.to(device)
        )
        norm_consis_qdm, chamf_dist_qdm = compute_all_metrics(
            coords, qdm_simp.to(device)
        )

        gp_results_dict = {
            "normals_consistency": norm_consis_proposed.item(),
            "chamfer_distance_pos": chamf_dist_proposed[0].item(),
            "chamfer_distance_norm": chamf_dist_proposed[1].item(),
        }
        rand_results_dict = {
            "normals_consistency": norm_consis_rand.item(),
            "chamfer_distance_pos": chamf_dist_rand[0].item(),
            "chamfer_distance_norm": chamf_dist_rand[1].item(),
        }
        tcp_results_dict = {
            "normals_consistency": norm_consis_tcp.item(),
            "chamfer_distance_pos": chamf_dist_tcp[0].item(),
            "chamfer_distance_norm": chamf_dist_tcp[1].item(),
        }
        fps_results_dict = {
            "normals_consistency": norm_consis_fps.item(),
            "chamfer_distance_pos": chamf_dist_fps[0].item(),
            "chamfer_distance_norm": chamf_dist_fps[1].item(),
        }
        qdm_results_dict = {
            "normals_consistency": norm_consis_qdm.item(),
            "chamfer_distance_pos": chamf_dist_qdm[0].item(),
            "chamfer_distance_norm": chamf_dist_qdm[1].item(),
        }

        with open(
            os.path.join(save_dir, "gp_metrics.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(gp_results_dict, f, ensure_ascii=False, indent=4)

        with open(
            os.path.join(save_dir, "rand_metrics.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(rand_results_dict, f, ensure_ascii=False, indent=4)

        with open(
            os.path.join(save_dir, "tcp_metrics.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(tcp_results_dict, f, ensure_ascii=False, indent=4)

        with open(
            os.path.join(save_dir, "fps_metrics.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(fps_results_dict, f, ensure_ascii=False, indent=4)

        with open(
            os.path.join(save_dir, "qdm_metrics.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(qdm_results_dict, f, ensure_ascii=False, indent=4)
