import torch
import numpy as np
import matplotlib.pyplot as plt

from gp_point_clouds.baselines import (
    random_simplify,
    top_curvature_simplify,
    farthest_point_simplify,
    qem_simplify,
)

from gp_point_clouds.metrics import compute_all_metrics

# GPU initialisation (if available)
if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.enabled = True
else:
    device = "cpu"

print("Device:", device, "\n")

# 0. Loading data
def get_data(file_name):

    data = np.load("resources/results/" + file_name + ".npz")
    org_coords, org_faces, simp_coords, org_curv = (
        torch.from_numpy(data["org_coords"].astype(np.float32)),
        torch.from_numpy(data["org_faces"].astype(np.float32)),
        torch.from_numpy(data["simp_coords"].astype(np.float32)),
        torch.from_numpy(data["org_curv"].astype(np.float32)),
    )
    return org_coords, org_faces, simp_coords, org_curv


file_name = str(
    input("Enter file name (exp. dragon_vrip_0.03): ")
    or "dragon_vrip_0.03"
)
org_coords, org_faces, simp_coords, org_curv = get_data(file_name)
target_num_points = simp_coords.size(0)
print("Size of simplified cloud: ", target_num_points)

# 1. Compute and plot different baseline simplification methods
rand_simp = random_simplify(org_coords.to(device), target_num_points).to("cpu")
tcp_simp = top_curvature_simplify(
    org_curv.to(device), org_coords.to(device), target_num_points
).to("cpu")
fps_simp = farthest_point_simplify(org_coords.to(device), target_num_points).to("cpu")
# qdm_simp_coords = qem_simplify(org_coords, org_faces, target_num_points)

np.savetxt("resources/results/" + file_name + "_rand.xyz", rand_simp.numpy(), delimiter=" ")
np.savetxt("resources/results/" + file_name + "_tcp.xyz", tcp_simp.numpy(), delimiter=" ")
np.savetxt("resources/results/" + file_name + "_fps.xyz", fps_simp.numpy(), delimiter=" ")
# np.savetxt("resources/results/" + file_name + "_qdm.xyz", qdm_simp_coords.numpy(), delimiter=" ")

fig = plt.figure(figsize=plt.figaspect(1))
ax = fig.add_subplot(2, 3, 1, projection="3d")
ax.set_axis_off()
ax.scatter(org_coords[:, 0], org_coords[:, 1], org_coords[:, 2], s=1, c=org_curv)
k = 2
for i in [simp_coords, rand_simp, tcp_simp, fps_simp]:
    ax = fig.add_subplot(2, 3, k, projection="3d")
    ax.set_axis_off()
    ax.scatter(i[:, 0], i[:, 1], i[:, 2], s=1)
    k += 1
plt.show()


# 2. Compute evaluation metrics
# norm_consis_proposed, chamf_dist_proposed = compute_all_metrics(org_coords, simp_coords)
# norm_consis_rand, chamf_dist_rand = compute_all_metrics(org_coords, rand_simp)
# norm_consis_tcp, chamf_dist_tcp = compute_all_metrics(org_coords, tcp_simp)
# norm_consis_fps, chamf_dist_fps = compute_all_metrics(org_coords, fps_simp)
# norm_consis_qdm, chamf_dist_qdm = compute_all_metrics(org_coords, qdm_simp_coords)
#
# print("Chamfer Distance", "Normals Consistency")
# print("Proposed", chamf_dist_proposed, norm_consis_proposed)
# print("Random", chamf_dist_rand, norm_consis_rand)
# print("TCP", chamf_dist_tcp, norm_consis_tcp)
# print("FPS", chamf_dist_fps, norm_consis_fps)
# print("QDM", chamf_dist_qdm, norm_consis_qdm)
