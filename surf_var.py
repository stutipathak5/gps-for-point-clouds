from gp_point_clouds.data import get_data_jak_sf
import torch

file_name = str(
    input("Enter file name (exp. bun_zipper.ply) in new folder: ") or "bun_zipper.ply"
)

neigh_size = int(
    input("Enter neighbourhood size for curvature computation: ")
    or 30
)


coords, curv, faces = get_data_jak_sf(file_name, neigh_size)

print(curv.shape)
mask = torch.isnan(curv) == False
curv = curv[mask]
print(curv.shape)


average = curv.mean()


print(average)