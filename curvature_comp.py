"curvature computation and saving using pyntcloud"

from pyntcloud import PyntCloud
import matplotlib.pyplot as plt

def curvature(no_neighbors, path):
    cloud = PyntCloud.from_file(path)
    k_neighbors = cloud.get_neighbors(k=no_neighbors)
    ev = cloud.add_scalar_field("eigen_values", k_neighbors=k_neighbors)
    cloud.add_scalar_field("curvature", ev=ev)
    cloud.points=cloud.points[["x", "y", "z", "curvature("+str(no_neighbors+1)+")"]]
    return cloud.points

# example with plot
df=curvature(60, 'resources/clouds/bun_zipper.ply')
df.to_csv('resources/curvature_pc/bun_zipper.csv', index=False)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_axis_off()
ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], c=df.iloc[:, 3], s=1)
plt.show()