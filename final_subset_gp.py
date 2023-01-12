import gpytorch
import torch
from geometric_kernels.frontends.pytorch.gpytorch import GPytorchGeometricKernel
from geometric_kernels.kernels import MaternKarhunenLoeveKernel
import matplotlib.pyplot as plt
from dgl.geometry import farthest_point_sampler
import numpy as np
from spaces import PointCloud
import time
import cloudComPy as cc

cc.initCC()


def get_data(file_name, curv_radius):
    if file_name.endswith(".ply"):
        mesh = cc.loadMesh("resources/clouds/" + file_name)
        cloud = mesh.getAssociatedCloud()
    else:
        cloud = cc.loadPointCloud("resources/clouds/" + file_name)
    cc.computeCurvature(
        cc.CurvatureType.NORMAL_CHANGE_RATE, curv_radius, [cloud]
    )  # compute curvature as a scalar field
    nsf = cloud.getNumberOfScalarFields()
    curv = torch.from_numpy(cloud.getScalarField(nsf - 1).toNpArray()).double()
    coords = torch.from_numpy(cloud.toNpArrayCopy()).double()

    return coords, curv


# Done both
# TODO introduce simplication ratio concept
# TODO assumed that target_num_points is a multiple of initial_set_size: CORRECT THIS


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, X, y, likelihood, kernel):
        super(GPModel, self).__init__(X, y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        """Draw samples from our GP."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


simp_ratio = float(input("Enter desired simplification ratio (exp. 0.01): "))
file_name = str(input("Enter file name (exp. armadillo): "))

st1 = time.time()

# Get original point cloud
X_coords, y = get_data(file_name, 0.002605)

# points=pd.DataFrame(X_coords, columns=['x', 'y', 'z'])
# cloud = PyntCloud(points)
# convex_hull_id = cloud.add_structure("convex_hull")
# convex_hull = cloud.structures[convex_hull_id]
# print(convex_hull.volume)

original_data_size = y.shape[0]
target_num_points = int(original_data_size * simp_ratio)
initial_set_size = int(target_num_points / 4)

if original_data_size > 15000:
    random_cloud_size = 15000
else:
    random_cloud_size = original_data_size

opt_subset_size = 100
n_iter = 50

# Select smaller random point cloud
X_idx = torch.arange(random_cloud_size)
random_idx = torch.randperm(X_coords.size(0))[
    :random_cloud_size
]  # randperm gives a tensor with randomly arranged values from 0 to n-1, [:n] selects first 5 values
X_coords = X_coords[random_idx]
y = y[random_idx]
y = torch.nan_to_num(y, nan=0.0)
X = {"idx": X_idx, "coords": X_coords}

# Select an even smaller set of random datapoints to estimate gp hyperparameters
opt_subset_idx = torch.randperm(X["idx"].size(0))[:opt_subset_size]
X_train = X["idx"][opt_subset_idx]
y_train = y[opt_subset_idx]

# Initialise space for kernel
point_cloud = PointCloud(X["coords"])

# Construct our geometric kernel
truncation_level = 20
base_kernel = MaternKarhunenLoeveKernel(point_cloud, truncation_level)
geometric_kernel = gpytorch.kernels.ScaleKernel(GPytorchGeometricKernel(base_kernel))
geometric_kernel.double()

# Initialise likelihood
likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
)
likelihood.noise = torch.tensor(1e-5)
likelihood.double()

# Initialise model and hyperparameters
model = GPModel(X_train, y_train, likelihood, geometric_kernel)
hypers = {
    "covar_module.base_kernel.lengthscale": torch.tensor(1.0),
    "covar_module.base_kernel.nu": torch.tensor(10000),
}  # NOTE - EQ (i.e. 5/2) seems to capture curvature best, but can try 1/2 and 3/2 too
model.initialize(**hypers)
model.double()

"""Implement greedy SoD algorithm for GP regression
(see 'A Fast and Greedy Subset-of-Data (SoD) Scheme for Sparsification in Gaussian processes')
"""
model.train()
likelihood.train()

# 0. Pre-compute kernel hyperparameters on a randomly selected subset of our data
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

print("Estimating hyperparameters...")
for i in range(n_iter + 1):
    optimizer.zero_grad()
    output = model(X_train)
    loss = -mll(output, y_train)
    loss.backward()
    optimizer.step()
    if i % 25 == 0:
        print("Iteration %d" % i)
        print("LS ", model.covar_module.base_kernel.lengthscale.item())
        print("Noise ", model.likelihood.noise.item())
        print()
print("Hyperparameter estimation complete.")

model.eval()
likelihood.eval()

# 1. Select multiple initial observations using farthest point sampling, add to active set and remove from the remainder set
print(
    "Desired size of simplified cloud (based on simplification ratio): ",
    target_num_points,
)
remainder_set_idx = torch.tensor([x for x in range(X["idx"].size(0))])
active_set_idx = torch.squeeze(
    farthest_point_sampler(torch.unsqueeze(X["coords"], 0), initial_set_size), 0
)
remainder_set_idx = remainder_set_idx[
    [i for i in remainder_set_idx.tolist() if i not in active_set_idx.tolist()]
]
# 2. Iterate over (number of points we wish to use to represent our point cloud)/10
st = time.time()
active_set_size = initial_set_size
i = 1
while active_set_size < target_num_points:
    print("No. of points added to simplified cloud:", active_set_size)
    # 3. Update posterior (see Eq. 3 and 4 in above paper)
    # TODO - make this quicker using updates in paper appendix (think about dense format too, avoid if possible).
    #        Another option for speed-ups later (on bigger data) is to replace the exact GP with a SVGP.
    X_i = X["idx"][active_set_idx]
    X_r = X["idx"][remainder_set_idx]
    y_i = y[active_set_idx]
    y_r = y[remainder_set_idx]
    K_ri = model.covar_module(X_r, X_i).to_dense()
    K_rr = model.covar_module(X_r, X_r).to_dense()
    K_ii = model.covar_module(X_i, X_i).to_dense()
    K_ii_plus_noise = K_ii + torch.eye(K_ii.size(0)) * likelihood.noise
    K_ii_plus_noise_inv = torch.cholesky_inverse(K_ii_plus_noise)
    mu_t = K_ri.matmul(K_ii_plus_noise_inv).matmul(y_i)
    sigma_t = K_rr - K_ri.matmul(K_ii_plus_noise_inv).matmul(K_ri.T)
    # 4. Compute selection metric and select next 10 observations (you can also increase this number independent of initial_set_size)
    selection_metric = torch.sqrt(torch.diag(sigma_t)) + torch.abs(mu_t - y_r)
    set_size = initial_set_size + int(initial_set_size / 5) * i
    if set_size + active_set_size >= target_num_points:
        set_size = target_num_points - active_set_size
    idx_to_remove_from_remainder_set = torch.topk(selection_metric, set_size)[1]
    active_set_idx = torch.cat(
        (active_set_idx, remainder_set_idx[idx_to_remove_from_remainder_set.tolist()])
    )
    remainder_set_idx = remainder_set_idx[
        np.delete(
            np.arange(len(remainder_set_idx)), idx_to_remove_from_remainder_set.tolist()
        )
    ]
    active_set_size = active_set_idx.size(0)
    i = +1

et = time.time()
et1 = time.time()

# plots
fig = plt.figure(figsize=plt.figaspect(2 / 2))
ax = fig.add_subplot(121, projection="3d")
ax.set_axis_off()
ax.scatter(X["coords"][:, 0], X["coords"][:, 1], X["coords"][:, 2], s=1, c=y)

ax = fig.add_subplot(122, projection="3d")
ax.set_axis_off()
ax.scatter(
    X["coords"][active_set_idx, 0],
    X["coords"][active_set_idx, 1],
    X["coords"][active_set_idx, 2],
    s=1,
)

plt.title(
    "Time Taken for Simplification Loop: " + str(et - st) + "s"
    "\n"
    "Total Time Taken: " + str(et1 - st1) + "s" + "\n"
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
