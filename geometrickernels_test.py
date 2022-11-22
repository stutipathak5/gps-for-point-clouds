import gpytorch
import numpy as np
import torch
import geometric_kernels.torch
from geometric_kernels.frontends.pytorch.gpytorch import GPytorchGeometricKernel
from geometric_kernels.kernels import MaternKarhunenLoeveKernel
import plotly.express as px

from spaces import PointCloud


num_eigenpairs = 500
target_num_points = 900
output_dir = "output"
num_samples = 8
seed = None


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


def get_data():
    data = np.loadtxt(
        "resources/curvature_pc/bun_zipper_res3.csv", skiprows=1, delimiter=","
    )
    X_coords = torch.tensor(
        np.loadtxt(
            "resources/curvature_pc/bun_zipper_res3.csv", delimiter=",", skiprows=1
        )[:, :3]
    ).double()
    X_idx = torch.arange(data.shape[0])
    X = {"idx": X_idx, "coords": X_coords}
    y = torch.tensor(data[:, 3]).double()

    # TODO - maybe try normalise inputs and target to be zero-mean and unit variance
    # X["idx"] = (X["idx"].double() - X["idx"].double().mean()) / X["idx"].double().std()
    y = (y - y.mean()) / y.std()

    return X, y


# Get data, set and plot initial locations of inducing points
X, y = get_data()
num_data = X["coords"].shape[0]

# NOTE - Uncomment below to plot full initial point cloud
# fig = px.scatter_3d(
#     x=X["coords"][:, 0],
#     y=X["coords"][:, 1],
#     z=X["coords"][:, 2],
# )
# fig.show()

# Select data to use for pre-computing kernel hyperparameters (can use a
# a random subset, but we just use it all here.)
opt_subset_size = 1889
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
    "covar_module.base_kernel.lengthscale": torch.tensor(0.1),
    "covar_module.base_kernel.nu": torch.tensor(
        5.0 / 2.0
    ),  # NOTE - EQ (i.e. 5/2) seems to capture curvature best, but can try 1/2 and 3/2 too
}
model.initialize(**hypers)
model.double()

"""Implement greedy SoD algorithm for GP regression
(see 'A Fast and Greedy Subset-of-Data (SoD) Scheme for Sparsification in Gaussian processes')
"""
n_iter = 1000
model.train()
likelihood.train()

# 0. Pre-compute kernel hyperparameters on a randomly selected subset of our data
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
for _ in range(n_iter):
    optimizer.zero_grad()
    output = model(X_train)
    loss = -mll(output, y_train)
    loss.backward()
    optimizer.step()
    print("LS ", model.covar_module.base_kernel.lengthscale.item())
    print("Noise ", model.likelihood.noise.item())
    print()
print("Hyperparameter estimation complete...")

model.eval()
likelihood.eval()

# 1. Randomly select an initial observation, add to active set and
#    remove from the remainder set
remainder_set_idx = [x for x in range(X["idx"].size(0))]
active_set_idx = []
initial_obs_idx = np.random.randint(0, X["idx"].size(0))
active_set_idx.append(initial_obs_idx)
remainder_set_idx.pop(initial_obs_idx)

# 2. Iterate over number of points we wish to use to represent our point cloud
for _ in range(target_num_points - 1):

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

    # 4. Compute selection metric and select next observation
    selection_metric = torch.sqrt(torch.diag(sigma_t)) + torch.abs(mu_t - y_r)

    # TODO - remove (for debugging). All entries of kernels are identical,
    # need to fix this!
    print(mu_t)
    print(sigma_t)
    print(y_r)
    print()

    idx_to_remove_from_remainder_set = torch.argmax(selection_metric).item()
    idx_to_add_to_active_set = remainder_set_idx[idx_to_remove_from_remainder_set]
    remainder_set_idx.pop(idx_to_remove_from_remainder_set)
    active_set_idx.append(idx_to_add_to_active_set)

# Plot chosen locations
fig = px.scatter_3d(
    x=X["coords"][active_set_idx, 0],
    y=X["coords"][active_set_idx, 1],
    z=X["coords"][active_set_idx, 2],
)
fig.show()
