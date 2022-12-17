import gpytorch
import time
import numpy as np
import torch
import geometric_kernels.torch
from geometric_kernels.frontends.pytorch.gpytorch import GPytorchGeometricKernel
from geometric_kernels.kernels import MaternKarhunenLoeveKernel
import plotly.express as px
import linear_operator

from spaces import PointCloud

num_eigenpairs = 500
target_num_points = 100
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

    # Normalise targets to zero mean and unit variance
    y = torch.tensor(data[:, 3]).double()
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
opt_subset_size = 100
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
    "covar_module.base_kernel.nu": torch.tensor(
        5.0 / 2.0
    ),  # NOTE - EQ (i.e. 5/2) seems to capture curvature best, but can try 1/2 and 3/2 too
}
model.initialize(**hypers)
model.double()

"""Implement greedy SoD algorithm for GP regression
(see 'A Fast and Greedy Subset-of-Data (SoD) Scheme for Sparsification in Gaussian processes')
"""
n_iter = 100
model.train()
likelihood.train()

# 0. Pre-compute kernel hyperparameters on a randomly selected subset of our data
opt_start = time.time()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
print("Estimating hyperparameters...")
for i in range(n_iter):
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
opt_end = time.time()

model.eval()
likelihood.eval()
_ = torch.set_grad_enabled(False)  # Disable grad tracking now HPs are optimised

# 1. Compute kernel with all inputs, which we will use for updates later (see A.1 of SoD paper)
K_full = model.covar_module(X["idx"], X["idx"])
K_ri, K_ii_plus_noise, K_ii_plus_noise_inv = None, None, None

# 2. Randomly select an initial observation, add to active set and
#    remove from the remainder set
sim_start = time.time()
remainder_set_idx = [x for x in range(X["idx"].size(0))]
active_set_idx = []
idx_to_remove_from_remainder_set = np.random.randint(0, X["idx"].size(0))
idx_to_add_to_active_set = idx_to_remove_from_remainder_set
active_set_idx.append(idx_to_add_to_active_set)
remainder_set_idx.pop(idx_to_remove_from_remainder_set)

# 3. Iterate over number of points we wish to use to represent our point cloud
for _ in range(target_num_points - 1):
    X_i = X["idx"][active_set_idx]
    X_r = X["idx"][remainder_set_idx]
    y_i = y[active_set_idx]
    y_r = y[remainder_set_idx]

    # 4. Update posterior (see Eq. 3 and 4 in SoD paper)

    # Compute or update cross covariance matrix. In latter case, K_ri is ((N-t) x t)
    # and we drop a row and add a column to make it ((N-(t+1)) x (t+1))

    # TODO - bypassing this for now to focus on other stuff, need to fix (or alternatively,
    #        if it's not too expensive we can just directly compute, need to test this out).
    #        The set indices make this more confusing than it looks, probably because of how
    #        I've implemented it.
    # print("CHECKPOINT 1")
    if True:  # K_ri is None:
        K_ri = model.covar_module(X_r, X_i)
    else:
        K_ri_new_col = model.covar_module(
            X["idx"][idx_to_add_to_active_set].unsqueeze(0), X_r
        ).T
        K_ri = gpytorch.lazy.CatLazyTensor(
            [K_ri[remainder_set_idx, :], K_ri_new_col], dim=1
        )

    # Compute or update active set covariance matrix, this grows by 1 row and 1 column
    # at each iteration. We update the inverse using block inversion (Schur complement).
    # print("CHECKPOINT 2")
    if True:
        # if K_ii_plus_noise is None:
        K_ii_plus_noise = (
            model.covar_module(X_i, X_i)
            + torch.eye(len(active_set_idx)) * likelihood.noise
        )
        # Compute lower triangular of Cholesky factor and invert it, then use to get K_ii
        K_ii_root_inv = K_ii_plus_noise.root_inv_decomposition()
        K_ii_plus_noise_inv = K_ii_root_inv.matmul(K_ii_root_inv.T)
    else:
        K_ii_new_row = model.covar_module(
            X["idx"][idx_to_add_to_active_set].unsqueeze(0), X_i
        )
        K_ii_plus_noise = gpytorch.lazy.CatLazyTensor(
            K_ii_plus_noise, K_ii_new_row[:, :-1], dim=0
        )
        K_ii_plus_noise = gpytorch.lazy.CatLazyTensor(
            K_ii_plus_noise, K_ii_new_row.T, dim=1
        )
        noise_diag = torch.zeros(K_ii_plus_noise.shape[0])
        noise_diag[-1] += likelihood.noise.item()
        K_ii_plus_noise.add_diag(noise_diag)

        # Turned debug mode off as there was a bug in GPyTorch which was stopping evaluation
        # as it thought the matrices were of an incorrect shape (something similar to
        # https://github.com/cornellius-gp/gpytorch/issues/1554 perhaps)
        print("CHECKPOINT 3")
        with gpytorch.settings.debug(False) and linear_operator.settings.debug(False):
            # Perform inverse update with block inversion,
            # see https://en.wikipedia.org/wiki/Invertible_matrix#Blockwise_inversion).
            A = K_ii_plus_noise[:-1, :-1]
            A_inv = K_ii_plus_noise_inv
            B = K_ii_new_row.T[:-1, :]
            C = K_ii_new_row[:, :-1]
            D = K_ii_new_row[-1:, -1:]
            schur_comp = D - C.matmul(A_inv).matmul(B)
            schur_comp_root_inv = schur_comp.root_inv_decomposition()
            schur_comp_inv = schur_comp_root_inv.matmul(schur_comp_root_inv.T)
            top_left = A_inv + A_inv.matmul(B).matmul(schur_comp_inv).matmul(C).matmul(
                A_inv
            )
            bottom_left = schur_comp_inv.mul(-1.0).matmul(C).matmul(A_inv)
            top_right = A_inv.mul(-1.0).matmul(B).matmul(schur_comp_inv)
            bottom_right = schur_comp_inv
            bottom = gpytorch.lazy.CatLazyTensor(bottom_left, bottom_right, dim=1)
            top = gpytorch.lazy.CatLazyTensor(top_right, top_left, dim=1)
            K_ii_plus_noise_inv = gpytorch.lazy.CatLazyTensor(top, bottom, dim=0)
    # print("CHECKPOINT 4")

    # Update remainder set covariance matrix, this shrinks by 1 row and 1 column at each
    # iteration from the full covariance we computed in Step 1.
    K_rr = K_full[remainder_set_idx, :][:, remainder_set_idx]
    # print("CHECKPOINT 5")

    # Finally, use all of the above to compute posterior mean and covariance
    mu_t = K_ri.matmul(K_ii_plus_noise_inv).matmul(y_i)
    sigma_t = K_rr - K_ri.matmul(K_ii_plus_noise_inv).matmul(K_ri.T)
    # print("CHECKPOINT 6")

    # 5. Compute selection metric and select next observation
    selection_metric = torch.sqrt(sigma_t.diag()) + torch.abs(mu_t - y_r)

    # TODO - the above is the bottleneck because it has to fully evaluate sigma_t; if you remove the '- K_ri.matmul(...'
    #        after K_rr in sigma_t then the whole thing runs in 3s. Need to implement suggestions from paper
    #        to speed things up (old code runs in 15s-ish)

    # print("CHECKPOINT 7\n")
    idx_to_remove_from_remainder_set = torch.argmax(selection_metric).item()
    idx_to_add_to_active_set = remainder_set_idx[idx_to_remove_from_remainder_set]
    remainder_set_idx.pop(idx_to_remove_from_remainder_set)
    active_set_idx.append(idx_to_add_to_active_set)

sim_end = time.time()
print("Hyperparameter Optimisation Runtime: %.2fs" % (opt_end - opt_start))
print("Simplification Runtime: %.2fs" % (sim_end - sim_start))

# Plot chosen locations
fig = px.scatter_3d(
    x=X["coords"][active_set_idx, 0],
    y=X["coords"][active_set_idx, 1],
    z=X["coords"][active_set_idx, 2],
)
fig.show()
