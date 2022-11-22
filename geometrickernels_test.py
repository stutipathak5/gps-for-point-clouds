import gpytorch
import numpy as np
import torch
import geometric_kernels.torch
from geometric_kernels.frontends.pytorch.gpytorch import GPytorchGeometricKernel
from geometric_kernels.kernels import MaternKarhunenLoeveKernel
import plotly.express as px

from spaces import PointCloud


num_eigenpairs = 500
target_num_points = 200
output_dir = "output"
num_samples = 8
seed = None


class GreedySubsetGP(gpytorch.models.ExactGP):
    def __init__(self, X, y, likelihood, kernel, target_num_points):
        super(GreedySubsetGP, self).__init__(X, y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel
        self.likelihood = likelihood
        self.target_num_points = target_num_points
        self.X, self.y = X, y
        self.remainder_set_idx = [x for x in range(self.X.size(0))]
        self.active_set_idx = []

    def forward(self, x):
        """Draw samples from our GP."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def optimize(self, n_iter=1000, opt_subset_size=1000):
        """Implement greedy SoD algorithm for GP regression
        (see 'A Fast and Greedy Subset-of-Data (SoD) Scheme for Sparsification in Gaussian processes')
        """

        # 0. Pre-compute kernel hyperparameters on a randomly selected subset of our data
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        opt_subset_idx = torch.randint(0, self.X.size(0), (opt_subset_size,))
        for _ in range(n_iter):
            optimizer.zero_grad()
            output = self.forward(self.X[opt_subset_idx])
            loss = -mll(output, self.y[opt_subset_idx])
            loss.backward()
            optimizer.step()
        print("Hyperparameter estimation complete...")
        self.eval()

        # 1. Randomly select an initial observation, add to active set and
        #    remove from the remainder set
        initial_obs_idx = np.random.randint(0, self.X.size(0))
        self.active_set_idx.append(initial_obs_idx)
        self.remainder_set_idx.pop(initial_obs_idx)

        # 2. Iterate over number of points we wish to use to represent our point cloud
        for _ in range(self.target_num_points - 1):

            # 3. Update posterior (see Eq. 3 and 4 in above paper)
            # TODO - make this quicker using updates in paper appendix (think about dense format too, avoid if possible)
            X_i = self.X[self.active_set_idx]
            X_r = self.X[self.remainder_set_idx]
            y_i = self.y[self.active_set_idx]
            y_r = self.y[self.remainder_set_idx]

            K_ri = self.covar_module(X_r, X_i).to_dense()
            K_rr = self.covar_module(X_r, X_r).to_dense()
            K_ii = self.covar_module(X_i, X_i).to_dense()
            K_ii_plus_noise = K_ii + torch.eye(K_ii.size(0)) * self.likelihood.noise
            K_ii_plus_noise_inv = torch.cholesky_inverse(K_ii_plus_noise)

            mu_t = K_ri.matmul(K_ii_plus_noise_inv).matmul(y_i)
            sigma_t = K_rr - K_ri.matmul(K_ii_plus_noise_inv).matmul(K_ri.T)

            # 4. Compute selection metric and select next observation

            # TODO - fix issue here, selection metric should be sqrt(diag(sigma_t)) +
            #        (mu_t - y_r)**2, but diagonal of sigma_t is negative for some reason,
            #        need to figure this out!
            print(mu_t)
            print(y_r)
            # print(sigma_t)
            print()
            # selection_metric = torch.sqrt(torch.diag(sigma_t)) + torch.abs(
            #     mu_t - y_r
            # )
            selection_metric = torch.diag(sigma_t) + torch.square(torch.abs(mu_t - y_r))

            idx_to_remove_from_remainder_set = torch.argmax(selection_metric).item()
            idx_to_add_to_active_set = self.remainder_set_idx[
                idx_to_remove_from_remainder_set
            ]
            self.remainder_set_idx.pop(idx_to_remove_from_remainder_set)
            self.active_set_idx.append(idx_to_add_to_active_set)

        return


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

    # TODO - normalise inputs and target to be zero-mean and unit variance
    X["idx"] = (X["idx"] - torch.min(X["idx"])) / (
        torch.max(X["idx"]) - torch.min(X["idx"])
    )
    y = (y - torch.min(y)) / (torch.max(y) - torch.min(y))

    return X, y


# Get data, set and plot initial locations of inducing points
X, y = get_data()
num_data = X["coords"].shape[0]

# NOTE - Uncomment below to plot full initial point cloud
# fig = px.scatter_3d(
#     x=X[:, 0],
#     y=X[:, 1],
#     z=X[:, 2],
# )
# fig.show()

# Initialise space for kernel
point_cloud = PointCloud(X["coords"])
print("Number of points in the cloud:", point_cloud.num_vertices)

# Construct our geometric kernel
nu = 1 / 2.0  # TODO - vary this?
truncation_level = 20
base_kernel = MaternKarhunenLoeveKernel(point_cloud, truncation_level)
geometric_kernel = GPytorchGeometricKernel(base_kernel)
geometric_kernel.double()

# Initialise likelihood and build model
likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
)
likelihood.noise = torch.tensor(1e-5)
likelihood.double()
model = GreedySubsetGP(X["idx"], y, likelihood, geometric_kernel, target_num_points)
model.double()
model.optimize(n_iter=1000, opt_subset_size=1000)

# Plot chosen locations
fig = px.scatter_3d(
    x=X["coords"][model.active_set_idx, 0],
    y=X["coords"][model.active_set_idx, 1],
    z=X["coords"][model.active_set_idx, 2],
)
fig.show()
