import gpytorch
import numpy as np
import torch
import geometric_kernels.torch
from geometric_kernels.frontends.pytorch.gpytorch import GPytorchGeometricKernel
from geometric_kernels.kernels import MaternKarhunenLoeveKernel
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
import plotly.express as px

from spaces import PointCloud


num_eigenpairs = 500
num_inducing_points = 200
output_dir = "output"
num_samples = 8
seed = None


class SVGP(ApproximateGP):
    def __init__(self, inducing_points, kernel):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super(SVGP, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def get_data():
    data = np.loadtxt(
        "resources/curvature_pc/bun_zipper_res3.csv", skiprows=1, delimiter=","
    )
    _X_coords = torch.tensor(
        np.loadtxt(
            "resources/curvature_pc/bun_zipper_res3.csv", delimiter=",", skiprows=1
        )[:, :3]
    ).double()
    _X_idx = torch.arange(data.shape[0])
    _X = {"idx": _X_idx, "coords": _X_coords}
    _y = torch.tensor(data[:, 3])[:, None].double()
    return _X, _y


# Get data, set and plot initial locations of inducing points
X, y = get_data()
num_data = X["coords"].shape[0]
init_inducing_locations = torch.randint(
    0, X["coords"].shape[0], (num_inducing_points,)
).double()
print("Inducing inputs shape:", init_inducing_locations.shape)

# NOTE - Uncomment to plot full initial point cloud and inducing locations

# fig = px.scatter_3d(
#     x=X["coords"][:, 0],
#     y=X["coords"][:, 1],
#     z=X["coords"][:, 2],
# )
# fig.show()

# fig = px.scatter_3d(
#     x=X["coords"][init_inducing_locations.long(), 0],
#     y=X["coords"][init_inducing_locations.long(), 1],
#     z=X["coords"][init_inducing_locations.long(), 2],
# )
# fig.show()


# Initialise space for kernel
point_cloud = PointCloud(X["coords"])
print("Number of points in the cloud:", point_cloud.num_vertices)

# Construct base kernel
nu = 1 / 2.0
truncation_level = 20
base_kernel = MaternKarhunenLoeveKernel(point_cloud, truncation_level)
geometric_kernel = GPytorchGeometricKernel(base_kernel)
geometric_kernel.double()

# Build model and likelihood
likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
)
likelihood.noise = torch.tensor(1e-5)

model = SVGP(init_inducing_locations, geometric_kernel)
model.double()
likelihood.double()

# Train model
model.train()
likelihood.train()

optimizer = torch.optim.Adam(
    [
        {"params": model.parameters()},
        {"params": likelihood.parameters()},
    ],
    lr=0.01,
)

# Our loss object. We're using the VariationalELBO
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y.size(0))

n_iter = 100
for i in range(n_iter):
    optimizer.zero_grad()
    output = model(X["idx"])
    loss = -mll(output, y)
    loss.backward()
    optimizer.step()
    print("Iteration %d complete" % (i + 1))

# Evaluate model (not needed right now)
# model.eval()
# likelihood.eval()
# X_test = X
# f_preds = model(X_test)
# m, v = f_preds.mean, f_preds.variance
# m, v = m.detach().numpy(), v.detach().numpy()
# sample = f_preds.sample(sample_shape=torch.Size([1])).detach().numpy()
# X_numpy = X.numpy().astype(np.int32)

# Plot optimised inducing point locations
fig = px.scatter_3d(
    x=X["coords"][model.variational_strategy.inducing_points.round().long(), 0],
    y=X["coords"][model.variational_strategy.inducing_points.round().long(), 1],
    z=X["coords"][model.variational_strategy.inducing_points.round().long(), 2],
)
fig.show()
