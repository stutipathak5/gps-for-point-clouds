import os
import gpytorch
import torch
import geometric_kernels.torch
import numpy as np
import plotly.graph_objects as go
from geometric_kernels.frontends.pytorch.gpytorch import GPytorchGeometricKernel
from geometric_kernels.spaces.mesh import Mesh
from geometric_kernels.kernels import MaternKarhunenLoeveKernel

num_eigenpairs = 500
output_dir = "output"
num_samples = 8
seed = None


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x):  # pylint: disable=arguments-differ
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def update_figure(fig):
    """Utility to clean up figure"""
    fig.update_layout(scene_aspectmode="cube")
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    # fig.update_traces(showscale=False, hoverinfo="none")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, visible=False),
            yaxis=dict(showbackground=False, showticklabels=False, visible=False),
            zaxis=dict(showbackground=False, showticklabels=False, visible=False),
        )
    )
    return fig


def plot_mesh(mesh: Mesh, vertices_colors=None):
    plot = go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        colorscale="Viridis",
        intensity=vertices_colors,
    )
    return plot


def get_data():
    _X = torch.tensor(np.loadtxt('resources/curvatures/bun_zipper_res3_node_tags.csv')).int()
    _y = torch.tensor(np.loadtxt('resources/curvatures/bun_zipper_res3.csv'))
    # scale y to be in range [0, 1]
    _y = (_y - torch.min(_y)) / (torch.max(_y) - torch.min(_y))
    return _X, _y

mesh = Mesh.load_mesh("resources/meshes/bun_zipper_res3.ply")
print("Number of vertices in the mesh:", mesh.num_vertices)
plot = plot_mesh(mesh)
fig = go.Figure(plot)
update_figure(fig)

# Build model
nu = 1 / 2.0
truncation_level = 20
base_kernel = MaternKarhunenLoeveKernel(mesh, truncation_level)
geometric_kernel = GPytorchGeometricKernel(base_kernel)
geometric_kernel.double()

gaussian = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
)
gaussian.noise = torch.tensor(1e-5)

X, y = get_data()
num_data = X.shape[0]
print("Number of vertices in training data:", num_data)
model = ExactGPModel(X, y, gaussian, geometric_kernel)
model.double()
gaussian.double()
model.eval()

# Evaluate model
X_test = torch.arange(mesh.num_vertices).int()
f_preds = model(X_test)
m, v = f_preds.mean, f_preds.variance
m, v = m.detach().numpy(), v.detach().numpy()
sample = f_preds.sample(sample_shape=torch.Size([1])).detach().numpy()

X_numpy = X.numpy().astype(np.int32)

# Plot predictions and samples
prediction_plot = plot_mesh(mesh, vertices_colors=m)
data_plot = go.Scatter3d(
    x=mesh.vertices[X.ravel()][:, 0],
    y=mesh.vertices[X.ravel()][:, 1],
    z=mesh.vertices[X.ravel()][:, 2],
    marker=dict(
        size=12,
        color=y.ravel(),  # set color to an array/list of desired values
        colorscale="Viridis",  # choose a colorscale
        opacity=0.8,
        cmin=m.min(),
        cmax=m.max(),
    ),
)
fig = go.Figure(data=[prediction_plot, data_plot])
fig = update_figure(fig)
out_str = output_dir + "preds"
fig.write_image(out_str, "pdf")
