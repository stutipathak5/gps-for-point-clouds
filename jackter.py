import open3d as o3d
from jakteristics import compute_features
import gpytorch
import torch
from geometric_kernels.frontends.pytorch.gpytorch import GPytorchGeometricKernel
from geometric_kernels.kernels import MaternKarhunenLoeveKernel
import matplotlib.pyplot as plt
from dgl.geometry import farthest_point_sampler
import numpy as np
from spaces import PointCloud
import time

def get_data(file_name):
    mesh = o3d.io.read_triangle_mesh("resources/clouds/"+file_name)
    coords = torch.from_numpy(np.asarray(mesh.vertices)).double()
    faces = np.asarray(mesh.triangles)
    curv = torch.from_numpy(compute_features(coords, search_radius=0.15, feature_names=["surface_variation"])).double()
    return coords, curv, faces

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

mode = int(input("Enter 0 for simp_ratio mode and 1 for parameters mode: "))

if mode==0:
    simp_ratio = float(input("Enter desired simplification ratio (exp. 0.01): "))
    file_name = str(input("Enter file name (exp. armadillo): "))
    # Get original point cloud
    coords, curv, faces = get_data(file_name)
    original_data_size = curv.shape[0]
    if original_data_size > 15000:
        random_cloud_size = 15000
    else:
        random_cloud_size = original_data_size
    target_num_points = int(original_data_size * simp_ratio)
    initial_set_size = int(target_num_points / 3)
    # initial_set_size = int(radius*1000)
    opt_subset_size = 100
    n_iter = 100
else:
    target_num_points = int(input("Enter desired size of simplified cloud (exp. 5000): "))
    random_cloud_size = int(input("Enter desired size of randomly selected cloud (exp. 20000 (max)): "))
    file_name = str(input("Enter file name (exp. armadillo): "))
    opt_subset_size = int(input("Enter size of subset of original cloud to be used for hyperparameter estimation(exp. 200): "))
    n_iter = int(input("Enter number of times hyperparameters need to be optimized (exp. 100): "))
    initial_set_size = int(input("Enter initial size of simplified cloud (exp. 1000): "))
    # Get original point cloud
    coords, curv, faces= get_data(file_name)
    original_data_size = curv.shape[0]

st1 = time.time()

# Select smaller random point cloud
X_idx = torch.arange(random_cloud_size)
random_idx = torch.randperm(coords.size(0))[:random_cloud_size]  # randperm gives a tensor with randomly arranged values from 0 to n-1, [:n] selects first 5 values
X_coords = coords[random_idx]
y = curv[random_idx]
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
likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
likelihood.noise = torch.tensor(1e-5)
likelihood.double()

# Initialise model and hyperparameters
model = GPModel(X_train, y_train, likelihood, geometric_kernel)
hypers = {"covar_module.base_kernel.lengthscale": torch.tensor(1.0), "covar_module.base_kernel.nu": torch.tensor(10000)}  # NOTE - EQ (i.e. 5/2) seems to capture curvature best, but can try 1/2 and 3/2 too
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
for i in range(n_iter+1):
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
print("Desired size of simplified cloud (based on simplification ratio): ",target_num_points)
remainder_set_idx = torch.tensor([x for x in range(X["idx"].size(0))])
active_set_idx = torch.squeeze(farthest_point_sampler(torch.unsqueeze(X["coords"], 0), initial_set_size), 0)
remainder_set_idx = remainder_set_idx[[i for i in remainder_set_idx.tolist() if i not in active_set_idx.tolist()]]
# 2. Iterate over (number of points we wish to use to represent our point cloud)/10
st = time.time()
active_set_size = initial_set_size
i=1
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
    # TODO - two approaches for same thing: one inside loop one outside, FIX IT!
    selection_metric = torch.sqrt(torch.diag(sigma_t)) + torch.abs(mu_t - y_r)
    set_size = int(initial_set_size/3) + int(initial_set_size/5)*i                                  #TODO see this!
    if set_size+active_set_size >= target_num_points:
        set_size = target_num_points - active_set_size
    idx_to_remove_from_remainder_set = torch.topk(selection_metric, set_size)[1]
    active_set_idx = torch.cat((active_set_idx, remainder_set_idx[idx_to_remove_from_remainder_set.tolist()]))
    remainder_set_idx = remainder_set_idx[np.delete(np.arange(len(remainder_set_idx)), idx_to_remove_from_remainder_set.tolist())]
    active_set_size = active_set_idx.size(0)
    i = +1

et = time.time()
et1 = time.time()

simp_coords = np.stack((X["coords"][active_set_idx, 0], X["coords"][active_set_idx, 1], X["coords"][active_set_idx, 2]), axis=1)

#plots
fig = plt.figure(figsize=plt.figaspect(2/2))
# ax = fig.add_subplot(121, projection='3d')
# ax.set_axis_off()
# ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=1, c=curv)

ax = fig.add_subplot(122, projection='3d')
ax.set_axis_off()
ax.scatter(simp_coords[:, 0], simp_coords[:, 1], simp_coords[:, 2], s=1)

plt.title(
    "Time Taken for Simplification Loop: "+str(et-st)+"s" "\n"
    "Total Time Taken: "+ str(et1-st1)+"s"+"\n"
    "Size of simplified cloud: "+ str(target_num_points)+"\n"
    "Size of randomly selected cloud: "+ str(random_cloud_size)+"\n"
    "Size of subset of original cloud used for hyperparameter estimation: "+ str(opt_subset_size)+"\n"
    "Number of times hyperparameters are optimized: "+ str(n_iter)+"\n"
    "Initial size of simplified cloud: "+ str(initial_set_size)+"\n"
    "original point cloud size: "+ str(original_data_size)
)
plt.show()
