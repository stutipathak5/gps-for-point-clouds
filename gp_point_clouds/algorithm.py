import time
import torch
import gpytorch
import numpy as np
from geometric_kernels.frontends.gpytorch import GPytorchGeometricKernel
from dgl.geometry import farthest_point_sampler
from gp_point_clouds.spaces import PointCloud
from gp_point_clouds.model import GPModel
from gp_point_clouds.kernel import MaternKarhunenLoeveKernelDeviceAgnostic


class SubsetAlgorithm:
    def __init__(
        self,
        coords,
        curv,
        target_num_points,
        random_cloud_size,
        opt_subset_size,
        n_iter,
        initial_set_size,
        device,
    ):
        """
        Greedy subset-of-data algorithm for point cloud simplification using GPs
        with kernels defined on Riemannian manifolds.

        :param coords: Coordinations of points in original cloud.
        :param curv: Change of curvature defined at all points in original cloud.
        :param target_num_points: Number of points desired in simplified cloud.
        :param random_cloud_size: Number of points in original cloud to consider (useful for large clouds).
        :param opt_subset_size: Number of points to use for GP hyperparameter optimisation.
        :param n_iter: Number of iterations to use for GP hyperparameter optimisation.
        :param initial_set_size: Size of initial pool of points to be selected using FPS.
        :param device: Device to run algorithm/model on ('cpu' or 'cuda')
        """
        self.coords = coords.to(device)
        self.curv = curv.to(device)
        self.target_num_points = target_num_points
        self.random_cloud_size = random_cloud_size
        self.opt_subset_size = opt_subset_size
        self.n_iter = n_iter
        self.initial_set_size = initial_set_size
        self.device = device

        # Select smaller random point cloud
        X_idx = torch.arange(random_cloud_size, device=self.device)
        random_idx = torch.randperm(self.coords.shape[0], device=self.device)[
            :random_cloud_size
        ]  # randperm gives a tensor with randomly arranged values from 0 to n-1, [:n] selects first 5 values
        X_coords = self.coords[random_idx]
        y = self.curv[random_idx]
        self.y = torch.nan_to_num(y, nan=0.0).to(device)
        self.X = {"idx": X_idx.to(device), "coords": X_coords.to(device)}

        # Select an even smaller set of random datapoints to estimate gp hyperparameters
        opt_subset_idx = torch.randperm(self.X["idx"].shape[0])[:opt_subset_size]
        self.X_train = self.X["idx"][opt_subset_idx].to(device)
        self.y_train = self.y[opt_subset_idx].to(device)

        # Initialise space for kernel
        self.point_cloud = PointCloud(self.X["coords"])

        # Construct our geometric kernel
        truncation_level = 20
        base_kernel = MaternKarhunenLoeveKernelDeviceAgnostic(
            self.point_cloud, truncation_level, device=self.device
        )
        self.geometric_kernel = gpytorch.kernels.ScaleKernel(
            GPytorchGeometricKernel(base_kernel)
        ).to(self.device)
        self.geometric_kernel.double()

        # Initialise likelihood
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
        ).to(self.device)
        self.likelihood.noise = torch.tensor(1e-5)
        self.likelihood.double()

        # Initialise model and hyperparameters
        self.model = GPModel(
            self.X_train, self.y_train, self.likelihood, self.geometric_kernel
        ).to(self.device)
        hypers = {
            "covar_module.base_kernel.lengthscale": torch.tensor(1.0),
            "covar_module.base_kernel.nu": torch.tensor(10000),
        }  # NOTE - EQ (i.e. 5/2) seems to capture curvature best, but can try 1/2 and 3/2 too
        self.model.initialize(**hypers)
        self.model.double()

    def run(self):
        """Run algorithm with settings specified upon initialisation."""
        # 0. Pre-compute kernel hyperparameters on a randomly selected subset of our data
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        print("Estimating hyperparameters...")
        for i in range(self.n_iter + 1):
            optimizer.zero_grad()
            output = self.model(self.X_train)
            loss = -mll(output, self.y_train)
            loss.backward()
            optimizer.step()
            if i % 25 == 0:
                print("Iteration %d" % i)
                print("LS ", self.model.covar_module.base_kernel.lengthscale.item())
                print("Noise ", self.model.likelihood.noise.item())
                print()
        print("Hyperparameter estimation complete.")

        self.model.eval()
        self.likelihood.eval()

        # 1. Select multiple initial observations using farthest point sampling, add to active set and remove from the remainder set
        print(
            "Desired size of simplified cloud (based on simplification ratio): ",
            self.target_num_points,
        )
        remainder_set_idx = torch.tensor(
            [x for x in range(self.X["idx"].shape[0])], device=self.device
        )
        # tcp_idx = torch.topk(self.y, self.initial_set_size/3)[1]
        # fps_idx = torch.squeeze(
        #     farthest_point_sampler(
        #         torch.unsqueeze(self.X["coords"], 0), self.initial_set_size
        #     ),
        #     0,
        # )
        # active_set_idx = torch.cat((tcp_idx,fps_idx))
        active_set_idx = torch.squeeze(
            farthest_point_sampler(
                torch.unsqueeze(self.X["coords"], 0), self.initial_set_size
            ),
            0,
        )
        remainder_set_idx = remainder_set_idx[
            [i for i in remainder_set_idx.tolist() if i not in active_set_idx.tolist()]
        ]

        # 2. Iterate over (number of points we wish to use to represent our point cloud)/10
        simp_loop_start = time.time()
        active_set_size = self.initial_set_size
        i = 1
        while active_set_size < self.target_num_points:
            print("No. of points added to simplified cloud:", active_set_size)

            # 3. Update posterior (see Eq. 3 and 4 in above paper)
            # TODO - make this quicker using updates in paper appendix (think about dense format too, avoid if possible).
            #        Another option for speed-ups later (on bigger data) is to replace the exact GP with a SVGP.
            X_i = self.X["idx"][active_set_idx]
            X_r = self.X["idx"][remainder_set_idx]
            y_i = self.y[active_set_idx]
            y_r = self.y[remainder_set_idx]
            K_ri = self.model.covar_module(X_r, X_i).to_dense()
            K_rr = self.model.covar_module(X_r, X_r).to_dense()
            K_ii = self.model.covar_module(X_i, X_i).to_dense()
            K_ii_plus_noise = (
                K_ii
                + torch.eye(K_ii.shape[0], device=self.device) * self.likelihood.noise
            )
            K_ii_plus_noise_inv = torch.cholesky_inverse(K_ii_plus_noise)
            mu_t = K_ri.matmul(K_ii_plus_noise_inv).matmul(y_i)
            sigma_t = K_rr - K_ri.matmul(K_ii_plus_noise_inv).matmul(K_ri.T)

            # 4. Compute selection metric and select next 10 observations (you can also increase this number independent of initial_set_size)
            # TODO - two approaches for same thing: one inside loop one outside, FIX IT!
            selection_metric = torch.sqrt(torch.diag(sigma_t)) + torch.abs(mu_t - y_r)
            set_size = self.initial_set_size + int(self.initial_set_size / 5) * i
            if set_size + active_set_size >= self.target_num_points:
                set_size = self.target_num_points - active_set_size
            idx_to_remove_from_remainder_set = torch.topk(selection_metric, set_size)[1]
            active_set_idx = torch.cat(
                (
                    active_set_idx,
                    remainder_set_idx[idx_to_remove_from_remainder_set.tolist()],
                )
            )
            remainder_set_idx = remainder_set_idx[
                np.delete(
                    np.arange(len(remainder_set_idx)),
                    idx_to_remove_from_remainder_set.tolist(),
                )
            ]
            active_set_size = active_set_idx.shape[0]
            i = +1

        simp_loop_time = time.time() - simp_loop_start

        simp_coords = np.stack(
            (
                self.X["coords"][active_set_idx, 0].cpu().numpy(),
                self.X["coords"][active_set_idx, 1].cpu().numpy(),
                self.X["coords"][active_set_idx, 2].cpu().numpy(),
            ),
            axis=1,
        )

        return simp_coords, simp_loop_time
