import gpytorch


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
