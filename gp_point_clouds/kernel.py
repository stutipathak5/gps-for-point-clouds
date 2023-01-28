import lab as B
import numpy as np
from geometric_kernels.kernels import MaternKarhunenLoeveKernel
from geometric_kernels.lab_extras import from_numpy


class MaternKarhunenLoeveKernelDeviceAgnostic(MaternKarhunenLoeveKernel):
    def __init__(self, space, num_eigenfunctions, device):
        super().__init__(space, num_eigenfunctions)
        self.device = device

    def _spectrum(
        self, s: B.Numeric, nu: B.Numeric, lengthscale: B.Numeric
    ) -> B.Numeric:
        """
        NOTE - modified to add cast of base to device; wasn't previously
               working in a device agnostic fashion across CPU/GPU.

        Matern or RBF spectrum evaluated at `s`.
        Depends on the `lengthscale` parameters.
        """
        if nu == np.inf:
            return B.exp(-(lengthscale**2) / 2.0 * from_numpy(lengthscale, s**2))
        elif nu > 0:
            power = -nu - self.space.dimension / 2.0
            base = 2.0 * nu / lengthscale**2 + B.cast(
                B.dtype(nu), from_numpy(nu, s**2)
            ).to(self.device)
            return base**power
        else:
            raise NotImplementedError
