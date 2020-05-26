"""
levin.py

Contains the Levin class, a general Levin integrator, as well as subclasses that implement specific integrals.

For details on the Levin method, see this writuep:
https://reference.wolfram.com/language/tutorial/NIntegrateIntegrationRules.html#32844337
"""
from typing import Tuple, Callable

import numpy as np
from scipy.special import spherical_jn


class Levin(object):
    """
    General Levin integration class
    
    To use:
    * Figure out amplitude function, kernel vector and differential matrix
    * Instantiate class
    * Set limits
    * Set amplitude
    * Set kernel
    * Set differential matrix
    * Call integrate
    
    The kernel and differential matrix will likely need to be constructed for each individual
    integral, but the limits and amplitude can probably be evaluated only a single time for a given
    class of integrals.
    
    The integrate method returns the integral and the error estimate (difference from previous refinement, most likely
    an overestimate).
    
    Note that no attempt is made to switch to a different method when the integrand is not oscillatory over
    the domain of the integral. This should be handled separately.
    """
    
    def __init__(self, starting_points: int = 20, refinements: int = 6,
                 abs_tol: float = 0, rel_tol: float = 1e-7) -> None:
        """Initialize the Levin integrator by precomputing grids"""
        # Store parameters
        self.starting_points = starting_points
        self.refinements = refinements
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        
        # Construct the different grid sizes
        assert refinements > 0
        self.grid_sizes = [starting_points]
        for _ in range(refinements):
            self.grid_sizes.append(self.grid_sizes[-1] * 2)

        # Construct collocation angles for the largest grid
        numpoints = self.grid_sizes[-1]
        theta = np.pi * np.linspace(0.0, 1.0, numpoints + 1)  # We use one extra point at 0
        # Collocation points in x space
        self.colloc = -np.cos(theta)

        # Precompute the Chebyshev polynomials on the largest grid
        # These are (n, collocation) indexed
        self.chebyTs = np.array([self._chebyT(theta, n) for n in range(numpoints + 1)])
        self.dchebyTs = np.array([n * self._chebyU(theta, n - 1) for n in range(0, numpoints + 1)])
        
        # Set up placeholders
        self.a = None
        self.b = None
        self.kvals = None
        self.kernel_a = None
        self.kernel_b = None
        self.amplitude = None
        self.dm = None

    @staticmethod
    def _chebyT(theta: np.array, n: int) -> np.array:
        """
        Returns Chebyshev polynomials of the first kind of order n at value -theta.
        Vectorized over theta.
        """
        return (-1) ** (n % 2) * np.cos(n * theta)

    @staticmethod
    def _chebyU(theta: np.array, n: int) -> np.array:
        """
        Returns Chebyshev polynomials of the second kind of order n at value -theta.
        Vectorized over theta.
        """
        if n == -1:
            return np.zeros_like(theta)
        subtheta = theta[1:-1]
        n1 = n + 1
        factor = (-1) ** (n % 2)
        return np.concatenate(([factor * n1], factor * np.sin(n1 * subtheta) / np.sin(subtheta), [n1]))

    def set_limits(self, a: float, b: float) -> None:
        """
        Sets limits of integration and construct collocation grid in k-space.
        x space ranges from -1 to 1, while k-space ranges from a to b.
        """
        self.a = a
        self.b = b
        self.kvals = (b - a) / 2 * self.colloc + (b + a) / 2

    def set_amplitude(self, amplitude: Callable) -> None:
        """
        Set the amplitude function.
        Evaluates the amplitude on all collocation gridpoints.
        """
        assert self.a is not None
        self.amplitude = np.array([amplitude(k) for k in self.kvals])

    def set_kernel(self, kernel: Callable) -> None:
        """
        Set the integration kernel.
        Evaluates the kernel at the limits of integration.
        """
        assert self.a is not None
        self.kernel_a = kernel(self.kvals[0])
        self.kernel_b = kernel(self.kvals[-1])

    def set_differential_matrix(self, differential_matrix: Callable) -> None:
        """
        Set the differential matrix.
        Evaluates the differential matrix on all collocation gridpoints.
        """
        assert self.a is not None
        assert self.kernel_a is not None
        # This is indexed by collocation, weight1, weight2, where weight1 and weight2 are set
        # so that kernel' = dm[idx].kernel.
        self.dm = np.array([differential_matrix(k) for k in self.kvals])
        # Check that the matrix has the right dimensions
        shape = self.dm[0].shape
        assert len(self.kernel_a) == shape[0] == shape[1]

    def _slice(self, numpoints: int) -> slice:
        """
        Constructs the slice that will extract the appropriate gridpoints from everything.

        :param numpoints: Number of collocation points to use
        :return: A slice object
        """
        # Find the refinement number for the given number of points
        try:
            refinement = self.grid_sizes.index(numpoints)
        except ValueError:
            raise ValueError('Number of collocation points was not initialized')

        # Compute the stepsize and construct the slice
        stepsize = 2 ** (self.refinements - refinement)
        sl = np.s_[0::stepsize]
        assert len(self.colloc[sl]) == numpoints + 1

        return sl
        
    def _integrate(self, numpoints: int) -> float:
        """
        Integrates functions of the form
        int_a^b f(k) w(k) dk
        using the Levin-collocation method, using the given number of points.
        
        We use a kernel vector w_i(k), with w(k) = w_1(k).
        The differential matrix dm is such that dw/dk = dm.w for the vector w.
        
        The kernel vector only needs to be evaluated at the limits a and b.
        The differential matrix needs to be computed at each collocation point.
        
        :param numpoints: Number of collocation points to use
        :return: Value of the integral
        """
        # Make sure everything is ready to go!
        assert self.a is not None
        assert self.b is not None
        assert self.kvals is not None
        assert self.kernel_a is not None
        assert self.kernel_b is not None
        assert self.amplitude is not None
        assert self.dm is not None

        # Construct the slice operator to extract values on the appropriate collocation grid
        sl = self._slice(numpoints)
        numpoints += 1  # Add the extra point for 0

        # Extract everything on the appropriate collocation points
        kernel_a = self.kernel_a
        kernel_b = self.kernel_b
        amplitude = self.amplitude[sl]
        dm = self.dm[sl]
        chebyTs = self.chebyTs[:numpoints, sl]  # Take first numpoints ell values, then slice for collocation points
        derivs = 2 / (self.b - self.a) * self.dchebyTs[:numpoints, sl]

        # Construct the amplitude matrix (including zero weights for unused kernels)
        kernel_len = len(self.kernel_a)
        # This is indexed by kernel index, collocation point
        amplitudes = np.vstack([amplitude] + [np.zeros_like(amplitude)] * (kernel_len - 1))
        
        # The collocation method now solves the following equation at the collocation points:
        # sum_i^kernel_len A_{i,j}(x) (sum_k^num_points u_k(x) c_{i, k}) + sum_k^num_points c_{j, k} u_k'(x) = f_j(x)
        # A is the differential matrix
        # f_j is the amplitude for weight j
        # c_{i, k} are the unknown coefficients to solve for
        # We use u_k(x) = Chebyshev polynomials and evaluate this equation at the collocation points
        # Number of collocation points = Number of Chebyshev polynomials used

        # Now construct the matrix equations a_{ijkl} c_{ik} = f_{jl}  (l is the collocation index)
        # c_{ik} is unknown, and needs to be solved for
        # f_{jl} is the amplitude matrix constructed above
        # a_{ijkl} needs to be constructed, and has indices of weight1, weight2, Chebyshev order, collocation index
        a = np.zeros([kernel_len, kernel_len, numpoints, numpoints])
        # Loop over collocation points
        for l in range(numpoints):
            # These tensordots are constructed as a tensor product (no summation)
            a[:, :, :, l] = np.tensordot(dm[l], chebyTs[:, l], 0) + np.tensordot(np.eye(kernel_len), derivs[:, l], 0)
        
        # Solve the system, moving indices i and k to the right to contract with c_{ik}
        cij = np.linalg.tensorsolve(a, amplitudes, axes=(0, 2))
        
        # Construct the integral. The antiderivative is
        # F_i(x) = sum_k c_{ik} T_k(x)
        # which we evaluate at the limits
        resulta = np.dot(np.dot(cij, chebyTs[:, 0]), kernel_a)
        resultb = np.dot(np.dot(cij, chebyTs[:, -1]), kernel_b)
        result = float(resultb - resulta)
        
        return result
    
    def integrate(self) -> Tuple[float, float]:
        """
        Performs the Levin integration. Starts out by using the minimum number of points specified, then refines
        repeatedly until the desired tolerances are met.

        :return: Value of the integral, error estimate
        """
        base = self._integrate(self.starting_points)

        for gridsize in self.grid_sizes[1:]:
            # Compute the integral on the next grid
            result = self._integrate(gridsize)
            
            # Check to see if the result is within the appropriate tolerance
            tolerance = self.abs_tol + result * self.rel_tol
            rel_err = abs(result - base) / result
            if abs(result - base) < tolerance:
                return result, rel_err
            
            # Make the result the next base, and try again
            base = result
        else:
            # If we got here, we didn't converge
            print('Warning: Did not converge to the desired tolerance within desired number of refinements')
            return base, rel_err


class LevinIntegrals(Levin):
    """
    Integration class for various spherical bessel integrals.
    
    Instantiate using the __init__ parameters of Levin.
    
    Before calling an integrate_X routine, set the limits and amplitude function using

    # Set the limits of integration
    LevinIntegrals.set_limits(a, b)
    # Set the amplitude function
    LevinIntegrals.set_amplitude(amplitude)
    """
    
    def integrate_I(self, ell: int, alpha: float) -> Tuple[float, float]:
        """
        Performs integrals of the form
        I = int_a^b f(k) j_l(alpha k) dk
        using the Levin-collocation method.

        The kernel vector is {j_ell(alpha*k), j_(ell+1)(alpha*k)}.
        The differential matrix is {{ell/k, -alpha}, {alpha, -(2+ell)/k}}.
        
        Note that the differential matrix becomes singular as k -> 0, so we can only treat positive limits.
        
        :param ell: Order of spherical bessel function to integrate
        :param alpha: Coefficient in the spherical bessel function
        :return: Value of the integral, error estimate
        """
        # Set the integration kernel
        def kernel(k):
            return np.array([spherical_jn(ell, alpha * k), spherical_jn(ell + 1, alpha * k)])
        self.set_kernel(kernel)

        # Set the differential matrix
        dm_diag = np.diag([ell, -(2 + ell)])
        dm_off = np.zeros((2, 2))
        dm_off[0, 1] = -alpha
        dm_off[1, 0] = alpha

        def dm(k):
            return dm_diag / k + dm_off
        self.set_differential_matrix(dm)

        # Perform the integration
        return self.integrate()
