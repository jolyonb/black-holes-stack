"""
linearevolve.py

Contains the Linear class, which performs linear evolution
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.fftpack import dst
from scipy.special import spherical_jn, spherical_yn

class Linear(object):
    """Performs linear evolution"""

    def __init__(self, model, phi, rvals):
        r"""
        Computes and stores \partial_\xi \delta_\rho from a given phi field,
        which should include the 1/N \sigma_0^2 normalization. Also computes
        \delta_U along the way.

        :param phi: numpy array of \tilde{\Phi}(\bar{r}) values
        :param rvals: numpy array of \bar{r} values (expected to be 0, delta/2, 3 delta/2, 5 delta/2, etc
        """
        # Store the field information
        self.model = model
        self.fullphi = phi
        self.fullrvals = rvals

        # Drop the first entry of phi and rvals, which we don't use
        self.phi = phi[1:]
        self.rvals = rvals[1:]

        # Compute the relevant derivatives
        self._compute_first_deriv()
        self._compute_second_deriv()

        # Compute \delta_U
        factor = self.model.beta * np.exp(- 2 * self.model.n_ef)
        phipower = self.phi ** (2 * self.model.beta - 2)
        self.delta_U = factor * phipower * (self.model.beta / 2 * self.phiprime**2
                                            - self.phi / self.rvals * self.phiprime)

        # Compute \partial_\xi \delta_\rho
        part1 = -2 * self.delta_U
        part2 = -2/3*np.exp(-self.model.n_ef)*(self.phi**(self.model.beta + 1)
                                               / (self.phi - self.model.beta * self.rvals * self.phiprime))
        dudr = self.model.beta * self.phiprime * self.phiprimeprime
        dudr -= self.phi * self.phiprimeprime / self.rvals
        dudr -= self.phiprime**2 / self.rvals
        dudr += self.phi * self.phiprimeprime / self.rvals**2
        dudr *= np.exp(-2*self.model.n_ef) * self.model.beta * self.phi**(2*self.model.beta - 2)
        dudr += (2*self.model.beta - 2) * self.phiprime / self.phi * self.delta_U
        self.delta_rho_dot = part1 + part2 * dudr

    def _compute_first_deriv(self):
        r"""
        Computes the first derivative of phi(r).
        Assumes evenly spaced gridpoints, first gridpoint at h/2, and even symmetry.
        Uses centered differences: (y(x + h) - y(x - h)) / h
        Right endpoint uses a one-sided derivative.

        Saves to self.phiprime
        """
        h = self.rvals[1] - self.rvals[0]
        # Compute the middle derivatives
        result = (self.phi[2:] - self.phi[:-2]) / (2*h)
        # Compute the first and last derivatives
        first = (self.phi[1] - self.phi[0]) / (2*h)
        last = (self.phi[-3] - 4*self.phi[-2] + 3*self.phi[-1]) / (2*h)
        # Store the result
        self.phiprime = np.concatenate((np.array([first]), result, np.array([last])))

    def _compute_second_deriv(self):
        r"""
        Computes the second derivative of phi(r).
        Assumes evenly spaced gridpoints, first gridpoint at h/2, and even symmetry.
        Uses second order centered difference: (y(x - h) - 2 * y(x) + y(x + h)) / h^2
        Left endpoint uses symmetry by fitting a parabola to four points (two across the origin).
        Right endpoint set to zero.

        Stores result in self.phiprimeprime
        """
        h = self.rvals[1] - self.rvals[0]
        # Compute the middle derivatives
        result = (self.phi[2:] + self.phi[:-2] - 2*self.phi[1:-1]) / h**2
        # Compute the first and last derivatives
        first = (self.phi[1] - self.phi[0]) / h**2
        last = 0
        # Return the result
        self.phiprimeprime = np.concatenate((np.array([first]), result, np.array([last])))

    def construct_grid(self, gridpoints=None):
        """
        Construct the grid to perform the linear evolution on

        :param gridpoints: Number of gridpoints to use. If omitted, uses the same number as in rvals
        """
        if gridpoints is None:
            gridpoints = len(self.rvals)
        self.gridpoints = gridpoints

        # Compute the barA points at each bar_r point we were given
        barA_vals = self.rvals * np.exp(self.model.n_ef) * self.phi**(-self.model.beta)

        # Estimate an outer boundary. This is the point where we impose boundary conditions.
        # If rvals goes up to (2n+1)/h, the outer boundary in bar_r should be at (2n+2)/h.
        # We can only do this approximately in A however.
        self.max_A = barA_vals[-1] + (barA_vals[-1] - barA_vals[-2])/2

        # Construct the set of wavenumbers
        self.wavenumbers = np.array(list(range(1, gridpoints + 1))) * np.pi / self.max_A

        # Set up the interpolator for delta_rho_dot as a function of barA
        # Add the outer boundary to barA_vals and delta_rho_dot to construct the interpolation
        interp = interp1d(np.append(barA_vals, self.max_A),
                          np.append(self.delta_rho_dot, 0),
                          kind="cubic",
                          fill_value="extrapolate")

        # Construct the grid between 0 and 1
        self.grid = (np.array(list(range(gridpoints))) + 0.5) / gridpoints
        # Construct the grid between 0 and max_A
        self.Agrid = self.grid * self.max_A

        # Construct the various spatial modes
        # Note that these modes are matrices. First index is the A position, second index is the wavenumber.
        # So, we can matrix multiply this by a vector of kn time values to get the desired evolutions.
        kn_times_A = np.outer(self.Agrid, self.wavenumbers)
        self.j0modes = spherical_jn(0, kn_times_A)
        self.j0modes_times_k = self.j0modes.copy() * self.wavenumbers
        self.j1modes = spherical_jn(1, kn_times_A)
        self.j1modes_on_ka = self.j1modes / kn_times_A

        # Construct the grid of rho values
        self.deltarho = np.zeros_like(self.grid)
        # Construct the grid of \partial_\xi \delta rho values on the Agrid
        self.rhodot = interp(self.Agrid)

    def construct_spectral(self):
        """Constructs the spectral representations of deltarho and rhodot"""
        # Construct the initial spectral representation of deltarho and rhodot
        self.bn = self.space2spectral(self.deltarho)
        self.cn = self.space2spectral(self.rhodot)

        # Construct the Bn and Cn coefficients for deltarho
        wavenums = self.wavenumbers / np.sqrt(3)
        self.Bn = - self.bn * np.cos(wavenums) - (self.bn + 2 * self.cn) * spherical_yn(1, wavenums)
        self.Cn = - self.bn * np.sin(wavenums) + (self.bn + 2 * self.cn) * spherical_jn(1, wavenums)

    def space2spectral(self, spatial):
        """
        Transforms spatial values to their spherical bessel coefficients

        Note that space2spectral and spectral2space are inverse operations,
        up to numerical precision.
        """
        # Sort out the normalization conventions...
        fact = 1.0 * np.array(list(range(1, self.gridpoints + 1)))
        fact[-1] /= 2
        coeff = np.pi / self.gridpoints / self.max_A

        spectral = dst(spatial * self.Agrid, type=2) * coeff * fact
        return spectral

    def spectral2space(self, spectral):
        """
        Transforms spherical bessel coefficients to their spatial values
        """
        # Sort out the normalization conventions...
        fact = 1.0 * np.array(list(range(1, self.gridpoints + 1)))
        fact[-1] /= 2
        coeff = np.pi / self.gridpoints / self.max_A

        spatial = dst(spectral / fact, type=3) / self.Agrid / coeff / 2 / self.gridpoints
        return spatial

    def rho_at_origin(self, xi):
        r"""
        Given a time xi, compute the value of \delta \rho at the origin
        """
        t = np.exp(xi / 2) / np.sqrt(3)
        value = self.Bn * spherical_jn(1, self.wavenumbers * t) + self.Cn * spherical_yn(1, self.wavenumbers * t)
        value *= self.wavenumbers
        return t * np.sum(value)

    def full_evolution(self, xi):
        r"""
        Given a time xi, compute the value of \tilde{\rho}, \tilde{m}, \tilde{u}, \tilde{r}
        """
        t = np.exp(xi / 2) / np.sqrt(3)
        j0t = spherical_jn(0, self.wavenumbers * t)
        j1t = spherical_jn(1, self.wavenumbers * t)
        y0t = spherical_yn(0, self.wavenumbers * t)
        y1t = spherical_yn(1, self.wavenumbers * t)
        sint = np.sin(self.wavenumbers * t)
        cost = np.cos(self.wavenumbers * t)

        j03 = spherical_jn(0, self.wavenumbers / np.sqrt(3))
        j13 = spherical_jn(1, self.wavenumbers / np.sqrt(3))
        y03 = spherical_yn(0, self.wavenumbers / np.sqrt(3))
        y13 = spherical_yn(1, self.wavenumbers / np.sqrt(3))

        timeval1 = self.Bn * j1t + self.Cn * y1t
        timeval2 = self.Bn * (sint - 2 * j1t) - self.Cn * (cost + 2 * y1t)

        deltarho = t * np.dot(self.j0modes_times_k, timeval1)
        deltam = 3 * t / self.Agrid * np.dot(self.j1modes, timeval1)
        deltaU = - 3 / 4 * t / self.Agrid * np.dot(self.j1modes, timeval2)

        deltaRB1 = np.dot(self.j0modes, self.Bn * (j0t - j03))
        deltaRB2 = - 3 * np.dot(self.j1modes_on_ka, self.Bn * (j0t + self.wavenumbers * j1t * t -
                                                               (j03 + self.wavenumbers * j13 / np.sqrt(3))))
        deltaRC1 = np.dot(self.j0modes, self.Cn * (y0t - y03))
        deltaRC2 = - 3 * np.dot(self.j1modes_on_ka, self.Cn * (y0t + self.wavenumbers * y1t * t -
                                                               (y03 + self.wavenumbers * y13 / np.sqrt(3))))

        deltaR = 1/16*(deltaRB1 + deltaRB2 + deltaRC1 + deltaRC2)

        tilderho = 1 + deltarho
        tildem = 1 + deltam
        tildeR = (1 + deltaR) * self.Agrid
        tildeU = (1 + deltaU) * tildeR

        return tildeR, deltaU, tildem, deltarho
        return tildeR, tildeU, tildem, tilderho



# To do:
# Data output (including scaling)
# Checking for collapse
# Comparison between linear and nonlinear
# Importing of initial conditions to nonlinear simulation
# DONUTS ARE GONE!
# Is our estimate for reflection off??? (No, needs to use inflated radius)
# Make sure that the \delta_U we originally calculate is really close
# to the \delta_U that we compute from the spectral method
# (perhaps we can get away without a second derivative???)
# Put in a test to make sure that the new time coordinate is timelike!
