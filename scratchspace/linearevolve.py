"""
linearevolve.py

Contains the Linear class, which performs linear evolution
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.fftpack import dst
from scipy.special import spherical_jn, spherical_yn
import matplotlib.pyplot as plt

class Linear(object):
    """Performs linear evolution"""

    def __init__(self, model, phi, rvals):
        """
        Computes and stores \partial_xi \delta_rho from a given phi field,
        which should include the 1/N \sigma_0^2 normalization. Also computes
        \delta_U0 along the way.

        :param phi: numpy array of tilde{\Phi}(bar{r}) values
        :param rvals: numpy array of bar{r} values (expected to be 0, delta/2, 3 delta/2, 5 delta/2, etc
        """
        # Store the field information
        self.model = model
        self.fullphi = phi
        self.fullrvals = rvals

        # Drop the first entry of phi and rvals, which we don't use
        self.phi = phi[1:]
        self.rvals = rvals[1:]

        # Compute the first derivative
        self._compute_first_deriv()

        # Compute \delta_U
        factor = self.model.beta * np.exp(- 2 * self.model.n_ef)
        phipower = self.phi ** (2 * self.model.beta - 2)
        self.delta_U = factor * phipower * (self.model.beta / 2 * self.phiprime**2
                                            - self.phi / self.rvals * self.phiprime)

    def _compute_first_deriv(self):
        """
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

    def construct_grid(self):
        """
        Construct the grid to perform the linear evolution on
        Saves as self.Agrid

        Also constructs the array of delta_rho values
        Saves as self.deltarho

        """

        # Compute the barA gridpoints at each bar_r point we were given
        # We will perform the linear evolution on this grid
        self.Agrid = self.rvals * np.exp(self.model.n_ef) * self.phi**(-self.model.beta)

        # Estimate an outer boundary. This is the point where we impose boundary conditions.
        # If rvals goes up to (2n+1)/h, the outer boundary in bar_r should be at (2n+2)/h.
        # We can only do this approximately in A however.
        self.max_A = self.Agrid[-1] + (self.Agrid[-1] - self.Agrid[-2])/2

        # Construct the set of wavenumbers
        self.wavenumbers = np.array(list(range(1, len(self.rvals) + 1))) * np.pi / self.max_A

        # Construct the various spatial modes
        # Note that these modes are matrices. First index is the A position, second index is the wavenumber.
        # So, we can matrix multiply this by a vector of kn time values to get the desired evolutions.
        kn_times_A = np.outer(self.Agrid, self.wavenumbers)
        self.j0modes = spherical_jn(0, kn_times_A)
        self.j0modes_times_k = self.j0modes.copy() * self.wavenumbers
        self.j1modes = spherical_jn(1, kn_times_A)
        self.j1modes_on_ka = self.j1modes / kn_times_A

        # Construct the grid of rho values
        self.deltarho = np.zeros_like(self.Agrid)

    def construct_spectral(self):
        """
        Constructs the spectral coefficients from delta_rho and delta_U
        """
	# Construct an array of \delta_rho and \delta_U spatial values
	# First position tells you if \delta_rho or \delta_U, second position determines for what A
        deltas = np.array([self.deltarho, self.delta_U]) # dim=(2, dim(Agrid))

	# Construct spatial modes for each \delta_rho and \delta_U, each Bn and Cn
        t0 = 1 / np.sqrt(3)
        j1t0 = spherical_jn(1, self.wavenumbers * t0) # dim=n
        y1t0 = spherical_yn(1, self.wavenumbers * t0)
        sinj1t0 = np.sin(self.wavenumbers * t0) - 2. * j1t0
        cosy1t0 = - (np.cos(self.wavenumbers * t0) + 2. * y1t0)

        rho_Bn_modes = t0 * self.j0modes_times_k * j1t0 # dim=(dim(Agrid), n)
        rho_Cn_modes = t0 * self.j0modes_times_k * y1t0
        U_Bn_modes = - 3. / 4. * t0 * ((self.j1modes * sinj1t0).T / self.Agrid).T 
        U_Cn_modes = - 3. / 4. * t0 * ((self.j1modes * cosy1t0).T / self.Agrid).T 

	# Merge into a single array of spatial modes
        rho_modes = np.array([rho_Bn_modes, rho_Cn_modes]) # dim=(2, dim(Agrid), n)
        U_modes = np.array([U_Bn_modes, U_Cn_modes])
        premodes = np.array([rho_modes, U_modes]) # dim=(2 [rho/U], 2 [Bn/Cn], dim(Agrid), n)
        modes = premodes.transpose(0,2,1,3) # dim=(2 [rho/U], dim(Agrid), 2 [Bn/Cn], n)

        # Linearly solve the matrix equation for the Bn and Cn coefficients 
        coeffs = np.linalg.tensorsolve(modes, deltas) # dim=(2,n)
        self.Bn = coeffs[0,:]
        self.Cn = coeffs[1,:]

    def rho_at_origin(self, xi):
        """
        computes the value of \delta_rho at the origin for some time xi

        :param xi: a time value (not an array! this cannot handle an array!)
        """
        t = np.exp(xi / 2) / np.sqrt(3)
        y1t = spherical_yn(1, self.wavenumbers * t)
        j1t = spherical_jn(1, self.wavenumbers * t)
        timeval1 = self.Bn * j1t + self.Cn * y1t
        return t * np.dot(self.wavenumbers, timeval1)

    def full_evolution(self, xi):
        """
        Compute the values of tilde{rho}, tilde{m}, tilde{u}, tilde{r} for some time xi

        :param xi: a time value (not an array! this cannot handle an array!)
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

        deltaR = 1/4*(deltaRB1 + deltaRB2 + deltaRC1 + deltaRC2)

        tilderho = 1 + deltarho
        tildem = 1 + deltam
        tildeR = (1 + deltaR) * self.Agrid
        tildeU = (1 + deltaU) * tildeR

        #return tildeR, tildeU, tildem, tilderho
        return deltaU



# To do (Jolyon):
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

# Changes (Michelle):
# Corrected typo in construction of part 2 (Jolyon's write up is missing a factor of A = r*tilde(phi)^-beta)
# Corrected typo in construction of du/dr
# Corrected factor of 1/16 in deltaR to 1/4
# Changed construct_spectral to compute spectral coeffs directly from \delta_U
# which avoids the second derivative!
# Removed evenly spaced Agrid and gridpoints param
# Correspondingly removed interpolation of \delta_U0 
