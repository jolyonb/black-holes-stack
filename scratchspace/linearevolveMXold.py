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

    def __init__(self, model, phi, rvals, rhomethod=False):
        """
        Computes and stores \partial_xi \delta_rho from a given phi field,
        which should include the 1/N \sigma_0^2 normalization. Also computes
        \delta_U along the way.

        :param phi: numpy array of tilde{\Phi}(bar{r}) values
        :param rvals: numpy array of bar{r} values (expected to be 0, delta/2, 3 delta/2, 5 delta/2, etc
	:param rhomethod: True if using \partial_xi \delta_rho to construct spectral coefficients,
	and False if using \delta U
        """
        # Store the field information
        self.model = model
        self.fullphi = phi
        self.fullrvals = rvals
        self.rhomethod = rhomethod

        # Drop the first entry of phi and rvals, which we don't use
        self.phi = phi[1:]
        self.rvals = rvals[1:]

        # Compute the relevant derivatives
        self._compute_first_deriv()
        self._compute_second_deriv()
        '''
        plt.plot(self.rvals, self.phi)
        plt.xlabel('r')
        plt.ylabel('Phi(r)')
        plt.title('Input function')
        plt.show()'''

        # Compute \delta_U
        factor = self.model.beta * np.exp(- 2 * self.model.n_ef)
        phipower = self.phi ** (2 * self.model.beta - 2)
        self.delta_U = factor * phipower * (self.model.beta / 2 * self.phiprime**2
                                            - self.phi / self.rvals * self.phiprime)

        # Compute \partial_\xi \delta_\rho, if necessary
        if self.rhomethod==True:
                self.compute_delta_rho_dot()
                plt.plot(self.rvals, self.delta_rho_dot)
                plt.xlabel('r')
                plt.ylabel('delta_rho_dot')
                plt.title('Delta_rho_dot(r)')
                plt.show()

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

    def compute_delta_rho_dot(self):
        """Computes \partial_xi \delta_rho values"""
        part1 = -2 * self.delta_U
        part2 = -2/3*np.exp(-self.model.n_ef)*(self.rvals)*(self.phi
                                               / (self.phi - self.model.beta * self.rvals * self.phiprime))
	# Compute \partial_r(\delta_U)
        dudr = self.model.beta * self.phiprime * self.phiprimeprime
        dudr -= self.phi * self.phiprimeprime / self.rvals
        dudr -= self.phiprime**2 / self.rvals
        dudr += self.phi * self.phiprime / self.rvals**2
        dudr *= np.exp(-2*self.model.n_ef) * self.model.beta * self.phi**(2*self.model.beta - 2)
        dudr += (2*self.model.beta - 2) * self.phiprime / self.phi * self.delta_U
	# Return result
        self.delta_rho_dot = part1 + part2 * dudr

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
        '''
        plt.plot(self.rvals, barA_vals)
        plt.xlabel('r')
        plt.ylabel('A')
        plt.title('r vs A values')
        plt.show()

        plt.plot(self.rvals[0:20], barA_vals[0:20])
        plt.xlabel('r')
        plt.ylabel('A')
        plt.title('r vs A values (zoomed in)')
        plt.show()

        plt.plot(self.rvals, np.exp(self.model.n_ef) * self.phi**(-self.model.beta))
        plt.xlabel('r')
        plt.ylabel('A/r')
        plt.title('A/r values (e^N * Phi^(-beta))')
        plt.show()
        '''
        # Estimate an outer boundary. This is the point where we impose boundary conditions.
        # If rvals goes up to (2n+1)/h, the outer boundary in bar_r should be at (2n+2)/h.
        # We can only do this approximately in A however.
        self.max_A = barA_vals[-1] + (barA_vals[-1] - barA_vals[-2])/2

        # Construct the set of wavenumbers
        self.wavenumbers = np.array(list(range(1, gridpoints + 1))) * np.pi / self.max_A

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

	# Construct \partial_\xi \delta_rho or \delta_U values (as necessary) on the Agrid
        if self.rhomethod==False:
        	# Set up the interpolator for delta_U as a function of barA
        	# Add the outer boundary to barA_vals and delta_U to construct the interpolation
        	self.interp = interp1d(np.append(barA_vals, self.max_A),
                          np.append(self.delta_U, 0), 
                          kind="cubic",
                          fill_value="extrapolate")
        	# Construct the grid of \delta_U values on the Agrid
        	self.delU = self.interp(self.Agrid)

    def construct_spectral_from_rho(self):
        """Constructs the spectral representations of deltarho and rhodot"""
        # Construct the initial spectral representation of deltarho and rhodot
        self.bn = self.space2spectral(self.deltarho)
        self.cn = self.space2spectral(self.rhodot)

        # Construct the Bn and Cn coefficients 
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

    def construct_spectral_from_u(self):
        """Constructs the spectral representations of delta_rho and delta_U"""
	#FIXME (check everything worked out)
	# Construct an array of \delta_rho and \delta_U spatial values
	# First position tells you if \delta_rho or \delta_U, second position determines for what A
        deltas = np.array([self.deltarho, self.delU]) # dim=(2, dim(Agrid))

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

    def construct_spectral_from_u1(self):
        """Constructs the spectral representations of delta_rho and delta_U"""
	#FIXME (check everything worked out)

	# Construct spectral representation of \delta_rho
        self.bn = self.space2spectral(self.deltarho)

	# Let an= 2cn - bn
	# Construct spatial modes for \delta_U
        an_modes =  - 3. / 4. * self.j1modes_on_ka

        # Linearly solve the matrix equation for the an coefficients 
        self.an = np.linalg.solve(an_modes, self.delU) # dim=(1,n)
        self.cn = (self.an + self. bn) / 2

        # Construct Bn and Cn coefficients
        t0 = 1 / np.sqrt(3)
        j1t0 = spherical_jn(1, self.wavenumbers * t0) 
        y1t0 = spherical_yn(1, self.wavenumbers * t0)
        self.Bn = - self.bn * np.cos(self.wavenumbers * t0) - (self.bn + 2 * self.cn) * y1t0
        self.Cn = - self.bn * np.sin(self.wavenumbers * t0) + (self.bn + 2 * self.cn) * j1t0

    def rho_at_origin(self, xi):
        r"""
        Given a time value xi, compute the value of \delta \rho at the origin
        """
        t = np.exp(xi / 2) / np.sqrt(3)
        y1t = spherical_yn(1, self.wavenumbers * t)
        j1t = spherical_jn(1, self.wavenumbers * t)
        #print(j1t)
        timeval1 = self.Bn * j1t + self.Cn * y1t
        #print("timeval1 is " + str(timeval1))
        deltarho = t * np.dot(self.wavenumbers, timeval1)
        return deltarho

        #value = self.Bn * spherical_jn(1, self.wavenumbers * t) + self.Cn * spherical_yn(1, self.wavenumbers * t)
        #value *= self.wavenumbers
        #return t * np.sum(value)

    def full_evolution(self, xi):
        """
        Given a time xi, compute the value of tilde{rho}, tilde{m}, tilde{u}, tilde{r}
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

        if xi==0:
                fig, (ax1, ax2) = plt.subplots(2)
                fig.suptitle('Testing delta_U construction')
                ax1.plot(self.Agrid, self.delU, label='Interpolated delta_U')
                ax1.plot(self.Agrid, deltaU, label='Reconstructed delta_U')
                ax1.set(ylabel='delta_U')
                ax2.plot(self.Agrid, np.absolute(deltaU - self.delU))
                ax2.set(xlabel='bar(A)', ylabel='Difference')
                ax1.legend()
                plt.show()

        '''
        plt.plot(self.Agrid, self.delta_U, label='OG delta_U') 
        plt.plot(self.Agrid, self.delU, label='Interpl delta_U') 
        plt.plot(self.Agrid, deltaU, label='reconstructed delta_U')
        plt.legend()
        plt.show()
        '''

        tilderho = 1 + deltarho
        tildem = 1 + deltam
        tildeR = (1 + deltaR) * self.Agrid
        tildeU = (1 + deltaU) * tildeR

        return tildeR, deltaU, tildem, deltarho
        #return tildeR, tildeU, tildem, tilderho



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
# Added option to compute spectral coeffs directly from \delta_U
# which avoids the second derivative!
