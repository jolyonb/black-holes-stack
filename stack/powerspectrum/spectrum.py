"""
Power spectrum of waterfall field perturbations in hybrid inflation
v1.0 by Jolyon Bloomfield, April 2017
See arXiv:xxxx.xxxxx for details
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy import exp, log, pi, log10, expm1
from scipy.integrate import ode
from scipy.interpolate import InterpolatedUnivariateSpline

from typing import TYPE_CHECKING

from stack.common import Persistence, Suppression

if TYPE_CHECKING:
    from stack import Model

# Raise errors on issues rather than printing warnings
np.seterr(all='raise')


class PowerSpectrum(Persistence):
    """
    Computes the power spectrum for hybrid inflation theories
    """
    filename = 'powerspectrum'

    def __init__(self, model: 'Model') -> None:
        """
        Initialize the class.
        
        :param model: Model class we are computing the power spectrum for.
        """
        super().__init__(model)

        self.min_k = None
        self.max_k = None
        self.kvals = None
        self.spectrum = None
        self.interp = None

        # Error tolerances used in computing ODE solutions
        self.err_abs = 0
        self.err_rel = 1e-13
        
        # Storage for mode function integration
        self.df_rvals = None
        self.df_times = None

    def load_data(self) -> None:
        """Load the power spectrum from file"""
        if self.model.test_ps:
            if self.model.verbose:
                print('    Using dummy power spectrum')
            self.min_k = self.model.min_k
            self.max_k = self.model.max_k
            return
        
        filename = self.filename + '.csv'
        path = self.file_path(filename)
        if not self.file_exists(filename):
            raise FileNotFoundError(f'Unable to load from {path}')

        df = pd.read_csv(path)
        
        self.kvals = df.k.values
        self.min_k = self.kvals[0]
        self.max_k = self.kvals[-1]

        self.spectrum = df.spectrum.values

        self.construct_interpolant()

    def save_data(self) -> None:
        """Saves the power spectrum to file"""
        if self.model.test_ps:
            return

        # Save the power spectrum
        df = pd.DataFrame([self.kvals, self.spectrum]).transpose()
        df.columns = ['k', 'spectrum']
        df.to_csv(self.file_path(self.filename + '.csv'), index=False)

        # Also save the mode evolution (not loaded back in when loading data)
        self.df_rvals.to_csv(self.file_path(self.filename + '-Rvals.csv'), index=False)
        self.df_times.to_csv(self.file_path(self.filename + '-Nvals.csv'), index=False)

    def compute_data(self) -> None:
        """Compute the power spectrum for the model"""
        if self.model.test_ps:
            self.min_k = self.model.min_k
            self.max_k = self.model.max_k
            return
        self.construct_grid()
        self.construct_spectrum()
        self.construct_interpolant()

    def __call__(self, k: float, suppression: Suppression) -> float:
        """
        Return the value of the power spectrum at a given k value, given the desired form of high-frequency suppression.
        """
        if self.model.test_ps:
            if k < -700:
                return float('inf')
            if k > 700:
                return 0
            value = exp(-k)
        else:
            if 0.999 * self.min_k < k < self.min_k:
                k = self.min_k
            elif self.max_k < k < 1.001 * self.max_k:
                k = self.max_k
            value = self.interp(k)
        if suppression == Suppression.RAW:
            suppression_val = 1
        elif suppression == Suppression.SAMPLING:
            assert self.model.grid.ready
            suppression_val = exp(-(k / self.model.grid.sampling_cutoff)**2)
        elif suppression == Suppression.PEAKS:
            raise NotImplementedError('Method not ready yet')
        else:
            raise ValueError(f'Bad suppression value passed in: {suppression}')
        return value * suppression_val

    def construct_grid(self) -> None:
        """Construct the grid in k space to evaluate the power spectrum at"""
        self.min_k = self.model.min_k
        self.max_k = self.model.max_k
        self.kvals = np.logspace(start=log10(self.min_k),
                                 stop=log10(self.max_k),
                                 num=self.model.num_modes,
                                 endpoint=True)
        
    def construct_interpolant(self) -> None:
        """Construct an interpolant over the power spectrum"""
        # Note: k = 5 performs better on the interpolation than k = 3 (empirical testing), at around 10^-6 relative error
        # across the board for 401 points
        self.interp = InterpolatedUnivariateSpline(self.kvals, self.spectrum, k=5, ext='raise')

    def construct_spectrum(self) -> None:
        """Compute the power spectrum"""

        # Grab some variables
        muphi2 = self.model.muphi2
        mupsi2 = self.model.mupsi2
        endN = self.model.n_efolds
        kvals = self.kvals
        kvals2 = kvals**2
        num_k = len(kvals)

        # Figure out the time to start integrating from. Each mode has its own value of N.
        # N = 0 is the waterfall transition
        startN = log(8 * 10**(-15) * kvals**4) / 4.0
        minN = np.min(startN)

        # Compute initial condition corrections
        correction01 = exp(startN) / (2 * kvals2)
        correction21 = correction01 * (muphi2 / 2) * (1 - exp(-mupsi2 * startN))
        correction1 = correction01 + correction21

        correction03 = - exp(3 * startN) / (8 * kvals**4)
        correction23 = - 0.5 * correction03 * muphi2 * (4 + exp(-mupsi2 * startN) * (mupsi2**2 - 5 * mupsi2 - 4))
        correction43 = - 1.25 * correction03 * muphi2**2 * (1 - exp(-mupsi2 * startN))**2
        correction2 = correction03 + correction23 + correction43

        correction05 = exp(5 * startN) / (16 * kvals**6)
        correction25 = - 0.25 * muphi2 * correction05 * (86 + exp(-mupsi2 * startN) * (mupsi2**4 - 14 * mupsi2**3 + 53 * mupsi2**2 - 24 * mupsi2 - 86))
        correction45 = - 0.25 * muphi2**2 * correction05 * (29 - exp(-mupsi2 * startN) * (9 * mupsi2**2 - 65 * mupsi2 + 58) + exp(-2 * mupsi2 * startN) * (14 * mupsi2**2 - 65 * mupsi2 + 29))
        correction65 = 15 / 8 * muphi2**3 * correction05 * (1 - exp(-mupsi2 * startN))**3
        correction3 = correction05 + correction25 + correction45 + correction65

        # Set up the initial conditions
        correction = correction1 + correction2 + correction3
        R0 = exp(-startN) + correction
        # TODO: Check the derivative initial condition corrections for Rdot0 and deltadot0. Pretty sure they're wrong.
        # Rdot0 = - exp(-startN) + correction + muphi2**2 / 2 * correction01 * exp(-mupsi2 * startN)
        
        # Convert to delta = log(R) + N
        delta0 = log(R0 * exp(startN))  # To avoid catastropic loss of precision due to cancellation
        # deltadot0 = Rdot0 / R0 + 1    # Affected by catastrophic loss of precision due to cancellation
        # The below is a first-order approximation that will be more accurate
        deltadot0 = (correction + muphi2**2 / 2 * correction01 * exp(-mupsi2 * startN)) * exp(startN)
        
        # For now, set up initial conditions to match Mathematica
        # TODO: Remove these lines
        delta0 *= 0
        deltadot0 *= 0
        
        ics = np.concatenate([delta0, deltadot0, startN])

        def derivs(t: float, x: np.array) -> np.array:
            """Compute the time derivatives for the mode function evolution ODE"""
            # Extract values
            delta = x[0:num_k]
            deltadot = x[num_k:2*num_k]
            Nvals = x[2*num_k:3*num_k]
            # Compute deltaddot
            deltaddot = - deltadot**2 - deltadot + 2 + kvals2 * exp(-2 * Nvals) * expm1(-4 * delta) - muphi2 * (exp(-mupsi2 * Nvals) - 1)
            # N increases linearly in time
            Ndot = np.ones_like(Nvals)
            # Return results
            derivatives = np.concatenate([deltadot, deltaddot, Ndot])
            return derivatives

        # Set up the integrator
        integrator = ode(derivs).set_integrator('dop853', rtol=self.err_rel, atol=self.err_abs, nsteps=100000, first_step=1e-6, max_step=1e-2)
        integrator.set_initial_value(ics, minN)

        # Save initial conditions
        Rvals = [exp(integrator.y[0:num_k] - integrator.y[2*num_k:3*num_k])]
        times = [integrator.y[2*num_k:3*num_k]]

        # Perform integration
        while integrator.successful() and integrator.t < endN:
            newN = integrator.t + 0.1
            if newN > endN:
                newN = endN + 1e-8
            integrator.integrate(newN)

            # Save results
            timevals = integrator.y[2 * num_k:3 * num_k]
            times.append(timevals)

            rvals = integrator.y[0:num_k] - timevals
            Rvals.append(exp(rvals))
            
        # Convert results into one big array
        Rvals = np.array(Rvals)
        times = np.array(times)

        self.df_rvals = pd.DataFrame(Rvals, columns=list(kvals))
        self.df_times = pd.DataFrame(times, columns=list(kvals))
        
        # For each k value, construct an interpolator over the R values and times to get the R value at N = endN
        Rend = []
        for idx, k in enumerate(kvals):
            interp = InterpolatedUnivariateSpline(times[:, idx], Rvals[:, idx], k=3, ext='raise')
            Rend.append(interp(endN))
        Rend = np.array(Rend)

        # Compute the power spectrum!
        self.spectrum = Rend**2 / (2*pi)**3 / 2 / kvals
        
        # Normalize the spectrum so that the first value is 100
        self.spectrum *= 100 / self.spectrum[0]
