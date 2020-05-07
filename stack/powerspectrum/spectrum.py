"""
Power spectrum of waterfall field perturbations in hybrid inflation
v1.0 by Jolyon Bloomfield, April 2017
See arXiv:xxxx.xxxxx for details
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy import exp, log, pi, log10
from scipy.integrate import ode
from scipy.interpolate import InterpolatedUnivariateSpline

from typing import TYPE_CHECKING

from stack.common import Persistence

if TYPE_CHECKING:
    from stack import Model

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
        self.err_abs = 1e-10
        self.err_rel = 1e-13
        self.df_rvals = None
        self.df_times = None

    def load_data(self) -> None:
        """Load the power spectrum from file"""
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
        # Save the power spectrum
        df = pd.DataFrame([self.kvals, self.spectrum]).transpose()
        df.columns = ['k', 'spectrum']
        df.to_csv(self.file_path(self.filename + '.csv'), index=False)

        # Also save the mode evolution (not loaded back in when loading data)
        self.df_rvals.to_csv(self.file_path(self.filename + '-Rvals.csv'), index=False)
        self.df_times.to_csv(self.file_path(self.filename + '-Nvals.csv'), index=False)

    def compute_data(self) -> None:
        """Compute the power spectrum for the model"""
        self.construct_grid()
        self.construct_spectrum()
        self.construct_interpolant()

    def __call__(self, k) -> float:
        """Return the value of the power spectrum at a given k value"""
        return self.interp(k)

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
        self.interp = InterpolatedUnivariateSpline(self.kvals, self.spectrum, k=3, ext='raise')

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
        # TODO: Check the derivative initial condition corrections
        Rdot0 = - exp(-startN) + correction + muphi2**2 / 2 * correction01 * exp(-mupsi2 * startN)
        
        # Convert to delta = log(R) - N
        delta0 = log(R0) + startN
        deltadot0 = Rdot0 / R0 + 1
        
        ics = np.concatenate([delta0, deltadot0, startN])

        def derivs(t: float, x: np.array) -> np.array:
            """Compute the time derivatives for the mode function evolution ODE"""
            # Extract values
            delta = x[0:num_k]
            deltadot = x[num_k:2*num_k]
            Nvals = x[2*num_k:3*num_k]
            # Compute the second derivative of R
            deltaddot = - (deltadot**2 + deltadot - 2 + kvals2 * exp(-2 * Nvals) * (1 - exp(-4 * delta)) + muphi2 * (exp(-mupsi2 * Nvals) - 1))
            # N increases linearly in time
            Ndot = np.ones_like(Nvals)
            # Return results
            derivatives = np.concatenate([deltadot, deltaddot, Ndot])
            return derivatives

        # Set up the integrator
        integrator = ode(derivs).set_integrator('dop853', rtol=self.err_rel, atol=self.err_abs, nsteps=100000)
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
