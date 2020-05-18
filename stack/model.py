"""
model.py

Contains the Model class, which stores the definitions of the given model.
"""
from __future__ import annotations

import os
from math import sqrt

from stack.powerspectrum import PowerSpectrum
from stack.moments import Moments
from stack.integrals import SingleBessel
from stack.grid import Grid
from stack.common import Suppression
from stack.correlations import Correlations

class Model(object):
    """Master class that controls all aspects of modelling"""

    def __init__(self,
                 # Model parameters
                 model_name: str,
                 # Inflation model parameters
                 n_efolds: float,
                 n_fields: int,
                 mpsi: float,
                 m0: float,
                 # Power spectrum parameters
                 min_k: float = 1e-5,
                 num_modes: int = 401,
                 max_k: float = 1e3,
                 test_ps: bool = False,
                 # Grid parameters
                 rmaxfactor: float = 20,
                 gridpoints: int = 200,
                 # Sampling parameters
                 sampling_cutoff_factor: float = 1.0,
                 # Control options
                 recompute_all: bool = False,
                 verbose: bool = False,
                 debug: bool = False,
                 ) -> None:
        """
        Initialize the model object.

        Model options
        :param model_name: Name of the model (directory to save to)
        
        Inflation model paramters
        :param n_efolds: Number of efolds from waterfall transition to end of inflation
        :param n_fields: Number of waterfall fields
        :param mpsi: Mass of psi field (units of H)
        :param m0: Mass of m0 field (units of H)
        
        Power spectrum parameters
        :param min_k: Minimum k to compute power spectrum at
        :param num_modes: Number of logarithmic steps to take for modes
        :param max_k: Maximum k to compute power spectrum at
        :param test_ps: Use a dummy power spectrum
        
        Grid parameters
        :param rmaxfactor: Number of FWHMs to go out to get to rmax
        :param gridpoints: Number of radial gridpoints to use in physical space (excluding origin)
        
        Sampling parameters
        :param sampling_cutoff_factor: Factor by which to multiply the Nyquist cutoff wavenumber to obtain the power
                                       spectrum cutoff wavenumber
        
        Control options
        :param recompute_all: Force recomputation of everything (do not load data)
        :param verbose: Enable verbose output
        :param debug: Enable debug output
        """
        # Store parameters
        # Model options
        self.model_name = model_name.title()
        self.path = os.path.join('models', model_name.lower())

        # Inflation model parameters
        self.n_efolds = n_efolds
        self.n_fields = n_fields
        self.m0 = m0
        self.mpsi = mpsi

        # Power spectrum parameters
        self.min_k = min_k
        self.num_modes = num_modes
        self.max_k = max_k
        self.test_ps = test_ps
        
        # Grid parameters
        self.rmaxfactor = rmaxfactor
        self.gridpoints = gridpoints
        
        # Sampling parameters
        self.sampling_cutoff_factor = sampling_cutoff_factor

        # Control options
        self.recompute_all = recompute_all
        self.verbose = verbose
        self.debug = debug

        # See if we should write some output
        if self.verbose:
            print(f'Initializing {self.model_name} model...')

        # Compute derivative quantities
        self.mupsi2 = 3 - sqrt(9 - 4*mpsi**2)
        self.muphi2 = m0**2
        self.lamda = -3/2 + sqrt(9/4 + m0**2)
        self.beta = 1/(2*self.lamda)

        # Ensure that the relevant folder exists
        os.makedirs(self.path, exist_ok=True)

        # Store object class instances
        self.powerspectrum = PowerSpectrum(self)
        self.moments_raw = Moments(self, Suppression.RAW)
        self.singlebessel = SingleBessel(self)
        self.grid = Grid(self)
        self.moments_sampling = Moments(self, Suppression.SAMPLING)
        self.correlations = Correlations(self)

    def construct_powerspectrum(self, recalculate: bool = False) -> None:
        """Construct the data for the power spectrum (either by loading or constructing it)"""
        print('Constructing the power spectrum...')
        self.powerspectrum.construct_data(recalculate=recalculate)
        print('    Done!')

    def construct_moments(self, recalculate: bool = False) -> None:
        """Construct the data for the moments of the power spectrum (either by loading or constructing them)"""
        print('Constructing raw moments of the power spectrum...')
        assert self.powerspectrum.ready
        self.moments_raw.construct_data(prev_timestamp=self.powerspectrum.timestamp, recalculate=recalculate)
        print('    Done!')

    def construct_singlebessel(self, recalculate: bool = False) -> None:
        """Initialize single bessel integrals"""
        print('Initializing single bessel integrals...')
        assert self.moments_raw.ready
        self.singlebessel.construct_data(prev_timestamp=self.moments_raw.timestamp, recalculate=recalculate)
        print('    Done!')

    def construct_grid(self, recalculate: bool = False) -> None:
        """Initialize grid"""
        print('Initializing grid...')
        assert self.singlebessel.ready
        self.grid.construct_data(prev_timestamp=self.singlebessel.timestamp, recalculate=recalculate)
        print('    Done!')

    def construct_moments2(self, recalculate: bool = False) -> None:
        """Construct moments for the power spectrum with sampling suppression"""
        print('Constructing sampling moments of the power spectrum...')
        assert self.grid.ready
        self.moments_sampling.construct_data(prev_timestamp=self.grid.timestamp, recalculate=recalculate)
        print('    Done!')

    def construct_correlations(self, recalculate: bool = False) -> None:
        """Construct correlation functions C(r) and D(r) on the grid"""
        print('Constructing correlations...')
        assert self.moments_sampling.ready
        self.correlations.construct_data(prev_timestamp=self.moments_sampling.timestamp, recalculate=recalculate)
        print('    Done!')
