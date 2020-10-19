"""
model.py

Contains the Model class, which stores the definitions of the given model.
"""
from __future__ import annotations

import os
from math import sqrt, exp
import time

from stack.powerspectrum import PowerSpectrum
from stack.moments import Moments
from stack.integrals import SingleBessel, DoubleBessel
from stack.grid import Grid
from stack.common import Suppression
from stack.correlations import Correlations
from stack.peakdensity import PeakDensity

class Model(object):
    """Master class that controls all aspects of modelling"""

    def __init__(self,
                 # Model parameters
                 model_name: str,
                 # Inflation model parameters
                 n_efolds: float = 15,
                 n_fields: int = 4,
                 mpsi: float = 0.1,
                 m0: float = 10,
                 potential_r: int = 2,
                 # Power spectrum parameters
                 min_k: float = 1e-5,
                 num_modes: int = 1001,
                 max_k: float = 250,
                 test_ps: bool = False,
                 # Grid parameters
                 rmaxfactor: float = 20,
                 gridpoints: int = 10,
                 ell_max: int = 1,
                 # Sampling parameters
                 sampling_cutoff_factor: float = 1.0,
                 # Number density of peaks parameters
                 peakdensity_samples: int = 1e5,
                 nu_steps: int = 50,
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
        :param m0: Mass of m0 field (units of H, also known as mu_phi)
        :param potential_r: Power of the potential
        
        Power spectrum parameters
        :param min_k: Minimum k to compute power spectrum at
        :param num_modes: Number of logarithmic steps to take for modes
        :param max_k: Maximum k to compute power spectrum at
        :param test_ps: Use a dummy power spectrum
        
        Grid parameters
        :param rmaxfactor: Number of FWHMs to go out to get to rmax
        :param gridpoints: Number of radial gridpoints to use in physical space (excluding origin)
        :param ell_max: Maximum ell to take for spherical harmonics
        
        Sampling parameters
        :param sampling_cutoff_factor: Factor by which to multiply the Nyquist cutoff wavenumber to obtain the power
                                       spectrum cutoff wavenumber
                                       
        Number density of peaks parameters
        :param peakdensity_samples: Number of samples to use
        :param nu_steps: Number of steps to sample for nu
        
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
        self.potential_r = potential_r

        # Power spectrum parameters
        self.min_k = min_k
        self.num_modes = num_modes
        self.max_k = max_k
        self.test_ps = test_ps
        
        # Grid parameters
        self.rmaxfactor = rmaxfactor
        self.gridpoints = gridpoints
        self.ell_max = ell_max
        
        # Sampling parameters
        self.sampling_cutoff_factor = sampling_cutoff_factor
        
        # Number density of peaks parameters
        self.peakdensity_samples = peakdensity_samples
        self.nu_steps = nu_steps

        # Control options
        self.recompute_all = recompute_all
        self.verbose = verbose
        self.debug = debug

        # See if we should write some output
        if self.verbose:
            print(f'Initializing {self.model_name} model...')

        # Compute derivative quantities
        self.mupsi2 = self.potential_r / 2 * (3 - sqrt(9 - 4*mpsi**2))
        self.muphi2 = m0**2
        self.lamda = (-3 + sqrt(9 + 4 * self.muphi2 * (1 - exp(-self.mupsi2 * self.n_efolds)))) / 2    # Eq. 39, https://arxiv.org/pdf/1210.8128.pdf  # TODO: should this include r?
        self.beta = 1/(2*self.lamda)

        # Ensure that the relevant folder exists
        os.makedirs(self.path, exist_ok=True)

        # Store object class instances
        self.powerspectrum = PowerSpectrum(self)
        self.moments_raw = Moments(self, Suppression.RAW)
        self.singlebessel = SingleBessel(self)
        self.doublebessel = DoubleBessel(self)
        self.grid = Grid(self)
        self.moments_sampling = Moments(self, Suppression.SAMPLING)
        self.correlations = Correlations(self)
        # Need a class that computes expected peak shape here
        self.moments_peaks = Moments(self, Suppression.PEAKS)
        self.peakdensity = PeakDensity(self)
        
    def get_moments(self, suppression: Suppression = Suppression.RAW):
        """Return the appropriate moments class, given the suppression method"""
        if suppression == Suppression.RAW:
            return self.moments_raw
        elif suppression == Suppression.SAMPLING:
            return self.moments_sampling
        elif suppression == Suppression.PEAKS:
            return self.moments_peaks
        else:
            raise ValueError(f'Bad suppression method: {suppression}')

    def construct_powerspectrum(self, recalculate: bool = False) -> None:
        """Construct the data for the power spectrum (either by loading or constructing it)"""
        print('Constructing the power spectrum...')
        start_time = time.time()
        self.powerspectrum.construct_data(recalculate=recalculate)
        end_time = time.time()
        print(f'    Done in {end_time - start_time:0.2f}s')

    def construct_moments(self, recalculate: bool = False) -> None:
        """Construct the data for the moments of the power spectrum (either by loading or constructing them)"""
        print('Constructing raw moments of the power spectrum...')
        start_time = time.time()
        assert self.powerspectrum.ready
        self.moments_raw.construct_data(prev_timestamp=self.powerspectrum.timestamp, recalculate=recalculate)
        end_time = time.time()
        print(f'    Done in {end_time - start_time:0.2f}s')

    def construct_singlebessel(self, recalculate: bool = False) -> None:
        """Initialize single bessel integrals"""
        print('Initializing single bessel integrals...')
        start_time = time.time()
        assert self.moments_raw.ready
        self.singlebessel.construct_data(prev_timestamp=self.moments_raw.timestamp, recalculate=recalculate)
        end_time = time.time()
        print(f'    Done in {end_time - start_time:0.2f}s')

    def construct_doublebessel(self, recalculate: bool = False) -> None:
        """Initialize double bessel integrals"""
        print('Initializing double bessel integrals...')
        start_time = time.time()
        assert self.singlebessel.ready
        self.doublebessel.construct_data(prev_timestamp=self.singlebessel.timestamp, recalculate=recalculate)
        end_time = time.time()
        print(f'    Done in {end_time - start_time:0.2f}s')

    def construct_grid(self, recalculate: bool = False) -> None:
        """Initialize grid"""
        print('Initializing grid...')
        start_time = time.time()
        assert self.doublebessel.ready
        self.grid.construct_data(prev_timestamp=self.singlebessel.timestamp, recalculate=recalculate)
        end_time = time.time()
        print(f'    Done in {end_time - start_time:0.2f}s')

    def construct_moments2(self, recalculate: bool = False) -> None:
        """Construct moments for the power spectrum with sampling suppression"""
        print('Constructing sampling moments of the power spectrum...')
        start_time = time.time()
        assert self.grid.ready
        self.moments_sampling.construct_data(prev_timestamp=self.grid.timestamp, recalculate=recalculate)
        end_time = time.time()
        print(f'    Done in {end_time - start_time:0.2f}s')

    def construct_correlations(self, recalculate: bool = False) -> None:
        """Construct correlation functions C(r) and D(r) on the grid"""
        print('Constructing correlations...')
        start_time = time.time()
        assert self.moments_sampling.ready
        self.correlations.construct_data(prev_timestamp=self.moments_sampling.timestamp, recalculate=recalculate)
        end_time = time.time()
        print(f'    Done in {end_time - start_time:0.2f}s')

    def construct_moments3(self, recalculate: bool = False) -> None:
        """Construct moments for the power spectrum with peaks suppression"""
        print('Constructing peaks moments of the power spectrum...')
        # TODO: Change to assert that the peak shape computation is ready
        start_time = time.time()
        assert self.correlations.ready
        self.moments_peaks.construct_data(prev_timestamp=self.correlations.timestamp, recalculate=recalculate)
        end_time = time.time()
        print(f'    Done in {end_time - start_time:0.2f}s')

    def construct_peakdensity(self, recalculate: bool = False) -> None:
        """Construct number density of peaks"""
        print('Constructing number density of peaks...')
        start_time = time.time()
        assert self.moments_peaks.ready
        self.peakdensity.construct_data(prev_timestamp=self.moments_peaks.timestamp, recalculate=recalculate)
        end_time = time.time()
        print(f'    Done in {end_time - start_time:0.2f}s')
