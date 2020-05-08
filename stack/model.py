"""
model.py

Contains the Model class, which stores the definitions of the given model.
"""
from __future__ import annotations

import os
from math import sqrt

from stack.powerspectrum import PowerSpectrum
from stack.moments import Moments

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
                 num_modes: int = 201,
                 max_k: float = 1e3,
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
        self.moments = Moments(self)
        
    def construct_powerspectrum(self, recalculate: bool = False) -> None:
        """Construct the data for the power spectrum (either by loading or constructing it)"""
        print('Constructing the power spectrum...')
        self.powerspectrum.construct_data(recalculate=recalculate)
        print('    Done!')

    def construct_moments(self, recalculate: bool = False) -> None:
        """Construct the data for the moments of the power spectrum (either by loading or constructing them)"""
        print('Constructing moments of the power spectrum...')
        assert self.powerspectrum.ready
        self.moments.construct_data(prev_timestamp=self.powerspectrum.timestamp, recalculate=recalculate)
        print('    Done!')
