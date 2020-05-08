"""
moments.py

Computes moments of the power spectrum.
"""
from __future__ import annotations

import pandas as pd
from math import sqrt, pi
from scipy.integrate import quad

from typing import TYPE_CHECKING

from stack.common import Persistence

if TYPE_CHECKING:
    from stack import Model

class Moments(Persistence):
    """
    Computes moments of a power spectrum
    """
    filename = 'moments'

    def __init__(self, model: 'Model') -> None:
        """
        Initialize the class.
        
        :param model: Model class
        """
        super().__init__(model)

        # Initialize storage for results
        self.sigma0 = None
        self.sigma1 = None
        self.sigma2 = None
        self.sigma0squared = None
        self.sigma1squared = None
        self.sigma2squared = None
        self.gamma = None
        
        # Error tolerances used in computing moment integrals
        self.err_abs = 0
        self.err_rel = 5e-6

    def load_data(self) -> None:
        """Load the power spectrum from file"""
        filename = self.filename + '.csv'
        path = self.file_path(filename)
        if not self.file_exists(filename):
            raise FileNotFoundError(f'Unable to load from {path}')

        df = pd.read_csv(path)

        self.sigma0squared = df.sigma0squared[0]
        self.sigma1squared = df.sigma1squared[0]
        self.sigma2squared = df.sigma2squared[0]
        self.compute_dependent()

    def save_data(self) -> None:
        """Saves the moments to file"""
        df = pd.DataFrame([[self.sigma0squared, self.sigma1squared, self.sigma2squared]],
                          columns=['sigma0squared', 'sigma1squared', 'sigma2squared'])
        df.to_csv(self.file_path(self.filename + '.csv'), index=False)

    def compute_data(self) -> None:
        """Compute the moments of the power spectrum"""
        self.sigma0squared = self.compute_sigma_n_squared(0)
        self.sigma1squared = self.compute_sigma_n_squared(1)
        self.sigma2squared = self.compute_sigma_n_squared(2)
        self.compute_dependent()
        
    def compute_dependent(self) -> None:
        """Computes and stores dependent quantities"""
        self.sigma0 = sqrt(self.sigma0squared)
        self.sigma1 = sqrt(self.sigma1squared)
        self.sigma2 = sqrt(self.sigma2squared)
        self.gamma = self.sigma1squared / (self.sigma0 * self.sigma2)

    def compute_sigma_n_squared(self, n: int) -> float:
        """Computes and returns the n^th moment of the power spectrum"""
        spectrum = self.model.powerspectrum

        def func(k):
            spec = spectrum(k)
            if spec < 0:
                raise ValueError(f'Found negative value for spectrum at k={k}, P(k)={spec}')
            return k ** (2 + 2 * n) * spec
        
        integral, err = quad(func, spectrum.min_k, spectrum.max_k, epsabs=self.err_abs, epsrel=self.err_rel)

        return 4 * pi * integral
