"""
moments.py

Computes moments and characteristic lengthscale of the power spectrum.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from math import sqrt, pi
from scipy.integrate import quad
from scipy import optimize

from typing import TYPE_CHECKING

from stack.common import Persistence, Suppression

if TYPE_CHECKING:
    from stack import Model

class Moments(Persistence):
    """
    Computes moments of a power spectrum, as well as the characteristic lengthscale
    """
    @property
    def filename(self) -> str:
        """Returns the filename for this class"""
        return f'moments-{self.suppression.value}'

    def __init__(self, model: 'Model', suppression: Suppression) -> None:
        """
        Initialize the class.
        
        :param model: Model class
        :param suppression: Type of suppression to use on the power spectrum when computing moments
        """
        super().__init__(model)
        
        # Store parameters
        self.suppression = suppression

        # Initialize storage for results
        self.sigma0 = None
        self.sigma1 = None
        self.sigma2 = None
        self.sigma0squared = None
        self.sigma1squared = None
        self.sigma2squared = None
        self.gamma = None
        self.lengthscale = None
        self.k2peak = None
        self.k3peak = None
        self.k4peak = None

        # Error tolerances used in computing moment integrals
        self.err_abs = 0
        self.err_rel = 1e-7

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
        self.lengthscale = df.lengthscale[0]
        self.k2peak = df.k2peak[0]
        self.k3peak = df.k3peak[0]
        self.k4peak = df.k4peak[0]
        self.compute_dependent()

    def save_data(self) -> None:
        """Saves the moments to file"""
        df = pd.DataFrame([[self.sigma0squared, self.sigma1squared, self.sigma2squared, self.lengthscale, self.k2peak, self.k3peak, self.k4peak]],
                          columns=['sigma0squared', 'sigma1squared', 'sigma2squared', 'lengthscale', 'k2peak', 'k3peak', 'k4peak'])
        df.to_csv(self.file_path(self.filename + '.csv'), index=False)

    def compute_data(self) -> None:
        """Compute the moments of the power spectrum"""
        self.sigma0squared = self.compute_sigma_n_squared(0)
        self.sigma1squared = self.compute_sigma_n_squared(1)
        self.sigma2squared = self.compute_sigma_n_squared(2)
        self.compute_lengthscales()
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
        suppression = self.suppression

        def func(k):
            spec = spectrum(k, suppression)
            if spec < 0:
                raise ValueError(f'Found negative value for spectrum at k={k}, P(k)={spec}')
            return k ** (2 + 2 * n) * spec
        
        integral, err = quad(func, spectrum.min_k, spectrum.max_k, epsabs=self.err_abs, epsrel=self.err_rel)

        return 4 * pi * integral

    def compute_lengthscales(self):
        """
        Computes the lengthscale of a power spectrum by finding the location of the maximum of k^2 P(k).
        Also stores the maximum of k^3 P(k) and k^4 P(k).
        """
        min_k = self.model.min_k
        max_k = self.model.max_k

        # k^2 P(k) peak
        def f(k):
            return - k*k*self.model.powerspectrum(k, self.suppression)
        
        result = optimize.minimize_scalar(f, method='bounded', bounds=(min_k, max_k),
                                          options={'xatol': 1e-5})

        if not result.success:
            raise ValueError("Unable to find characteristic lengthscale of power spectrum")
        
        self.k2peak = result.x
        self.lengthscale = 2 * pi / self.k2peak

        # k^3 P(k) peak
        def f(k):
            return - k * k * k * self.model.powerspectrum(k, self.suppression)

        result = optimize.minimize_scalar(f, method='bounded', bounds=(min_k, max_k),
                                          options={'xatol': 1e-5})

        if not result.success:
            raise ValueError("Unable to find characteristic lengthscale of power spectrum")

        self.k3peak = result.x

        # k^4 P(k) peak
        def f(k):
            return - k * k * k * k * self.model.powerspectrum(k, self.suppression)

        result = optimize.minimize_scalar(f, method='bounded', bounds=(min_k, max_k),
                                          options={'xatol': 1e-5})

        if not result.success:
            raise ValueError("Unable to find characteristic lengthscale of power spectrum")

        self.k4peak = result.x
