"""
peakdensity.py

Computes the number density of peaks of a chi-squared distribution.
"""
from __future__ import annotations

import numpy as np
import pyximport

pyximport.install(setup_args={'include_dirs': np.get_include()}, language_level=2)  # pyx file is written in python 2

import pandas as pd
import vegas
from math import pi, exp, sqrt
from scipy.special import gamma

from typing import TYPE_CHECKING, Tuple

import stack.peakdensity.integrand as integrand
from stack.common import Persistence

if TYPE_CHECKING:
    from stack import Model

class PeakDensity(Persistence):
    """
    Computes number density of peaks of a chi-squared distribution.
    """
    filename = 'peakdensity'

    def __init__(self, model: 'Model') -> None:
        """
        Initialize the class.
        
        :param model: Model class
        """
        super().__init__(model)
        
        # Grab parameters from model
        self.peakdensity_samples = model.peakdensity_samples
        self.nu_steps = model.nu_steps
        self.n_fields = model.n_fields
        
        # Construct the grid in nu
        # We compute number density of peaks values ranging from nu = 0 to nu = 3 * sqrt(n) (well past peaks, but useful for checking)
        self.nu_min = 0
        self.nu_max = 3 * sqrt(self.n_fields)
        self.nu_vals = np.linspace(self.nu_min, self.nu_max, self.nu_steps)
        
        # Initialize storage for results
        # Peak densities
        self.min_vec = np.zeros_like(self.nu_vals)
        self.saddleppm_vec = np.zeros_like(self.nu_vals)
        self.saddlepmm_vec = np.zeros_like(self.nu_vals)
        self.max_vec = np.zeros_like(self.nu_vals)
        # Error estimates
        self.min_err_vec = np.zeros_like(self.nu_vals)
        self.saddleppm_err_vec = np.zeros_like(self.nu_vals)
        self.saddlepmm_err_vec = np.zeros_like(self.nu_vals)
        self.max_err_vec = np.zeros_like(self.nu_vals)
        # Analytic results
        self.signed_vec = np.zeros_like(self.nu_vals)

    def load_data(self) -> None:
        """Load the peak density data from file"""
        filename = self.filename + '.csv'
        path = self.file_path(filename)
        if not self.file_exists(filename):
            raise FileNotFoundError(f'Unable to load from {path}')

        df = pd.read_csv(path)

        self.nu_vals = df['nu'].values
        self.min_vec = df['min'].values
        self.saddleppm_vec = df['saddleppm'].values
        self.saddlepmm_vec = df['saddlepmm'].values
        self.max_vec = df['max'].values
        self.min_err_vec = df['min_err'].values
        self.saddleppm_err_vec = df['saddleppm_err'].values
        self.saddlepmm_err_vec = df['saddlepmm_err'].values
        self.max_err_vec = df['max_err'].values
        self.signed_vec = df['signed'].values

    def save_data(self) -> None:
        """Saves the peak density data to file"""
        df = pd.DataFrame([self.nu_vals,
                           self.min_vec,
                           self.saddleppm_vec,
                           self.saddlepmm_vec,
                           self.max_vec,
                           self.min_err_vec,
                           self.saddleppm_err_vec,
                           self.saddlepmm_err_vec,
                           self.max_err_vec,
                           self.signed_vec]).transpose()
        df.columns = ['nu', 'min', 'saddleppm', 'saddlepmm', 'max',
                      'min_err', 'saddleppm_err', 'saddlepmm_err', 'max_err', 'signed']
        df.to_csv(self.file_path(self.filename + '.csv'), index=False)

    def compute_data(self) -> None:
        """Compute the number density of peaks at each nu value"""
        # Grab values we need
        gamma = self.model.moments_peaks.gamma
        sigma0 = self.model.moments_peaks.sigma0
        sigma1 = self.model.moments_peaks.sigma1
        
        print(sigma0, sigma1, gamma)
        
        # Compute numbers at each value of nu
        for idx, nu in enumerate(self.nu_vals):
            if self.model.verbose:
                print(f'    Computing {idx+1}/{len(self.nu_vals)} at nu={nu}')
            if nu == 0:
                # We need to treat nu=0 as a special case
                exact = signed_exact(self.n_fields, 0.0, sigma0, sigma1)
                values = np.array([0, exact, 0, 0, 0])
                err = np.array([0, 0, 0, 0, 0])
                self.assign_value(idx, values, err, exact)
            else:
                # Invoke the vegas routines
                values, err = number_density(int(self.n_fields), gamma, nu, sigma0, sigma1, int(self.peakdensity_samples))
                signedval = signed_exact(self.n_fields, nu, sigma0, sigma1)
                self.assign_value(idx, values, err, signedval)
    
    def assign_value(self, idx: int, values: np.array, err: np.array, signedval: float) -> None:
        """
        Stores integration values at the specified index
        Assumes that values and err are provided in the form [unused, min, signedppm, signedpmm, max]
        """
        self.min_vec[idx] = values[1]
        self.saddleppm_vec[idx] = values[2]
        self.saddlepmm_vec[idx] = values[3]
        self.max_vec[idx] = values[4]
        self.min_err_vec[idx] = err[1]
        self.saddleppm_err_vec[idx] = err[2]
        self.saddlepmm_err_vec[idx] = err[3]
        self.max_err_vec[idx] = err[4]
        self.signed_vec[idx] = signedval


def number_density(n: int, gamma_val: float, nu: float, sigma0: float, sigma1: float,
                   num_samples: int) -> Tuple[np.array, np.array]:
    """Use vegas to compute the number density of peaks for the given parameters"""
    # Set up the domain for our 9 variables
    domain = [[-pi / 2, pi / 2],
              [-pi / 2, pi / 2],
              [-pi / 2, pi / 2],
              [0, pi / 2],
              [0, pi / 2],
              [0, pi / 2],
              [-pi / 2, pi / 2],
              [-pi / 2, pi / 2],
              [-pi / 2, pi / 2]]

    # Initialize the integrator
    integ = vegas.Integrator(domain, nhcube_batch=1000)

    # Construct the integration function
    f = integrand.f_cython(dim=9, N=n, nu=nu, gamma=gamma_val)

    # Perform the integration
    # TODO: Is this next line redundant? I'm turning it off here...
    # integ(f, nitn=10, neval=num_samples)
    vecresult = integ(f, nitn=10, neval=num_samples)
    
    # Compute the scaling prefactor
    prefactor = scale(n, nu, sigma0, sigma1) / V_n(n, gamma_val)

    # Extract results
    # signed, min, saddle++-, saddle+--, max
    integrals = np.zeros(5)
    errors = np.zeros(5)
    for i in range(5):
        integrals[i] = prefactor * vecresult[i].mean
        errors[i] = prefactor * vecresult[i].sdev
    
    return integrals, errors

def scale(n: int, nu: float, sigma0: float, sigma1: float) -> float:
    """Compute alpha dP/dnu"""
    if nu > 0:
        alpha = 1 / (6 * pi)**1.5 * (sigma1 / sigma0 / nu)**3
        dpdnu = nu ** (n - 1) * exp(-nu * nu / 2) / 2 ** (n / 2 - 1) / gamma(n / 2)
        return alpha * dpdnu
    elif nu == 0.0:
        if n > 4:
            return 0.0
        elif n == 4:
            return 1 / (6 * pi) ** 1.5 * (sigma1 / sigma0) ** 3 * exp(-nu*nu / 2) / 2 ** (n / 2 - 1) / gamma(n / 2)
        elif n < 4:
            raise ValueError("Scale diverges for n < 4 at nu = 0")
    else:
        raise ValueError("Scale is not defined for nu < 0")

def signed_exact(n: int, nu: float, sigma0: float, sigma1: float) -> float:
    """Compute the exact signed number density"""
    val = (n - 1)*(n - 2)*(n - 3) - 3*(n - 1)*(n - 1)*nu**2 + 3*n*nu**4 - nu**6
    return val * scale(n, nu, sigma0, sigma1)

def V_n(n: int, gamma_val: float):
    """Compute V_n, the normalization factor for the MCMC integral"""
    # Compute the multivariate gamma function (gamma3)
    x = (n - 1) / 2
    gamma3 = pi ** 1.5 * gamma(x) * gamma(x - 0.5) * gamma(x - 1.0)
    # Compute the result
    return 2 ** (1.5 * (n - 1)) / 5 ** 2.5 / 27 * pi * sqrt(1 - gamma_val * gamma_val) * gamma3
