"""
singlebessel.py

Code to compute integrals of the form
int_{k_min}^{k_max} dk k^2 P(k) j_l(k r)
where l = 0 or 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from math import pi
from scipy.integrate import quad
from scipy.special import spherical_jn

from stack.common import Persistence, Suppression

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stack import Model

# Raise errors on issues rather than printing warnings
np.seterr(all='raise')


class SingleBessel(Persistence):
    """
    Computes single bessel integrals over the power spectrum.
    """
    filename = 'singlebessel'

    def __init__(self, model: 'Model') -> None:
        """
        Initialize the class.
        
        :param model: Model class we are computing integrals for.
        """
        super().__init__(model)
        
        # Error tolerances used in computing integrals
        self.err_abs = 0
        self.err_rel = 1e-8

    def load_data(self) -> None:
        """This class does not save any data, so has nothing to load"""
        pass

    def compute_data(self) -> None:
        """This class does not save any data, so has nothing to compute"""
        pass

    def save_data(self) -> None:
        """This class does not save any data, but does output some results for comparison with Mathematica"""
        # Construct a grid in physical space
        rvals = np.logspace(start=-3,
                            stop=3,
                            num=21,
                            endpoint=True)
        # Compute C and D on that grid
        Cvals = np.array([self.compute_C(r, Suppression.RAW) for r in rvals])
        Dvals = np.array([self.compute_D(r, Suppression.RAW) for r in rvals])
        # Save them to file
        df = pd.DataFrame([rvals, Cvals, Dvals]).transpose()
        df.columns = ['r', 'C(r)', 'D(r)']
        df.to_csv(self.file_path(self.filename + '.csv'), index=False)

    def compute_C(self, r: float, suppression: Suppression) -> float:
        """
        Computes the integral
        C(r) = 4 pi int_{k_min}^{k_max} dk k^2 P(k) j_0(k r)
        
        We've tested this against Mathematica for the dummy power spectrum, and find agreement
        to 10^-14 (relative).
        
        :param r: Value of r to use in the integral
        :param suppression: High frequency suppression method to use
        :return: Result of the integral
        """
        # Treat the special case
        if r == 0:
            if suppression == Suppression.RAW:
                return self.model.moments_raw.sigma0squared
            elif suppression == Suppression.SAMPLING:
                return self.model.moments_sampling.sigma0squared
            else:
                raise ValueError(f'Bad suppression method: {suppression}')
        
        pk = self.model.powerspectrum
        min_k = self.model.min_k
        max_k = self.model.max_k
        if suppression == suppression.SAMPLING:
            # No need to go far out into the tail of the suppressing Gaussian
            # At k = n k_0, the suppression is exp(-n^2/2)
            # At n = 6, this is ~10^-8, which seems like a good place to stop
            max_k = min(max_k, self.model.grid.sampling_cutoff * 6)

        def f(k):
            return k*pk(k, suppression)

        # Perform the integration
        result = quad(f, min_k, max_k, weight='sin', wvar=r,
                      epsrel=self.err_rel, epsabs=self.err_abs, full_output=1, limit=60)
        # Check for any warnings
        if len(result) == 4:
            print('Warning when integrating C(r) at r =', r)
            print(result[-1])
        
        # Rescale the result
        integral = result[0] * 4 * pi / r

        return integral

    def compute_D(self, r: float, suppression: Suppression) -> float:
        """
        Computes the integral
        D(r) = 4 pi int_{k_min}^{k_max} dk k^3 P(k) j_1(k r)
        
        Note that j_1(k r) = sin(k r) / (k r)^2 - cos(k r) / (k r)
        To do the integral, we do a sin integral and a cos integral separately, and difference the two:

        D(r) = 4 pi int_{k_min}^{k_max} dk k^3 P(k) (sin(k r) / (k r)^2 - cos(k r) / (k r))
             =   (4 pi)/r^2 int_{k_min}^{k_max} dk k P(k) sin(k r)
               - (4 pi)/r   int_{k_min}^{k_max} dk k^2 P(k) cos(k r)

        We've tested this against Mathematica for the dummy power spectrum, and find agreement
        to 10^-11 for low values of r, where we expect the worst loss of precision to occur.

        :param r: Value of r to use in the integral
        :param suppression: High frequency suppression method to use
        :return: Result of the integral
        """
        if r == 0:
            # Treat the special case
            return 0.0

        pk = self.model.powerspectrum
        min_k = self.model.min_k
        max_k = self.model.max_k
        if suppression == suppression.SAMPLING:
            # No need to go far out into the tail of the suppressing Gaussian
            # At k = n k_0, the suppression is exp(-n^2/2)
            # At n = 6, this is ~10^-8, which seems like a good place to stop
            max_k = min(max_k, self.model.grid.sampling_cutoff * 6)

        def f_sin(k):
            return k * pk(k, suppression)

        def f_cos(k):
            return k * k * pk(k, suppression)
        
        def f(k):
            return k * k * k * pk(k, suppression) * spherical_jn(1, k * r)

        # Choose methodology
        if max_k * r > 5 * pi:
            # Perform the integrations using sin and cos quadrature
            sin_result = quad(f_sin, min_k, max_k, weight='sin', wvar=r,
                              epsrel=self.err_rel, epsabs=self.err_abs, full_output=1, limit=60)
            cos_result = quad(f_cos, min_k, max_k, weight='cos', wvar=r,
                              epsrel=self.err_rel, epsabs=self.err_abs, full_output=1, limit=60)
            # Check for any warnings
            if len(sin_result) == 4:
                print('Warning when integrating D_sin(r) at r =', r)
                print(sin_result[-1])
            if len(cos_result) == 4:
                print('Warning when integrating D_cos(r) at r =', r)
                print(cos_result[-1])
            
            # Construct the result
            result = 4 * pi / r**2 * sin_result[0] - 4 * pi / r * cos_result[0]
        else:
            # Perform the integration using direct quadrature
            int_result = quad(f, min_k, max_k,
                              epsrel=self.err_rel, epsabs=self.err_abs, full_output=1, limit=60)
            # Check for any warnings
            if len(int_result) == 4:
                print('Warning when integrating D(r) at r =', r)
                print(int_result[-1])

            # Construct the result
            result = 4 * pi * int_result[0]

        return result
