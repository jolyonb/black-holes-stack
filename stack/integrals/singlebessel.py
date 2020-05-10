"""
singlebessel.py

Code to compute integrals of the form
int_{k_min}^{k_max} dk k^2 P(k) j_l(k r)
where l = 0 or 1.
"""
from __future__ import annotations

import numpy as np
from math import pi
from scipy.integrate import quad

from stack.common import Persistence

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stack import Model

# Raise errors on issues rather than printing warnings
np.seterr(all='raise')


class SingleBessel(Persistence):
    """
    Computes single bessel integrals over the power spectrum.
    """
    filename = None

    def __init__(self, model: 'Model') -> None:
        """
        Initialize the class.
        
        :param model: Model class we are computing integrals for.
        """
        super().__init__(model)
        
        # Error tolerances used in computing integrals
        self.err_abs = 0
        self.err_rel = 1e-10

    def load_data(self) -> None:
        """This class does not save any data, so has nothing to load"""
        pass

    def compute_data(self) -> None:
        """This class does not save any data, so has nothing to compute"""
        pass

    def save_data(self) -> None:
        """This class does not save any data"""
        pass

    def compute_C(self, r: float) -> float:
        """
        Computes the integral
        C(r) = 4 pi int_{k_min}^{k_max} dk k^2 P(k) j_0(k r)
        
        We've tested this against Mathematica for the dummy power spectrum, and find agreement
        to 10^-14 (relative).
        
        :param r: Value of r to use in the integral
        :return: Result of the integral
        """
        if r == 0:
            # Treat the special case
            return self.model.moments.sigma0squared
        
        pk = self.model.powerspectrum

        def f(k):
            return k*pk(k)

        # Perform the integration
        result, err = quad(f, self.model.min_k, self.model.max_k, weight='sin', wvar=r,
                           epsrel=self.err_rel, epsabs=self.err_abs)
        
        # Rescale the result
        result *= 4 * pi / r

        return result

    def compute_D(self, r: float) -> float:
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
        :return: Result of the integral
        """
        if r == 0:
            # Treat the special case
            return 0.0

        pk = self.model.powerspectrum

        def f_sin(k):
            return k * pk(k)

        def f_cos(k):
            return k * k * pk(k)

        # Perform the integrations
        sin_result, sin_err = quad(f_sin, self.model.min_k, self.model.max_k, weight='sin', wvar=r)
        cos_result, cos_err = quad(f_cos, self.model.min_k, self.model.max_k, weight='cos', wvar=r)
        # Warning: this may lead to catastrophic cancellation of precision at VERY small values of r.
        
        # Construct the result
        result = 4 * pi / r**2 * sin_result - 4 * pi / r * cos_result

        return result
