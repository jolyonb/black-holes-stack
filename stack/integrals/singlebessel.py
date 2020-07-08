"""
singlebessel.py

Code to compute integrals of the form
C(r) = 4 pi int_{k_min}^{k_max} dk k^2 P(k) j_0(k r),
D(r) = 4 pi int_{k_min}^{k_max} dk k^3 P(k) j_1(k r),
K1(r) = 4 pi int_{k_min}^{k_max} dk k^4 P(k) j_0(k r)
and
F(r) = 4 pi int_{k_min}^{k_max} dk k^4 P(k) j_2(k r).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from math import pi
from scipy.integrate import quad
from scipy.special import spherical_jn

from stack.common import Suppression
from stack.integrals.common import Integrals

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from stack import Model

# Raise errors on issues rather than printing warnings (except for underflow; we don't care)
np.seterr(all='raise')
np.seterr(under='ignore')

class SingleBessel(Integrals):
    """
    Computes single bessel integrals over the power spectrum.
    """
    filename = 'singlebessel'

    def save_data(self) -> None:
        """This class does not save any data, but does output some results for comparison with Mathematica"""
        # Construct a grid in physical space
        rvals = np.logspace(start=-3,
                            stop=2.5,
                            num=21,
                            endpoint=True)
        # Compute C, D, K1 and F on that grid
        Cvals = np.array([self.compute_C(r, Suppression.RAW) for r in rvals])
        Dvals = np.array([self.compute_D(r, Suppression.RAW) for r in rvals])
        K1vals = np.array([self.compute_K1(r, Suppression.RAW) for r in rvals])
        Fvals = np.array([self.compute_F(r, Suppression.RAW) for r in rvals])
        # Save them to file
        df = pd.DataFrame([rvals, Cvals, Dvals, K1vals, Fvals]).transpose()
        df.columns = ['r', 'C(r)', 'D(r)', 'K1(r)', 'F(r)']
        df.to_csv(self.file_path(self.filename + '.csv'), index=False)

    def compute_C(self, r: float, suppression: Suppression) -> float:
        """
        Computes the integral
        C(r) = 4 pi int_{k_min}^{k_max} dk k^2 P(k) j_0(k r)
        
        :param r: Value of r to use in the integral
        :param suppression: High frequency suppression method to use
        :return: Result of the integral
        """
        moments = self.model.get_moments(suppression)
        suppression_factor = None
        if suppression == Suppression.PEAKS:
            raise ValueError(f'Bad suppression method: {suppression}')
        elif suppression == Suppression.RAW:
            suppression_factor = self.model.grid.sampling_cutoff

        # Treat the special case
        if r == 0:
            return moments.sigma0
        
        pk = self.model.powerspectrum
        min_k = self.model.min_k
        max_k = self.model.max_k
        if suppression == suppression.SAMPLING:
            # No need to go far out into the tail of the suppressing Gaussian
            # At k = n k_0, the suppression is exp(-n^2/2)
            # At n = 6, this is ~10^-8, which seems like a good place to stop
            max_k = min(max_k, self.model.grid.sampling_cutoff * 6)

        # Construct the list of domains
        osc = 2 * pi / r
        domains = self.generate_domains(min_k, max_k, moments.k2peak, osc, 10 * osc, suppression_factor)

        # Define integration functions
        def f(k):
            """Straight function to integrate"""
            return k * k * pk(k, suppression) * spherical_jn(0, k * r)
        low_osc = self.gen_low_osc(f, "C", r)

        def hi_osc(min_k: float, max_k: float) -> float:
            """Compute integrals for highly-oscillatory functions"""
            def f_sin(k):
                """Define function to integrate"""
                return k * pk(k, suppression)

            # Compute the integral using sine-weighted quadrature
            int_result = quad(f_sin, min_k, max_k, weight='sin', wvar=r,
                              epsrel=self.err_rel, epsabs=self.err_abs, full_output=1, limit=70)

            # Check for any warnings
            if len(int_result) == 4 and 'roundoff error is detected' not in int_result[-1]:
                print('Warning when integrating C(r) at r =', r)
                print(int_result[-1])

            return int_result[0] / r
        
        # Define selector function
        def selector(min_k: float, max_k: float) -> Callable:
            """Returns the function to use to perform integration on the given domain"""
            if max_k > 10 * osc:
                return hi_osc
            return low_osc

        # Perform integration
        result = self.perform_integral(domains, selector)
        
        return result


    def compute_D(self, r: float, suppression: Suppression) -> float:
        """
        Computes the integral
        D(r) = 4 pi int_{k_min}^{k_max} dk k^3 P(k) j_1(k r)
        
        Note that j_1(k r) = sin(k r) / (k r)^2 - cos(k r) / (k r)
        
        We integrate the first oscillation using normal quadrature, where cancellation issues are largest.
        We then integrate the rest of the intergral over a variety of domains by computing the sine and cosine integrals
        separately:

        D(r) = 4 pi int_{k_min}^{k_max} dk k^3 P(k) (sin(k r) / (k r)^2 - cos(k r) / (k r))
             =   (4 pi)/r^2 int_{k_min}^{k_max} dk k P(k) sin(k r)
               - (4 pi)/r   int_{k_min}^{k_max} dk k^2 P(k) cos(k r)

        :param r: Value of r to use in the integral
        :param suppression: High frequency suppression method to use
        :return: Result of the integral
        """
        moments = self.model.get_moments(suppression)
        suppression_factor = None
        if suppression == Suppression.PEAKS:
            raise ValueError(f'Bad suppression method: {suppression}')
        elif suppression == Suppression.RAW:
            suppression_factor = self.model.grid.sampling_cutoff

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

        # Construct the list of domains
        oscillation1 = 7.72525183693771    # 1 oscillation of j_1(x)
        oscillation10 = 64.38711959055742  # 10 oscillations of j_1(x)
        osc1 = oscillation1 / r
        osc10 = oscillation10 / r
        domains = self.generate_domains(min_k, max_k, moments.k3peak, osc1, osc10, suppression_factor)

        # Define integration functions
        def f(k):
            """Straight function to integrate"""
            return k * k * k * pk(k, suppression) * spherical_jn(1, k * r)
        low_osc = self.gen_low_osc(f, "D", r)

        def hi_osc(min_k: float, max_k: float) -> float:
            """Compute integrals for highly-oscillatory functions"""
            def f_sin(k):
                return k * pk(k, suppression)

            def f_cos(k):
                return k * k * pk(k, suppression)

            # Perform the integrations using sin and cos quadrature
            sin_result = quad(f_sin, min_k, max_k, weight='sin', wvar=r,
                              epsrel=self.err_rel, epsabs=self.err_abs, full_output=1, limit=70)
            cos_result = quad(f_cos, min_k, max_k, weight='cos', wvar=r,
                              epsrel=self.err_rel, epsabs=self.err_abs, full_output=1, limit=70)

            # Check for any warnings
            if len(sin_result) == 4 and 'roundoff error is detected' not in sin_result[-1]:
                print('Warning when integrating D_sin(r) at r =', r)
                print(sin_result[-1])
            if len(cos_result) == 4 and 'roundoff error is detected' not in cos_result[-1]:
                print('Warning when integrating D_cos(r) at r =', r)
                print(cos_result[-1])

            # Construct the result
            int_result = sin_result[0] / (r*r) - cos_result[0] / r

            return int_result

        # Define selector function
        def selector(min_k: float, max_k: float) -> Callable:
            """Returns the function to use to perform integration on the given domain"""
            if max_k > osc10:
                return hi_osc
            return low_osc

        # Perform integration
        result = self.perform_integral(domains, selector)

        return result

    def compute_K1(self, r: float, suppression: Suppression) -> float:
        """
        Computes the integral
        K_1(r) = 4 pi int_{k_min}^{k_max} dk k^4 P(k) j_0(k r)

        :param r: Value of r to use in the integral
        :param suppression: High frequency suppression method to use
        :return: Result of the integral
        """
        moments = self.model.get_moments(suppression)
        suppression_factor = None
        if suppression == Suppression.PEAKS:
            raise ValueError(f'Bad suppression method: {suppression}')
        elif suppression == Suppression.RAW:
            suppression_factor = self.model.grid.sampling_cutoff
    
        # Treat the special case
        if r == 0:
            return moments.sigma1
    
        pk = self.model.powerspectrum
        min_k = self.model.min_k
        max_k = self.model.max_k
        if suppression == suppression.SAMPLING:
            # No need to go far out into the tail of the suppressing Gaussian
            # At k = n k_0, the suppression is exp(-n^2/2)
            # At n = 6, this is ~10^-8, which seems like a good place to stop
            max_k = min(max_k, self.model.grid.sampling_cutoff * 6)

        # Construct the list of domains
        osc = 2 * pi / r
        domains = self.generate_domains(min_k, max_k, moments.k4peak, osc, 10 * osc, suppression_factor)

        # Define integration functions
        def f(k):
            """Straight function to integrate"""
            return k * k * k * k * pk(k, suppression) * spherical_jn(0, k * r)

        low_osc = self.gen_low_osc(f, "K1", r)

        def hi_osc(min_k: float, max_k: float) -> float:
            """Compute integrals for highly-oscillatory functions"""
    
            def f_sin(k):
                """Define function to integrate"""
                return k * k * k * pk(k, suppression)
    
            # Compute the integral using sine-weighted quadrature
            int_result = quad(f_sin, min_k, max_k, weight='sin', wvar=r,
                              epsrel=self.err_rel, epsabs=self.err_abs, full_output=1, limit=70)
    
            # Check for any warnings
            if len(int_result) == 4 and 'roundoff error is detected' not in int_result[-1]:
                print('Warning when integrating K1(r) at r =', r)
                print(int_result[-1])
    
            return int_result[0] / r

        # Define selector function
        def selector(min_k: float, max_k: float) -> Callable:
            """Returns the function to use to perform integration on the given domain"""
            if max_k > 2 * osc:
                return hi_osc
            return low_osc

        # Perform integration
        result = self.perform_integral(domains, selector)

        return result

    def compute_F(self, r: float, suppression: Suppression) -> float:
        """
        Computes the integral
        F(r) = 4 pi int_{k_min}^{k_max} dk k^4 P(k) j_2(k r)

        Note that j_2(k r) = 3 * sin(k r) / (k r)^3 - 3 * cos(k r) / (k r)^2 - sin(k r) / (k r)

        We integrate the first oscillation using normal quadrature, where cancellation issues are largest.
        We then integrate the rest of the intergral over a variety of domains by computing the sine and cosine integrals
        separately.

        :param r: Value of r to use in the integral
        :param suppression: High frequency suppression method to use
        :return: Result of the integral
        """
        moments = self.model.get_moments(suppression)
        suppression_factor = None
        if suppression == Suppression.PEAKS:
            raise ValueError(f'Bad suppression method: {suppression}')
        elif suppression == Suppression.RAW:
            suppression_factor = self.model.grid.sampling_cutoff
    
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

        # Construct the list of domains
        halfoscillation = 5.76345919689455  # Half oscillation of j_2(x)
        oscillation1 = 9.09501133047638     # 1 oscillation of j_2(x)
        oscillation10 = 65.92794150295865   # 10 oscillations of j_2(x)
        halfosc = halfoscillation / r
        osc1 = oscillation1 / r
        osc10 = oscillation10 / r
        domains = self.generate_domains(min_k, max_k, moments.k4peak, osc1, osc10, suppression_factor)

        # Define integration functions
        def f(k):
            """Straight function to integrate"""
            return k * k * k * k * pk(k, suppression) * spherical_jn(2, k * r)
        low_osc = self.gen_low_osc(f, "F", r)

        def hi_osc(min_k: float, max_k: float) -> float:
            """Compute integrals for highly-oscillatory functions"""
            def f_sin1(k):
                return k * pk(k, suppression)

            def f_cos2(k):
                return k * k * pk(k, suppression)

            def f_sin3(k):
                return k * k * k * pk(k, suppression)

            # Perform the integrations using sin and cos quadrature
            sin1_result = quad(f_sin1, min_k, max_k, weight='sin', wvar=r,
                               epsrel=self.err_rel, epsabs=self.err_abs, full_output=1, limit=70)
            cos2_result = quad(f_cos2, min_k, max_k, weight='cos', wvar=r,
                               epsrel=self.err_rel, epsabs=self.err_abs, full_output=1, limit=70)
            sin3_result = quad(f_sin3, min_k, max_k, weight='sin', wvar=r,
                               epsrel=self.err_rel, epsabs=self.err_abs, full_output=1, limit=70)
            # Check for any warnings
            if len(sin1_result) == 4 and 'roundoff error is detected' not in sin1_result[-1]:
                print('Warning when integrating F_sin1(r) at r =', r)
                print(sin1_result[-1])
            if len(cos2_result) == 4 and 'roundoff error is detected' not in cos2_result[-1]:
                print('Warning when integrating F_cos2(r) at r =', r)
                print(cos2_result[-1])
            if len(sin3_result) == 4 and 'roundoff error is detected' not in sin3_result[-1]:
                print('Warning when integrating F_sin3(r) at r =', r)
                print(sin3_result[-1])

            # Construct the result
            int_result = 3 * sin1_result[0] / (r * r * r) - 3 * cos2_result[0] / (r * r) - sin3_result[0] / r

            return int_result

        # Define selector function
        def selector(min_k: float, max_k: float) -> Callable:
            """Returns the function to use to perform integration on the given domain"""
            if max_k > halfosc:
                return hi_osc
            return low_osc

        # Perform integration
        result = self.perform_integral(domains, selector)

        return result
