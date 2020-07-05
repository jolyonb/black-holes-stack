"""
singlebessel.py

Code to compute integrals of the form
C(r) = 4 pi int_{k_min}^{k_max} dk k^2 P(k) j_0(k r)
and
D(r) = 4 pi int_{k_min}^{k_max} dk k^3 P(k) j_1(k r).
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

# Raise errors on issues rather than printing warnings (except for underflow; we don't care)
np.seterr(all='raise')
np.seterr(under='ignore')

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
        self.err_rel = 1e-9

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
        if suppression == suppression.PEAKS:
            raise ValueError(f'Bad suppression method: {suppression}')

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

        def f_sin(k):
            return k*pk(k, suppression)

        def f(k):
            return k*k*pk(k, suppression)*spherical_jn(0, k*r)

        # Integrate first oscillation using normal quadrature
        k1 = min(2 * pi / r, max_k)
        int_result = quad(f, min_k, k1,
                          epsrel=self.err_rel, epsabs=self.err_abs, full_output=1, limit=70)
        # print(f'{{{min_k}, {k1}, {int_result[0]}}}')
        # Check for any warnings
        if len(int_result) == 4 and 'roundoff error is detected' not in int_result[-1]:
            print('Warning when integrating C(r) (Step 1) at r =', r)
            print(int_result[-1])
        # Construct result
        result = 4 * pi * int_result[0]
        if k1 == max_k:
            return result

        # Rest of the integral will be handled using sine-weighted quadrature
        # Start by setting up the integration ranges
        endpoints = [10 * k1, moments.k2peak * 5, moments.k2peak * 50, moments.k2peak * 500]
        using_endpoints = [k1]
        # Select the points we want to use
        for endpoint in endpoints:
            if endpoint > max_k:
                break
            if endpoint > using_endpoints[-1]:
                using_endpoints.append(endpoint)
        using_endpoints.append(max_k)
        
        # Perform integration using sine-weighted quadrature
        for idx in range(0, len(using_endpoints) - 1):
            int_result = quad(f_sin, using_endpoints[idx], using_endpoints[idx+1], weight='sin', wvar=r,
                              epsrel=self.err_rel, epsabs=self.err_abs, full_output=1, limit=70)
            # print(f'{{{using_endpoints[idx]}, {using_endpoints[idx+1]}, {int_result[0] / r}}}')
            # Check for any warnings
            if len(int_result) == 4 and 'roundoff error is detected' not in int_result[-1]:
                print('Warning when integrating C(r) at r =', r)
                print(int_result[-1])
            # Construct result
            result += 4 * pi * int_result[0] / r

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
        if suppression == suppression.PEAKS:
            raise ValueError(f'Bad suppression method: {suppression}')

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

        # Integrate first peak using normal quadrature
        oscillation1 = 4.49341  # Half oscillation of j_1(x)
        oscillation10 = 64.3871  # 10 oscillations of j_1(x)
        k1 = min(oscillation1 / r, max_k)
        int_result = quad(f, min_k, k1,
                          epsrel=self.err_rel, epsabs=self.err_abs, full_output=1, limit=70)
        # print(f'{{{min_k}, {k1}, {int_result[0]}}}'.replace('e', '*^'))
        # Check for any warnings
        if len(int_result) == 4 and 'roundoff error is detected' not in int_result[-1]:
            print('Warning when integrating D(r) at r =', r)
            print(int_result[-1])
        # Construct result
        result = 4 * pi * int_result[0]
        if k1 == max_k:
            return result

        # Rest of the integral will be handled using sine-weighted quadrature
        # Start by setting up the integration ranges
        endpoints = [oscillation10 / r, moments.k3peak * 5, moments.k3peak * 25, moments.k3peak * 50, moments.k3peak * 75, moments.k3peak * 100,
                     moments.k3peak * 100, moments.k3peak * 150, moments.k3peak * 200,
                     moments.k3peak * 250, moments.k3peak * 300, moments.k3peak * 350,
                     moments.k3peak * 400, moments.k3peak * 450]
        using_endpoints = [k1]
        # Select the points we want to use
        for endpoint in endpoints:
            if endpoint > max_k:
                break
            if endpoint > using_endpoints[-1]:
                using_endpoints.append(endpoint)
        using_endpoints.append(max_k)

        # Perform integration using sine-weighted quadrature
        for idx in range(0, len(using_endpoints) - 1):
            # Perform the integrations using sin and cos quadrature
            sin_result = quad(f_sin, using_endpoints[idx], using_endpoints[idx + 1], weight='sin', wvar=r,
                              epsrel=self.err_rel, epsabs=self.err_abs, full_output=1, limit=70)
            cos_result = quad(f_cos, using_endpoints[idx], using_endpoints[idx + 1], weight='cos', wvar=r,
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
            # print(f'{{{using_endpoints[idx]}, {using_endpoints[idx+1]}, {int_result}}}'.replace('e', '*^'))
            result += 4 * pi * int_result

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
        if suppression == suppression.PEAKS:
            raise ValueError(f'Bad suppression method: {suppression}')
    
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
    
        def f_sin(k):
            return k * k * k * pk(k, suppression)
    
        def f(k):
            return k * k * k * k * pk(k, suppression) * spherical_jn(0, k * r)
    
        # Integrate first oscillation using normal quadrature
        k1 = min(2 * pi / r, max_k)
        int_result = quad(f, min_k, k1,
                          epsrel=self.err_rel, epsabs=self.err_abs, full_output=1, limit=70)
        # print(f'{{{min_k}, {k1}, {int_result[0]}}}')
        # Check for any warnings
        if len(int_result) == 4 and 'roundoff error is detected' not in int_result[-1]:
            print('Warning when integrating K1(r) (Step 1) at r =', r)
            print(int_result[-1])
        # Construct result
        result = 4 * pi * int_result[0]
        if k1 == max_k:
            return result
    
        # Rest of the integral will be handled using sine-weighted quadrature
        # Start by setting up the integration ranges
        endpoints = [10 * k1, moments.k4peak * 5, moments.k4peak * 50, moments.k4peak * 500]
        using_endpoints = [k1]
        # Select the points we want to use
        for endpoint in endpoints:
            if endpoint > max_k:
                break
            if endpoint > using_endpoints[-1]:
                using_endpoints.append(endpoint)
        using_endpoints.append(max_k)
    
        # Perform integration using sine-weighted quadrature
        for idx in range(0, len(using_endpoints) - 1):
            int_result = quad(f_sin, using_endpoints[idx], using_endpoints[idx + 1], weight='sin', wvar=r,
                              epsrel=self.err_rel, epsabs=self.err_abs, full_output=1, limit=70)
            # print(f'{{{using_endpoints[idx]}, {using_endpoints[idx+1]}, {int_result[0] / r}}}')
            # Check for any warnings
            if len(int_result) == 4 and 'roundoff error is detected' not in int_result[-1]:
                print('Warning when integrating K_1(r) at r =', r)
                print(int_result[-1])
            # Construct result
            result += 4 * pi * int_result[0] / r
    
        return result

    def compute_F(self, r: float, suppression: Suppression) -> float:
        """
        Computes the integral
        D(r) = 4 pi int_{k_min}^{k_max} dk k^4 P(k) j_2(k r)

        Note that j_2(k r) = 3 * sin(k r) / (k r)^3 - 3 * cos(k r) / (k r)^2 - sin(k r) / (k r)

        We integrate the first oscillation using normal quadrature, where cancellation issues are largest.
        We then integrate the rest of the intergral over a variety of domains by computing the sine and cosine integrals
        separately.

        :param r: Value of r to use in the integral
        :param suppression: High frequency suppression method to use
        :return: Result of the integral
        """
        moments = self.model.get_moments(suppression)
        if suppression == suppression.PEAKS:
            raise ValueError(f'Bad suppression method: {suppression}')
    
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
    
        def f_sin1(k):
            return k * pk(k, suppression)
    
        def f_cos2(k):
            return k * k * pk(k, suppression)

        def f_sin3(k):
            return k * k * k * pk(k, suppression)

        def f(k):
            return k * k * k * k * pk(k, suppression) * spherical_jn(2, k * r)
    
        # Integrate first peak using normal quadrature
        oscillation1 = 5.76345919689455  # Half oscillation of j_2(x)
        oscillation10 = 65.92794150295865  # 10 oscillations of j_2(x)
        k1 = min(oscillation1 / r, max_k)
        int_result = quad(f, min_k, k1,
                          epsrel=self.err_rel, epsabs=self.err_abs, full_output=1, limit=70)
        # print(f'{{{min_k}, {k1}, {int_result[0]}}}'.replace('e', '*^'))
        # Check for any warnings
        if len(int_result) == 4 and 'roundoff error is detected' not in int_result[-1]:
            print('Warning when integrating F(r) at r =', r)
            print(int_result[-1])
        # Construct result
        result = 4 * pi * int_result[0]
        if k1 == max_k:
            return result
    
        # Rest of the integral will be handled using sine-weighted quadrature
        # Start by setting up the integration ranges
        endpoints = [oscillation10 / r, moments.k4peak * 5, moments.k4peak * 25, moments.k4peak * 50,
                     moments.k4peak * 75, moments.k4peak * 100,
                     moments.k4peak * 100, moments.k4peak * 150, moments.k4peak * 200,
                     moments.k4peak * 250, moments.k4peak * 300, moments.k4peak * 350,
                     moments.k4peak * 400, moments.k4peak * 450]
        using_endpoints = [k1]
        # Select the points we want to use
        for endpoint in endpoints:
            if endpoint > max_k:
                break
            if endpoint > using_endpoints[-1]:
                using_endpoints.append(endpoint)
        using_endpoints.append(max_k)
    
        # Perform integration using weighted quadrature
        for idx in range(0, len(using_endpoints) - 1):
            # Perform the integrations using sin and cos quadrature
            sin1_result = quad(f_sin1, using_endpoints[idx], using_endpoints[idx + 1], weight='sin', wvar=r,
                               epsrel=self.err_rel, epsabs=self.err_abs, full_output=1, limit=70)
            cos2_result = quad(f_cos2, using_endpoints[idx], using_endpoints[idx + 1], weight='cos', wvar=r,
                               epsrel=self.err_rel, epsabs=self.err_abs, full_output=1, limit=70)
            sin3_result = quad(f_sin3, using_endpoints[idx], using_endpoints[idx + 1], weight='sin', wvar=r,
                               epsrel=self.err_rel, epsabs=self.err_abs, full_output=1, limit=70)
            # Check for any warnings
            if len(sin1_result) == 4 and 'roundoff error is detected' not in sin1_result[-1]:
                print('Warning when integrating F_sin1(r) at r =', r)
                print(sin1_result[-1])
            if len(cos2_result) == 4 and 'roundoff error is detected' not in cos2_result[-1]:
                print('Warning when integrating F_cos2(r) at r =', r)
                print(cos2_result[-1])
            if len(sin3_result) == 4 and 'roundoff error is detected' not in sin3_result[-1]:
                print('Warning when integrating F_sin1(r) at r =', r)
                print(sin3_result[-1])

            # Construct the result
            int_result = 3 * sin1_result[0] / (r * r * r) - 3 * cos2_result[0] / (r * r) - sin3_result[0] / r
            # print(f'{{{using_endpoints[idx]}, {using_endpoints[idx+1]}, {int_result}}}'.replace('e', '*^'))
            result += 4 * pi * int_result
    
        return result
