"""
common.py

Contains helper code for computing integrals
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from stack.common import Persistence

from typing import TYPE_CHECKING, List, Tuple, Callable, Optional

if TYPE_CHECKING:
    from stack import Model

# Raise errors on issues rather than printing warnings (except for underflow; we don't care)
np.seterr(all='raise')
np.seterr(under='ignore')

class Integrals(Persistence):
    """
    Base class for computing integrals
    """
    
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
    
    @staticmethod
    def generate_domains(min_k: float, max_k: float, peak: float,
                         osc1: float, osc10: float, suppress: Optional[float]) -> List[Tuple[float, float]]:
        """
        Generates a list of domain tuples (start_k, stop_k) that span the domain (min_k, max_k), with splits
        based on the following parameters:

        :param min_k: k value to start integration at
        :param max_k: k value to stop integration at
        :param peak: Peak value of integral
        :param osc1: First oscillation k value
        :param osc10: 10th oscillation k value
        :param suppress: Suppression scale (or None if not using)
        :return: List of (min, max) tuples
        """
        possible_points = [min_k, max_k, osc1, 2*osc1, osc10/3, 2/3*osc10, osc10, 1.5*osc10, 2*osc10, 5*osc10, 5*peak, 10*peak,
                           0.2 * max_k, 0.4 * max_k, 0.5 * max_k, 0.6 * max_k, 0.7 * max_k, 0.9 * max_k, 0.8 * max_k]
        if suppress is not None:
            # Make sure the suppression factor doesn't drop too quickly over the domain
            # by inserting suppression scale domain endpoints
            # End of domain is 6*suppress, which has exp(-18) ~ 1.5e-8
            possible_points += [n * suppress for n in range(1, 6)]
        points = sorted([p for p in possible_points if min_k <= p <= max_k])
        tuples = [(points[i], points[i+1]) for i in range(len(points) - 1)]
        return tuples
    
    @staticmethod
    def perform_integral(domains: List[Tuple[float, float]], selector: Callable) -> float:
        """
        Performs an integral over a list of domains, using a selector function to determine which integration
        method to use.

        :param domains: List of domain tuples (min, max)
        :param selector: Function that returns another function corresponding to the integration method
        :return: Result of the integral
        """
        # Start by integrating each domain
        results = []
        errors = []
        for mink, maxk in domains:
            integrator = selector(mink, maxk)
            res, err = integrator(mink, maxk)
            results.append(res)
            errors.append(err)

        # Now compare errors to the result, to see if we need to go and recompute with combined domains
        results = np.array(results)
        errors = np.array(errors)
        result = np.sum(results)
        err_est = np.sum(errors)
        rel_err = abs(err_est / result)
        
        # Compare the biggest result to the final result to determine the cancellation error
        cancel_err = np.max(np.abs(results)) / np.abs(result)
        
        if cancel_err > 1e5:
            # Try combining domains for results that were too big, and rerunning
            threshold = np.abs(1e5 * result)
            bad_domains = []
            # Flag all results that were bigger than the threshold, making sure we get at least two domains to combine
            while len(bad_domains) < 2:
                threshold /= 10
                bad_domains = np.argwhere(np.abs(results) > threshold)
            first = int(bad_domains[0])
            last = int(bad_domains[-1])
            # Need to bring together all domains that were bad, in order for them to cancel
            mink = domains[first][0]
            maxk = domains[last][1]
            integrator = selector(mink, maxk)
            res, err = integrator(mink, maxk)
            # Combine results again
            results2 = np.concatenate([results[0:first], np.array([res]), results[last+1:]])
            errors2 = np.concatenate([errors[0:first], np.array([err]), errors[last+1:]])
            result = np.sum(results)
            err_est = np.sum(errors2)
            cancel_err = np.max(np.abs(results2)) / np.abs(result)
            rel_err = abs(err_est / result)
            print(f'    result: {result}')
            print(f'    abs: {err_est}')
            print(f'    rel: {rel_err}')
            print(f'    cancel: {cancel_err}')

        return 4 * np.pi * result

    def gen_low_osc(self, f: Callable, name: str, rval: float) -> Callable:
        """Generates a function that computes an integral of f using normal quadrature"""
        def func(min_k: float, max_k: float) -> Tuple[float, float]:
            # Compute the integral
            int_result = quad(f, min_k, max_k,
                              epsrel=self.err_rel, epsabs=self.err_abs, full_output=1, limit=70)
    
            # Check for any warnings
            if len(int_result) == 4 and 'roundoff error is detected' not in int_result[-1]:
                print(f'Warning when integrating {name}(r) at r = {rval}')
                print(int_result[-1])
    
            # print(f"(* low_osc *)")
            # print(f'mink={min_k}; maxk={max_k}; ans={int_result[0]};'.replace("e", "*^"))

            return int_result[0], int_result[1]
  
        return func
