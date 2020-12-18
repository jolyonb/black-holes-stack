"""
sampling.py

Computes moments and characteristic lengthscale of the power spectrum.
"""
from __future__ import annotations

import numpy as np
import os
from typing import TYPE_CHECKING

from stack.common import Persistence

if TYPE_CHECKING:
    from stack import Model

class Sampler(Persistence):
    """
    Constructs samples of chi squared fields
    """
    @property
    def filename(self) -> str:
        """Returns the filename for this class"""
        return f'sampler'

    def __init__(self, model: 'Model') -> None:
        """
        Initialize the class.
        
        :param model: Model class
        """
        super().__init__(model)
        
    def load_data(self) -> None:
        """Do nothing"""

    def save_data(self) -> None:
        """Do nothing"""
        
    def compute_data(self) -> None:
        """Do nothing"""

    def generate_sample(self, num: int, bias: float) -> None:
        """Generate sample with the given number and bias"""
        
        # Treat alpha = 1 specially
        # Treat ell = 0, 1, 2 specially
        results = []

        # Construct the filename for the CSV output
        path = os.path.join(self.model.path, 'samples', str(num))
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, 'waterfall_phi_lm.csv')
        
        root_4_pi = np.sqrt(4 * np.pi)

        # Generate the fields, writing results to CSV with columns:
        # alpha, ell, m, phi(0), ..., phi(r_N), phi'(0), ..., phi'(r_N), phi''(0)
        # We now compute Eq. (183) in stats.pdf for Phi00
        Phi00 = np.zeros(self.model.gridpoints + 1)
        Phi00p = np.zeros(self.model.gridpoints + 1)
        hessian = np.zeros((3, 3))
        with open(filename, 'w') as f:
            for alpha in range(1, self.model.n_fields + 1):
                for ell in range(0, self.model.ell_max + 1):
                    for m in range(-ell, ell + 1):
                        # Biasing rules:
                        # alpha = 1
                        # ell = 0: phi(0) is biased to the given bias * sqrt(4 pi)
                        # ell = 1: phi'(0) is biased to 0
                        # ell > 1: no biasing

                        # alpha > 1
                        # ell = 0: phi is biased to 0
                        # ell > 0: no biasing

                        # Construct the phi_ellm mode
                        if alpha == 1 and (ell == 0 or ell == 1):
                            if ell == 0:
                                phi, phip, phipp = self.model.correlations2.generate_sample(ell=0, bias_val=bias * root_4_pi)
                            else:
                                # ell == 1
                                phi, phip, phipp = self.model.correlations2.generate_sample(ell=1, bias_val=0)
                        elif alpha > 1 and ell == 0:
                            phi, phip, phipp = self.model.correlations2.generate_sample(ell=0, bias_val=0)
                        else:
                            phi, phip, phipp = self.model.correlations2.generate_sample(ell=ell)
                        
                        # Write the mode out
                        entries = [alpha, ell, m] + list(phi) + list(phip) + [phipp if phipp else 0]
                        f.write(','.join(map(str, entries)) + '\n')

                        # Construct the contribution to the chi-squared field 00 mode
                        Phi00 += phi * phi
                        Phi00p += phi * phip
                        # Construct the contribution to the hessian
                        # TODO

        Phi00 /= root_4_pi
        Phi00p *= 2 / root_4_pi
