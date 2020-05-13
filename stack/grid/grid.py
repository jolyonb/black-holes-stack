"""
grid.py

Contains the Grid class, which constructs a radial grid in physical space.
"""

# -- finished --
# Use rho(r) = C(r) / sigma_0^2
# Compute on a reasonably fine grid, start at characteristic scale / 100, go to characteristic scale * 100
# Use persistence to output to file for easy plotting
# Make a MMA script that loads and plots
# Find the point at which rho(r) has fallen to 0.5 (estimate)
# This is the peak FWHM estimate (very rough, does the job)
# Use num_gridpoints and max_r_factor from model parameters to construct a grid
# r_max = FWHM * max_r_factor
# linear gridpoints at r_max / (num_gridpoints - 1) spacing (start at 0, go to r_max)

# -- unfinished --
# Save these parameters in the persistence data
# Persist the grid (save/load)
# Estimate ell_max (persist this too!) using Eq 174.

# Bonus points: separate PR to construct persistence model parameters nicely

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from stack.common import Persistence
# from stack.integrals import singlebessel
# from stack.moments import moments

if TYPE_CHECKING:
    from stack import Model


class Grid(Persistence):
    """
    Constructs a radial grid in physical space.
    """
    filename = 'grid'

    def __init__(self, model: 'Model') -> None:
        """
        Initialize the class.

        :param model: Model class we are computing integrals for.
        """

        super().__init__(model)

    def load_data(self) -> None:
        """This class does not save any data, so has nothing to load"""
        pass

    def compute_data(self) -> None:
        """This class does not save any data, so has nothing to compute"""
        pass

    def save_data(self) -> None:
        """This class does not save any data, but does output some results for comparison with Mathematica"""
        sb = self.model.singlebessel
        mom = self.model.moments

        # Construct a grid in physical space
        rstart = mom.compute_lengthscale() / 10
        rend = 10 * mom.compute_lengthscale()
        step = rstart/100
        rvals = np.arange(rstart, rend, step)

        # Compute rhoC on this grid
        rhoCvals = [(sb.compute_C(item) / mom.compute_sigma_n_squared(0)) for item in rvals]

        # characteristic length-scale
        characteristic = mom.compute_lengthscale()
        print("characteristic length-scale:", characteristic)

        # find FWHM radius
        res = list(map(lambda i: i < 0.5, rhoCvals)).index(True)
        FWHM = rvals[res]
        print("The FWHM occurs at r =", FWHM)

        # How to grab rmax factor and numgridpoints?
        rmaxfactor = 10
        num_gridpoints = 100
        r_max = rmaxfactor * FWHM
        grid = np.arange(0, r_max, r_max / (num_gridpoints - 1) )

        # l_max estimate
        l_max = r_max * (2 * np.pi / FWHM)

        # Save them to file
        df = pd.DataFrame([rvals, rhoCvals]).transpose()
        df.columns = ['r', 'rhoC(r)']

        df.to_csv(self.file_path(self.filename + '.csv'), index=False)

    def computeRhoC(self, r: float) -> float:
        sb = self.model.singlebessel
        mom = self.model.moments

        result = (sb.compute_C(r) / mom.compute_sigma_n_squared(0))

        return result