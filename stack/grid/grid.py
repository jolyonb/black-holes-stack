"""
grid.py

Contains the Grid class, which constructs a radial grid in physical space.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from stack.common import Persistence

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
        self.grid = None
        self.C = None
        self.D = None
        self.rhoC = None
        self.rhoD = None

    @property
    def rmax(self):
        """Returns the maximum radius in the grid"""
        return self.grid[-1]

    @property
    def gridpoints(self):
        """Returns the number of gridpoints in the grid"""
        return self.model.gridpoints

    def load_data(self) -> None:
        """Loads the grid, C(r), D(r) and rhoC(r) from file"""
        filename = self.filename + '.csv'
        path = self.file_path(filename)
        if not self.file_exists(filename):
            raise FileNotFoundError(f'Unable to load from {path}')

        df = pd.read_csv(path)

        self.grid = df['r'].values
        self.C = df['C(r)'].values
        self.D = df['D(r)'].values
        self.rhoC = df['rhoC(r)'].values
        self.rhoD = df['rhoD(r)'].values

    def compute_data(self) -> None:
        """Constructs the radial grid"""
        sb = self.model.singlebessel
        mom = self.model.moments

        # Construct a test grid in physical space, using the characteristic lengthscale as a yardstick
        rstart = mom.lengthscale / 10
        rend = 8 * mom.lengthscale
        step = rstart / 10
        rvals = np.arange(rstart, rend, step)

        # Compute rhoC on this grid
        rhoCvals = np.array([sb.compute_C(r) for r in rvals]) / mom.sigma0squared

        # Find FWHM radius
        FWHM = rvals[np.argmax(rhoCvals < 0.5)]

        # Construct the real radial grid, using the FWHM as a yardstick
        self.grid = np.linspace(0, FWHM * self.model.rmaxfactor, self.gridpoints + 1, endpoint=True)
        
        # Compute C(r), D(r) and rhoC(r) on the radial grid
        self.C = np.array([sb.compute_C(r) for r in self.grid])
        self.D = np.array([sb.compute_D(r) for r in self.grid])
        self.rhoC = self.C / mom.sigma0squared
        self.rhoD = self.D * np.sqrt(3 / mom.sigma0squared / mom.sigma1squared)

    def save_data(self) -> None:
        """Save precomputed values to file"""
        df = pd.DataFrame([self.grid, self.C, self.D, self.rhoC, self.rhoD]).transpose()
        df.columns = ['r', 'C(r)', 'D(r)', 'rhoC(r)', 'rhoD(r)']
        df.to_csv(self.file_path(self.filename + '.csv'), index=False)
