"""
grid.py

Contains the Grid class, which constructs a radial grid in physical space.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from math import pi

from stack.common import Persistence, Suppression

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
        self.FWHM = None
        self.sampling_cutoff = None

    @property
    def rmax(self):
        """Returns the maximum radius in the grid"""
        return self.grid[-1]

    @property
    def gridpoints(self):
        """Returns the number of gridpoints in the grid"""
        return self.model.gridpoints

    def load_data(self) -> None:
        """Loads saved values from file"""
        # First file
        filename = self.filename + '.csv'
        path = self.file_path(filename)
        if not self.file_exists(filename):
            raise FileNotFoundError(f'Unable to load from {path}')

        df = pd.read_csv(path)

        self.grid = df['r'].values

        # Second file
        filename = self.filename + '2.csv'
        path = self.file_path(filename)
        if not self.file_exists(filename):
            raise FileNotFoundError(f'Unable to load from {path}')
    
        df = pd.read_csv(path)
    
        self.FWHM = df['FWHM'].values[0]
        self.sampling_cutoff = df['sampling_cutoff'].values[0]

    def compute_data(self) -> None:
        """Constructs the radial grid"""
        sb = self.model.singlebessel
        mom = self.model.moments_raw

        # Construct a test grid in physical space, using the characteristic lengthscale as a yardstick
        rstart = mom.lengthscale / 10
        rend = 5 * mom.lengthscale
        step = rstart / 2
        rvals = np.arange(rstart, rend, step)

        # Compute rhoC on this grid
        rhoCvals = np.array([sb.compute_C(r, Suppression.RAW) for r in rvals]) / mom.sigma0squared

        # Find FWHM radius
        self.FWHM = rvals[np.argmax(rhoCvals < 0.5)]

        # Construct the real radial grid, using the FWHM as a yardstick
        self.grid = np.linspace(0, self.FWHM * self.model.rmaxfactor, self.gridpoints + 1, endpoint=True)
        
        # Construct the sampling cutoff in k space based on the grid size
        gridspacing = self.grid[1]
        wavelength = gridspacing / 2
        self.sampling_cutoff = 2 * pi / wavelength * self.model.sampling_cutoff_factor
        
    def save_data(self) -> None:
        """Save precomputed values to file"""
        df = pd.DataFrame([self.grid]).transpose()
        df.columns = ['r']
        df.to_csv(self.file_path(self.filename + '.csv'), index=False)

        df = pd.DataFrame([[self.FWHM, self.sampling_cutoff]])
        df.columns = ['FWHM', 'sampling_cutoff']
        df.to_csv(self.file_path(self.filename + '2.csv'), index=False)
