"""
correlations.py

Contains the Correlations class, which stores the correlation functions C(r), D(r), K_1(r) and F(r) on the sampling grid.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from math import pi

from stack.common import Persistence, Suppression

if TYPE_CHECKING:
    from stack import Model


class Correlations(Persistence):
    """
    Constructs correlations on the physical grid.
    """
    filename = 'correlations'

    def __init__(self, model: 'Model') -> None:
        """
        Initialize the class.

        :param model: Model class we are computing integrals for.
        """
        super().__init__(model)
        self.C = None
        self.D = None
        self.K1 = None
        self.F = None
        self.rhoC = None
        self.rhoD = None

    def load_data(self) -> None:
        """Loads saved values from file"""
        filename = self.filename + '.csv'
        path = self.file_path(filename)
        if not self.file_exists(filename):
            raise FileNotFoundError(f'Unable to load from {path}')

        df = pd.read_csv(path)

        self.C = df['C(r)'].values
        self.D = df['D(r)'].values
        self.K1 = df['K1(r)'].values
        self.F = df['F(r)'].values
        self.rhoC = df['rhoC(r)'].values
        self.rhoD = df['rhoD(r)'].values

    def compute_data(self) -> None:
        """Constructs the radial grid"""
        sb = self.model.singlebessel
        mom = self.model.moments_sampling
        grid = self.model.grid.grid

        # Compute C(r), D(r), K1(r), F(r), rhoC(r) and rhoD(r) on the radial grid
        self.C = np.array([sb.compute_C(r, Suppression.SAMPLING) for r in grid])
        self.D = np.array([sb.compute_D(r, Suppression.SAMPLING) for r in grid])
        self.K1 = np.array([sb.compute_K1(r, Suppression.SAMPLING) for r in grid])
        self.F = np.array([sb.compute_F(r, Suppression.SAMPLING) for r in grid])
        self.rhoC = self.C / mom.sigma0squared
        self.rhoD = self.D * np.sqrt(3 / mom.sigma0squared / mom.sigma1squared)

    def save_data(self) -> None:
        """Save precomputed values to file"""
        df = pd.DataFrame([self.C, self.D, self.K1, self.F, self.rhoC, self.rhoD]).transpose()
        df.columns = ['C(r)', 'D(r)', 'K1(r)', 'F(r)', 'rhoC(r)', 'rhoD(r)']
        df.to_csv(self.file_path(self.filename + '.csv'), index=False)
