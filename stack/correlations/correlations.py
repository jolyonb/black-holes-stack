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
        self.dC = None
        self.dD = None
        self.dK1 = None
        self.dF = None
        self.rhoC = None
        self.rhoD = None
        self.covariance = None
        self.covariance_errs = None

    def load_data(self) -> None:
        """Loads saved values from file"""
        # Read single Bessel integrals
        filename = self.filename + '.csv'
        path = self.file_path(filename)
        if not self.file_exists(filename):
            raise FileNotFoundError(f'Unable to load from {path}')

        df = pd.read_csv(path)

        self.C = df['C(r)'].values
        self.D = df['D(r)'].values
        self.K1 = df['K1(r)'].values
        self.F = df['F(r)'].values
        self.dC = df['dC(r)'].values
        self.dD = df['dD(r)'].values
        self.dK1 = df['dK1(r)'].values
        self.dF = df['dF(r)'].values
        self.rhoC = df['rhoC(r)'].values
        self.rhoD = df['rhoD(r)'].values

        # Read covariance matrices
        filename = self.filename + '_matrices.npy'
        path = self.file_path(filename)
        if not self.file_exists(filename):
            raise FileNotFoundError(f'Unable to load from {path}')

        self.covariance = [None] * (self.model.ell_max + 1)
        self.covariance_errs = [None] * (self.model.ell_max + 1)

        with open(path, 'rb') as f:
            for ell in range(self.model.ell_max + 1):
                self.covariance[ell] = np.load(f)
                self.covariance_errs[ell] = np.load(f)

    def compute_data(self) -> None:
        """Constructs correlations on the radial grid"""
        sb = self.model.singlebessel
        db = self.model.doublebessel
        mom = self.model.moments_sampling
        grid = self.model.grid.grid

        # Compute C(r), D(r), K1(r), F(r), rhoC(r) and rhoD(r) on the radial grid
        print('    Computing C integrals...')
        self.C = np.array([sb.compute_C(r, Suppression.SAMPLING) for r in grid])
        print('    Computing D integrals...')
        self.D = np.array([sb.compute_D(r, Suppression.SAMPLING) for r in grid])
        print('    Computing K1 integrals...')
        self.K1 = np.array([sb.compute_K1(r, Suppression.SAMPLING) for r in grid])
        print('    Computing F integrals...')
        self.F = np.array([sb.compute_F(r, Suppression.SAMPLING) for r in grid])
        # Everything is currently a 2-column array of values, errors. Split these out.
        self.dC = self.C[:, 1]
        self.dD = self.D[:, 1]
        self.dK1 = self.K1[:, 1]
        self.dF = self.F[:, 1]
        self.C = self.C[:, 0]
        self.D = self.D[:, 0]
        self.K1 = self.K1[:, 0]
        self.F = self.F[:, 0]
        # Compute the correlation functions
        self.rhoC = self.C / mom.sigma0squared
        self.rhoD = self.D * np.sqrt(3 / mom.sigma0squared / mom.sigma1squared)
        
        # Now compute the full covariance matrices for each ell.
        # For each ell from 0 to ell_max, compute covariance matrix <phi(r1), phi(r2)>
        self.covariance = [None] * (self.model.ell_max + 1)
        self.covariance_errs = [None] * (self.model.ell_max + 1)
        for ell in range(0, self.model.ell_max + 1):
            # Compute E integrals first (r1 = r2)
            print(f'    Computing covariance matrices for ell = {ell}...')
            print(f'        Computing E integrals...')
            Evals = np.array([db.compute_E(ell, r, Suppression.SAMPLING) for r in grid])
            Eerrs = Evals[:, 1]
            Evals = Evals[:, 0]

            # Next, compute G integrals (r1 < r2)
            print(f'        Computing G integrals...')
            covariance = np.zeros((len(grid), len(grid)))
            errors = np.zeros((len(grid), len(grid)))
            for idx1, r1 in enumerate(grid):
                for idx2, r2 in enumerate(grid):
                    if not r1 < r2:
                        continue
                    print(f"        {idx1 + 1}, {idx2 + 1} / {len(grid)}")
                    result, err = db.compute_G(ell, r1, r2, Suppression.SAMPLING)
                    covariance[idx1, idx2] = result
                    errors[idx1, idx2] = err

            # Construct the full covariance matrix
            covariance = covariance + np.transpose(covariance) + np.diag(Evals)
            errors = errors + np.transpose(errors) + np.diag(Eerrs)
            
            # Store the results
            self.covariance[ell] = covariance
            self.covariance_errs[ell] = errors

    def save_data(self) -> None:
        """Save precomputed values to file"""
        # Save single Bessel integrals
        df = pd.DataFrame([self.C, self.D, self.K1, self.F,
                           self.dC, self.dD, self.dK1, self.dF,
                           self.rhoC, self.rhoD]).transpose()
        df.columns = ['C(r)', 'D(r)', 'K1(r)', 'F(r)', 'dC(r)', 'dD(r)', 'dK1(r)', 'dF(r)', 'rhoC(r)', 'rhoD(r)']
        df.to_csv(self.file_path(self.filename + '.csv'), index=False)
        
        # Save covariance matrices
        with open(self.file_path(self.filename + '_matrices.npy'), 'wb') as f:
            for ell in range(self.model.ell_max + 1):
                np.save(f, self.covariance[ell])
                np.save(f, self.covariance_errs[ell])

        # Save a CSV version of everything for human readability
        for ell in range(self.model.ell_max + 1):
            np.savetxt(self.file_path(self.filename + f'_matrices_cov_{ell}.csv'), self.covariance[ell], delimiter=',')
            np.savetxt(self.file_path(self.filename + f'_matrices_cov_errs_{ell}.csv'), self.covariance_errs[ell], delimiter=',')
