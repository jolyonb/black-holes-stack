"""
correlations2.py

Contains the Correlations2 class, which computes sampling and covariance matrices on the sampling grid.

Here is what we need for each ell value:

ell = 0:
phi(0) (need to be able to bias)
phi''(0)
phi(r)
phi'(r)

ell = 1:
phi'(0) (need to be able to bias)
phi(r)
phi'(r)

ell = 2:
phi''(0)
phi(r)
phi'(r)

ell >= 3:
phi(r)
phi'(r)

All missing quantities are identically vanishing. We need covariances for all pairs of the above (at each ell).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from math import pi
from scipy.special import spherical_jn
from numpy.linalg import svd

from stack.common import Persistence, Suppression

if TYPE_CHECKING:
    from stack import Model


class Correlations2(Persistence):
    """
    Constructs correlations on the physical grid.
    """
    filename = 'correlations2'

    def __init__(self, model: 'Model') -> None:
        """
        Initialize the class.

        :param model: Model class we are computing integrals for.
        """
        super().__init__(model)
        self.covariance = None
        self.sampling = None
        self.biased_covariance_0 = None
        self.biased_covariance_1 = None
        self.biased_sampling_0 = None
        self.biased_sampling_1 = None
        self.sigma02 = None
        self.sigma12 = None
        self.Cr = None
        self.Dr = None
        self.K1r = None
        self.Fr = None

    def load_data(self) -> None:
        """Loads saved values from file"""
        # Read covariance matrices
        filename = self.filename + '_matrices.npy'
        path = self.file_path(filename)
        if not self.file_exists(filename):
            raise FileNotFoundError(f'Unable to load from {path}')

        self.covariance = [None] * (self.model.ell_max + 1)
        self.sampling = [None] * (self.model.ell_max + 1)

        with open(path, 'rb') as f:
            for ell in range(self.model.ell_max + 1):
                self.covariance[ell] = np.load(f)
                self.sampling[ell] = np.load(f)
            self.biased_sampling_0 = np.load(f)
            self.biased_sampling_1 = np.load(f)
            self.biased_covariance_0 = np.load(f)
            self.biased_covariance_1 = np.load(f)
        
        self.extract_values()

    def compute_data(self) -> None:
        """Constructs correlations on the radial grid"""
        # Grab some constants we'll need
        r_grid = self.model.grid.grid
        numpoints = self.model.num_k_points
        method = self.model.method
        scaling = self.model.scaling
        
        # Construct some things we'll need
        min_k = self.model.powerspectrum.min_k
        max_k = self.model.powerspectrum.max_k
        max_k = min(max_k, self.model.grid.sampling_cutoff * 6)  # TODO: Move this into the power spectrum class

        # Set up the integration method
        w_vec = np.ones(numpoints) * 2
        w_vec[0] = 1
        w_vec[-1] = 1
        if method == 'trapezoid':
            # 1 2 2 2 2 2 1
            w_vec /= 2
        elif method == 'simpson':
            # 1 4 2 4 2 4 1
            assert numpoints % 2 == 1
            w_vec[1::2] = 4
            w_vec /= 3
        else:
            raise ValueError()

        # Set up the domain
        if scaling == 'linear':
            k_grid = np.linspace(min_k, max_k, numpoints, endpoint=True)
            w_vec *= k_grid[1] - k_grid[0]
        elif scaling == 'log':
            if self.model.include_0:
                # Adjust k value for the logarithmic grid construction.
                min_k = self.model.min_k
            k_grid = np.geomspace(min_k, max_k, numpoints, endpoint=True)
            w_vec *= np.log(max_k / min_k) / (numpoints - 1) * k_grid
            if self.model.include_0:
                # Add in a trapezoid rule contribution from k=0 and k=min_k.
                # Note that the k=0 point doesn't contribute, as all integrals have powers of k.
                # Hence, just modify the min_k coefficient.
                w_vec[0] += min_k / 2
        else:
            raise ValueError()

        # Evaluate power spectrum
        pk_grid = np.array([self.model.powerspectrum(k, Suppression.SAMPLING) for k in k_grid])

        # Compute A matrix. Each row is a value of alpha.
        self.covariance = [None] * (self.model.ell_max + 1)
        self.sampling = [None] * (self.model.ell_max + 1)
        for ell in range(0, self.model.ell_max + 1):
            stack = []
            # Construct the coefficient that depends only on n (vector across all n values)
            An_coeff = 4 * pi * k_grid * np.sqrt(w_vec * pk_grid)

            # Add in special values for ell = 0
            if ell == 0:
                # Add in phi(0) row
                row = An_coeff
                stack.append(row)
                # Add in phi''(0) row
                row = An_coeff * k_grid * k_grid * ((-1) / 3)  # j_0''(0) = -1/3
                stack.append(row)
            elif ell == 1:
                # Add in phi'(0) row
                row = An_coeff * k_grid / 3  # j_1'(0) = 1/3
                stack.append(row)
            elif ell == 2:
                # Add in phi''(0) row
                row = An_coeff * k_grid * k_grid * (2 / 15)  # j_2''(0) = 2/15
                stack.append(row)

            # Add in phi(r) rows for r > 0
            stack += [An_coeff * spherical_jn(ell, k_grid * r) for r in r_grid[1:]]
            
            # Add in phi'(r) rows
            stack += [An_coeff * k_grid * spherical_jn(ell, k_grid * r, derivative=True) for r in r_grid[1:]]
            
            # Number of variables to sample is K
            K = len(stack)
            
            # We have now constructed the A matrix
            # This is the sampling matrix (in k space)
            A = np.array(stack)

            # Perform a Singular Value Decomposition of A
            u, s, vh = svd(A, full_matrices=False, compute_uv=True, hermitian=False)

            # First index of A is length M
            # Second index of A is length K
            # Shape of A is (K, M): K-1 rows, M columns

            # vh has the vectors we want: its shape is K, M. Transpose it to get the basis vectors we want.
            v = vh.transpose()

            # Construct C_ij as the matrix product of A (KxM) and v (MxK), contracting on the M indices.
            # This is the sampling matrix (in position space)
            C_ij = np.matmul(A, v)
            # First index of C is i, and comes from A
            # Second index of C is j, and comes from v
            
            # Construct covariance matrix
            cov = np.matmul(C_ij, C_ij.transpose())

            # Store the results
            self.covariance[ell] = cov
            self.sampling[ell] = C_ij
            
            # If we have ell = 0 or ell = 1, construct a biased sampling matrix
            if ell == 0 or ell == 1:
                # The approach here is the same for ell = 0 or ell = 1
                # We want to make sure that the first vector is proportional to the first column of A
                vec1 = A[0].copy()
                vec1 /= np.sqrt(np.dot(vec1, vec1))
                
                # Compute Abar, where other columns of A are orthogonal to vec1
                Abar = A.copy()
                for i in range(1, K):
                    Abar[i] -= np.dot(vec1, Abar[i]) * vec1
                    
                # Do the SVD decomposition of the remaining columns of Abar
                u, s, vh = svd(Abar[1:], full_matrices=False, compute_uv=True, hermitian=False)
    
                # First index of Abar is length M
                # Second index of Abar is length K-1
                # Shape of Abar is (K-1, M): K-1 rows, M columns
    
                # vh has the vectors we want: its shape is K-1, M. Transpose it to get the basis vectors we want.
                v = vh.transpose()
                
                # Add in vec1
                v = np.append(np.array([vec1]).transpose(), v, axis=1)
    
                # Construct C_ij as the matrix product of Abar (KxM) and v (MxK), contracting on the M indices.
                # This is the sampling matrix (in position space)
                C_ij = np.matmul(A, v)
                # First index of C is i, and comes from Abar
                # Second index of C is j, and comes from v

                # Construct covariance matrix
                cov = np.matmul(C_ij, C_ij.transpose())
                
                # Store results
                if ell == 0:
                    self.biased_sampling_0 = C_ij
                    self.biased_covariance_0 = cov
                else:
                    # ell = 1
                    self.biased_sampling_1 = C_ij
                    self.biased_covariance_1 = cov

        self.extract_values()

    def extract_values(self):
        """Extracts useful values from the covariance matrices"""
        fourpi = 4 * np.pi
        row0 = self.covariance[0][0]
        row1 = self.covariance[0][1]
        row2 = self.covariance[2][0]
        self.sigma02 = row0[0] / fourpi
        self.sigma12 = - 3 * row0[1] / fourpi
        self.Cr = row0[2:12] / fourpi
        self.Dr = - row0[12:22] / fourpi
        self.K1r = - 3 * row1[2:12] / fourpi
        self.Fr = row2[1:11] * 15 / 2 / fourpi

    def save_data(self) -> None:
        """Save precomputed values to file"""
        # Save covariance and sampling matrices
        with open(self.file_path(self.filename + '_matrices.npy'), 'wb') as f:
            for ell in range(self.model.ell_max + 1):
                np.save(f, self.covariance[ell])
                np.save(f, self.sampling[ell])
            np.save(f, self.biased_sampling_0)
            np.save(f, self.biased_sampling_1)
            np.save(f, self.biased_covariance_0)
            np.save(f, self.biased_covariance_1)
        
        # Save a CSV version of everything for human readability
        for ell in range(self.model.ell_max + 1):
            np.savetxt(self.file_path(self.filename + f'_matrices_cov_{ell}.csv'), self.covariance[ell], delimiter=',')
            np.savetxt(self.file_path(self.filename + f'_matrices_sample_{ell}.csv'), self.sampling[ell], delimiter=',')
        np.savetxt(self.file_path(self.filename + '_matrices_bias_sample_0.csv'), self.biased_sampling_0, delimiter=',')
        np.savetxt(self.file_path(self.filename + '_matrices_bias_sample_1.csv'), self.biased_sampling_1, delimiter=',')
        np.savetxt(self.file_path(self.filename + '_matrices_bias_cov_0.csv'), self.biased_covariance_0, delimiter=',')
        np.savetxt(self.file_path(self.filename + '_matrices_bias_cov_1.csv'), self.biased_covariance_1, delimiter=',')

    def generate_sample(self, ell: int, bias_val: Optional[float] = None):
        """
        Generates a sample for the given ell value and bias value. Does not bias if bias_val is not provided.
        
        Returns two numpy arrays and a float:
        * (phi(0), phi(r_1), ..., phi(r_N)) - field values on grid
        * (phi'(0), phi'(r_1), ..., phi'(r_N)) - field derivatives on grid
        * phi''(0) or None - field second derivative at origin (ell = 0 or 2 only), or None (ell = 1 or >= 3)
        """
        if ell > self.model.ell_max:
            raise ValueError(f'Bad ell value: {ell} (ell_max = {self.model.ell_max})')

        # Select the sampling matrix
        if ell == 0 and bias_val is not None:
            sampler = self.biased_sampling_0
        elif ell == 1 and bias_val is not None:
            sampler = self.biased_sampling_1
        else:
            sampler = self.sampling[ell]
        
        # Construct a vector of unit variance Gaussian random numbers
        random_vec = np.random.normal(size=sampler.shape[0])
        
        # If biasing, we need to fix the first value
        if bias_val is not None and (ell == 0 or ell == 1):
            random_vec[0] = bias_val / sampler[0, 0]
            
        # Perform the matrix multiply to construct samples
        samples = np.matmul(sampler, random_vec)

        # Split the samples into vectors appropriately
        N = len(self.model.grid.grid) - 1  # Number of grid points not at 0
        if ell == 0:
            phi0 = samples[0]
            phip0 = 0
            phipp0 = samples[1]
            phir = samples[2:N+2]
            phipr = samples[N+2:]
        elif ell == 1:
            phi0 = 0
            phip0 = samples[0]
            phipp0 = None
            phir = samples[1:N+1]
            phipr = samples[N+1:]
        elif ell == 2:
            phi0 = 0
            phip0 = 0
            phipp0 = samples[0]
            phir = samples[1:N+1]
            phipr = samples[N+1:]
        else:
            phi0 = 0
            phip0 = 0
            phipp0 = None
            phir = samples[0:N]
            phipr = samples[N:]

        phi = np.concatenate(([phi0], phir))
        phip = np.concatenate(([phip0], phipr))
        
        return phi, phip, phipp0
