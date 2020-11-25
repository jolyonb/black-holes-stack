"""
Quick and dirty test suite for the full stack (uses a toy analytic power spectrum).
"""
from stack import Model

import numpy as np

def load_model(model_name, method, scaling):
    model = Model(model_name=model_name, n_efolds=15, n_fields=4, mpsi=0.1, m0=10.0, verbose=True,
                  test_ps=True, ell_max=2, num_k_points=15001, method=method, scaling=scaling)
    model.construct_powerspectrum()
    model.construct_moments()
    model.construct_singlebessel()
    model.construct_doublebessel()
    model.construct_grid()
    model.construct_moments2()
    model.construct_correlations()
    model.construct_correlations2()
    # model.construct_moments3()
    # model.construct_peakdensity()
    
    return model

def main():
    model_lin_trap = load_model('toy_ps', 'trapezoid', 'linear')
    model_lin_simp = load_model('toy_ps_simp', 'simpson', 'linear')
    model_log_trap = load_model('toy_ps_log', 'trapezoid', 'log')
    model_log_simp = load_model('toy_ps_simp_log', 'simpson', 'log')

    ell = 0
    cov = model_lin_trap.correlations.covariance[ell][1:, 1:]
    errs = model_lin_trap.correlations.covariance_errs[ell][1:, 1:]

    if ell == 2:
        start = 1
        end = 11
    else:
        start = 2
        end = 12
    cov_lin_trap = model_lin_trap.correlations2.covariance[ell][start:end, start:end] / 4 / np.pi
    cov_lin_simp = model_lin_simp.correlations2.covariance[ell][start:end, start:end] / 4 / np.pi
    cov_log_trap = model_log_trap.correlations2.covariance[ell][start:end, start:end] / 4 / np.pi
    cov_log_simp = model_log_simp.correlations2.covariance[ell][start:end, start:end] / 4 / np.pi

    import sys
    sys.exit()
    
    from math import pi
    fourpi = 4 * pi
    row0 = model.correlations2.covariance[0][0]
    sigma02 = row0[0] / fourpi
    sigma12 = - 3 * row0[1] / fourpi
    Cr = row0[2:12] / fourpi
    Dr = - row0[12:22] / fourpi
    row1 = model.correlations2.covariance[0][1]
    K1r = - 3 * row1[2:12] / fourpi
    row2 = model.correlations2.covariance[2][0]
    Fr = row2[1:11] * 15 / 2 / fourpi
    othersigma02 = model.moments_sampling.sigma0squared
    othersigma12 = model.moments_sampling.sigma1squared
    otherCr = model.correlations.C[1:]
    otherDr = model.correlations.D[1:]
    otherK1r = model.correlations.K1[1:]
    otherFr = model.correlations.F[1:]
    
    # Compare biased covariance matrices
    cov0 = model.correlations2.covariance[0][2:12, 2:12]
    bcov0 = model.correlations2.biased_covariance_0[2:12, 2:12]
    othercov0 = model.correlations.covariance[0][1:, 1:] * fourpi
    
    test0 = cov0 / othercov0
    
    cov1 = model.correlations2.covariance[1][1:11, 1:11]
    bcov1 = model.correlations2.biased_covariance_1[1:11, 1:11]
    othercov1 = model.correlations.covariance[1][1:, 1:] * fourpi
    
    test1 = cov1 / othercov1

    # Comparison of biased covariance matrix computation methods
    # Should have unit ratio
    test0 = (cov0 - fourpi * np.outer(Cr, Cr) / sigma02) / bcov0
    test1 = (cov1 - fourpi * np.outer(Dr, Dr) / sigma12) / bcov1

    # Play with number of gripoints in k, integration method, compare with what's happening at different ell values
    # Optimize!
    
    # Means should be:
    mean = Cr / sigma02
    computed_means = model.correlations2.biased_sampling_0[2:12, 0] / np.sqrt(fourpi * sigma02)  # denominator from chi_0 = nu / sqrt(fourpi * sigma02)

    # Still worried about factors of 1/4pi from Y_00...
    
    print('here')
    
    # Construct the A matrix
    from stack.common import Suppression
    from scipy.special import spherical_jn
    from numpy.linalg import svd
    max_k = model.powerspectrum.max_k
    min_k = model.powerspectrum.min_k
    max_k = min(max_k, model.grid.sampling_cutoff * 6)    # Move this into the power spectrum class

    numpoints = 20001   # Must be odd to support Simpson's rule
    integrator = 'trapezoid'
    method = 'log'
    ell = 2

    w_vec = np.ones(numpoints) * 2
    w_vec[0] = 1
    w_vec[-1] = 1
    if integrator == 'trapezoid':
        # 1 2 2 2 2 2 1
        w_vec /= 2
    elif integrator == 'simpson':
        # 1 4 2 4 2 4 1
        assert numpoints % 2 == 1
        w_vec[1::2] = 4
        w_vec /= 3
    else:
        raise ValueError()

    if method == 'linear':
        k_grid = np.linspace(min_k, max_k, numpoints, endpoint=True)
        w_vec *= k_grid[1] - k_grid[0]
    elif method == 'log':
        k_grid = np.logspace(np.log10(min_k), np.log10(max_k), numpoints, endpoint=True)
        w_vec *= (np.log(k_grid[1]) - np.log(k_grid[0])) * k_grid
    else:
        raise ValueError()

    p_grid = np.array([model.powerspectrum(k, Suppression.SAMPLING) for k in k_grid])
    r_grid = model.grid.grid

    # integral = np.dot(w_vec, test_vec)

    An_coeff = 4 * np.pi * k_grid * np.sqrt(w_vec * p_grid)

    # M = discretization of integral
    # N = discretization of space
    M = len(k_grid)
    N = len(r_grid)

    A = np.array([[An_coeff[m] * spherical_jn(ell, k_grid[m] * r) for m in range(M)] for r in r_grid])
    u, s, vh = svd(A, full_matrices=False, compute_uv=True, hermitian=False)
    v = vh.transpose()

    # Test the dot product of every vector with every other vector
    orthonormality_check = np.dot(vh, v)

    # Construct C_ij as the matrix product of A (NxM) and v (MxN), contracting on the M indices.
    C_ij = np.matmul(A, v)
    # First index of C is i, and comes from A
    # Second index of X is j, and comes from v

    cov = np.matmul(C_ij, C_ij.transpose())

    othercov = model.correlations.covariance[ell]

    ratio = cov[1:, 1:] / othercov[1:, 1:] / 4 / np.pi - 1

    test = cov / 4 / np.pi - othercov

    output = test[1:, 1:] / model.correlations.covariance_errs[ell][1:, 1:]

    print('here')

    # AAT = np.matmul(A, A.transpose())
    # Agree, up to about 13 digits at worst (mostly 16 digits)

    print('done')
    
    
    # Check the corner integral here - do we still have catastropic cancellation occurring?



    # TODO:
    # * Test covariance prior to SVD (~13 digits at worst (cancellations), ~16 digits typical)

    # * Estimate number of points needed
    # Delta k in integral to be ~2 pi / (10(r+r')) = 1 / (pi rmax) to resolve fastest oscillating mode in integral
    # At rmax = 63, this gives Delta k = 5e-3
    # 200,000 points was running at Delta k = 6e-5, almost 100 times smaller than this!
    # 2,000 points does terribly
    # 20,000 points increases accuracy by about 4 digits
    # 10,000 points is also comparable
    # max_k is 12. With Deltak = 5e-3, this gives ~2,400 points. 3k is definitely not enough
    # Need enough points to capture all oscillations of the fastest oscillating mode, given by rmax.
    # 15k seems a good number. deltak = 8e-4, or about 60 trapezoid slices for a cosine.
    # Delta k should depend on rmax, by Delta K = 2 pi / (60 * 2rmax)

    # * Test Simpson's rule instead of trapezoid
    # Still need about 15k points to get same performance of trapezoid. Nothing special here.

    # * Discretization - log k space
    # linear wins, hands down. Significantly better - 500k points had 10^-10 accuracy, compared to 10^-14 at 15k in linear
    # Need discretization over the oscillation lengthscale.

    # * Corners
    # * Package things up
    # * Implement derivative covariance
    # * Biasing
    
    # Implement pipeline version of covariance computation
    # Implement biasing
    # Implement derivatives
    # Implement switches - grid points, trapezoid/simpson, log/linear



if __name__ == '__main__':
    main()
