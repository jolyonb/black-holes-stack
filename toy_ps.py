"""
Quick and dirty test suite for the full stack (uses a toy analytic power spectrum).
"""
from stack import Model

def main():
    model = Model(model_name='toy_ps', n_efolds=15, n_fields=4, mpsi=0.1, m0=10.0, verbose=True,
                  test_ps=True)
    model.construct_powerspectrum()
    model.construct_moments()
    model.construct_singlebessel()
    model.construct_doublebessel()
    model.construct_grid()
    model.construct_moments2()
    model.construct_correlations()
    # model.construct_moments3()
    # model.construct_peakdensity()
    
    # Construct the A matrix
    import numpy as np
    from stack.common import Suppression
    from scipy.special import spherical_jn
    from numpy.linalg import svd
    max_k = model.powerspectrum.max_k
    min_k = model.powerspectrum.min_k
    max_k = min(max_k, model.grid.sampling_cutoff * 6)    # Move this into the power spectrum class
    numpoints = 15000
    
    method = 'linear'
    if method == 'linear':
        k_grid = np.linspace(min_k, max_k, numpoints, endpoint=True)
        w_vec = np.ones_like(k_grid) * 2
        w_vec[0] = 1
        w_vec[-1] = 1
        w_vec *= (k_grid[1] - k_grid[0]) / 2

        s_vec = np.ones_like(k_grid)
        s_vec[::2] = 0
        s_vec += 1
        s_vec *= 2
        s_vec[0] = 1
        s_vec[-1] = 1
        s_vec *= (k_grid[1] - k_grid[0]) / 3

    elif method == 'log':
        k_grid = np.logspace(np.log10(min_k), np.log10(max_k), numpoints, endpoint=True)

        diffs = np.diff(k_grid)
        adiff = np.concatenate(([0], diffs))
        bdiff = np.concatenate((diffs, [0]))
        w_vec = (adiff + bdiff) / 2
    else:
        raise ValueError()

    p_grid = np.array([model.powerspectrum(k, Suppression.SAMPLING) for k in k_grid])
    r_grid = model.grid.grid

    integrator = 'trapezoid'
    if integrator == 'trapezoid':
        usew = w_vec
    elif integrator == 'simpsons':
        usew = s_vec
    else:
        raise ValueError()


    # integral = np.dot(w_vec, test_vec)


    An_coeff = 4 * np.pi * k_grid * np.sqrt(w_vec * p_grid)
    
    # M = discretization of integral
    # N = discretization of space
    M = len(k_grid)
    N = len(r_grid)
    ell = 2
    
    A = np.array([[An_coeff[m] * spherical_jn(ell, k_grid[m] * r) for m in range(M)] for r in r_grid])

    u, s, vh = svd(A, full_matrices=False, compute_uv=True, hermitian=False)
   
    # First index of A is length M
    # Second index of A is length N
    # Shape of A is (N, M) - N rows, M columns

    # vh has the vectors we want: its shape is N, M. Transpose it to get the basis vectors we want.
    v = vh.transpose()
    
    # Test the dot product of every vector with every other vector
    orthonormality_check = np.dot(vh, v)
    
    # Construct C_ij as the matrix product of A (NxM) and v (MxN), contracting on the M indices.
    C_ij = np.matmul(A, v)
    # First index of C is i, and comes from A
    # Second index of X is j, and comes from v
    
    cov = np.matmul(C_ij, C_ij.transpose())
    
    othercov = model.correlations.covariance[ell]

    # For some reason, cov is larger than our covariance by a factor of 4 pi.
    # This appears to come from Alan's Eq 1, where he has 16 pi^2 (but our cov matrices only use 4 pi)
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



if __name__ == '__main__':
    main()
