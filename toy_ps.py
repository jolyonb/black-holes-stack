"""
Quick and dirty test suite for the full stack (uses a toy analytic power spectrum).
"""
from stack import Model

def main():
    model = Model(model_name='toy_ps', n_efolds=15, n_fields=4, mpsi=0.1, m0=10.0, verbose=True,
                  test_ps=True, ell_max=4)
    model.construct_powerspectrum()
    model.construct_moments()
    model.construct_singlebessel()
    model.construct_doublebessel()
    model.construct_grid()
    model.construct_moments2()
    model.construct_correlations()
    model.construct_correlations2()
    model.construct_moments3()
    # model.construct_peakdensity()
    model.construct_sampler()
    
    model.sampler.generate_sample(1, 1)
    
    
    # TO DO LIST
    # * Check k=0 contributions work in logarithmic correlation computations (same answers as linear)
    # * Fix hessian implementation to account for alpha index
    # * Check covariance matrices agree for real P(k)
    # * Implement linear evolution
    # * Create LaTeX guide to code and equations implemented
    # * Save output of generated samples
    # * Send Alan cov matrix elements we've checked
    # * Add in Alan's power spectrum and test
    

if __name__ == '__main__':
    main()
