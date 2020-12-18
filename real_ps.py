"""
Quick and dirty test suite for the full stack (computes a realistic power spectrum).
"""
from stack import Model

def main():
    model = Model(model_name='real_ps', n_efolds=15, n_fields=4, mpsi=0.1, m0=10.0, verbose=True, ell_max=4)

    model.construct_powerspectrum()
    model.construct_moments()
    model.construct_singlebessel()
    model.construct_doublebessel()
    model.construct_grid()
    model.construct_moments2()
    model.construct_correlations()
    model.construct_correlations2()
    model.construct_moments3()
    model.construct_peakdensity()
    model.construct_sampler()

    model.sampler.generate_sample(1, 1)


if __name__ == '__main__':
    main()
