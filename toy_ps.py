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
    model.construct_grid()
    model.construct_moments2()
    model.construct_correlations()
    model.construct_moments3()
    model.construct_peakdensity()
    
    

if __name__ == '__main__':
    main()
