"""
Quick and dirty test suite for the full stack.
"""
from stack import Model

def main():
    model = Model(model_name='toy_ps', n_efolds=15, n_fields=4, mpsi=0.1, m0=10.0, verbose=True,
                  test_ps=True, max_k=20, min_k=0, num_modes=401)
    model.construct_powerspectrum()
    # model.construct_moments()
    # model.construct_singlebessel()
    # model.construct_grid()
    # model.construct_moments2()
    # model.construct_correlations()
    
if __name__ == '__main__':
    main()
