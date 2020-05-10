"""
Quick and dirty test suite for the full stack.
"""
from stack import Model

def main():
    model = Model(model_name='testmodel', n_efolds=15, n_fields=4, mpsi=0.1, m0=10.0, verbose=True,
                  test_ps=True, max_k=5, min_k=0)
    model.construct_powerspectrum()
    model.construct_moments()
    model.construct_singlebessel()
    
    # Output singlebessel integrals for comparison to mathematica (copy/paste!)
    # for r in range(1, 10):
    #     print(f'{model.singlebessel.compute_C(r)},')
    # for r in range(1, 10):
    #     print(f'{model.singlebessel.compute_D(r/10)},')

if __name__ == '__main__':
    main()
