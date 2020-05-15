"""
Quick and dirty test suite for the full stack.
"""
from stack import Model

def main():
    model = Model(model_name='real_ps', n_efolds=15, n_fields=4, mpsi=0.1, m0=10.0, verbose=True)
    model.construct_powerspectrum()
    model.construct_moments()
    model.construct_singlebessel()
    model.construct_grid()

if __name__ == '__main__':
    main()