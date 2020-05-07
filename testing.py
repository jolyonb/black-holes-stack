"""
Quick and dirty test suite for the full stack.
"""
from math import sqrt

from stack import Model

def main():
    model = Model(model_name='testmodel', n_efolds=15, n_fields=4, mpsi=0.1, m0=sqrt(10.0), verbose=True)
    model.construct_powerspectrum(recalculate=True)
    model.construct_moments()

if __name__ == '__main__':
    main()
