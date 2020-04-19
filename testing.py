"""
Quick and dirty test suite for the full stack.
"""
from stack import Model

def main():
    model = Model(model_name='testmodel', n_efolds=15, n_fields=4, mpsi=1, m0=1, verbose=True)
    model.construct_powerspectrum()

if __name__ == '__main__':
    main()
