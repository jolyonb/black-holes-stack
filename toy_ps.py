"""
Quick and dirty test suite for the full stack (uses a toy analytic power spectrum).
"""
from stack import Model
from stack.common import Suppression

def main():
    model = Model(model_name='toy_ps', n_efolds=15, n_fields=4, mpsi=0.1, m0=10.0, verbose=True,
                  test_ps=True)
    model.construct_powerspectrum()
    model.construct_moments()
    model.construct_singlebessel()
    model.construct_doublebessel()
    # model.construct_grid()
    # model.construct_moments2()
    # model.construct_correlations()
    # model.construct_moments3()
    # model.construct_peakdensity()
    
    db = model.doublebessel
    print(db.compute_G(ell=25, r=90, rp=200, suppression=Suppression.RAW))

if __name__ == '__main__':
    main()
