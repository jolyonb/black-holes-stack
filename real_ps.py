"""
Quick and dirty test suite for the full stack (computes a realistic power spectrum).
"""
from stack import Model
from stack.common import Suppression

def main():
    model = Model(model_name='real_ps', n_efolds=15, n_fields=4, mpsi=0.1, m0=10.0, verbose=True)
    model.construct_powerspectrum()
    model.construct_moments()
    model.construct_singlebessel()
    model.construct_doublebessel()
    # model.construct_grid()
    # model.construct_moments2()
    # model.construct_correlations()
    # model.construct_moments3()
    # model.construct_peakdensity()
    
    ellvals = [7, 12, 20, 25, 30]
    rvals = [0.01, 10, 50, 100, 250]
    for ell in ellvals:
        for idx1 in range(0, len(rvals)):
            for idx2 in range(idx1 + 1, len(rvals)):
                r = rvals[idx1]
                r2 = rvals[idx2]
                print(f"Working on {ell}, {r}, {r2}...")
                model.doublebessel.compute_G(ell, r, r2, Suppression.RAW)

if __name__ == '__main__':
    main()
