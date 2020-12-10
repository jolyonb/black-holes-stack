"""
Quick and dirty test suite for the full stack (uses a toy analytic power spectrum).
"""
from stack import Model

def load_model(model_name, method, scaling, recalculate):
    model = Model(model_name=model_name, n_efolds=15, n_fields=4, mpsi=0.1, m0=10.0, verbose=True,
                  test_ps=True, ell_max=4, num_k_points=20001, method=method, scaling=scaling)
    model.construct_powerspectrum()
    model.construct_moments()
    model.construct_singlebessel()
    model.construct_doublebessel()
    model.construct_grid()
    model.construct_moments2()
    model.construct_correlations()
    model.construct_correlations2(recalculate)
    # model.construct_moments3()
    # model.construct_peakdensity()
    
    return model

def main():
    model_log_simp = load_model('toy_ps_simp_log', 'simpson', 'log', False)

    ell = 1
    bias = 100
    phi, phip, phipp = model_log_simp.correlations2.generate_sample(ell=ell, bias_val=bias)
    
    print(phi)
    print(phip)
    print(phipp)

if __name__ == '__main__':
    main()
