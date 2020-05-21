"""
Test the quality of P(k) interpolation.

Compare results with 'Interpolation Comparison.nb'.
"""
import numpy as np
import pandas as pd

from stack import Model
from stack.common import Suppression

def main():
    model = Model(model_name='test_ps1', n_efolds=15, n_fields=4, mpsi=0.1, m0=10.0, verbose=True, num_modes=401)
    model.construct_powerspectrum()

    model2 = Model(model_name='test_ps2', n_efolds=15, n_fields=4, mpsi=0.1, m0=10.0, verbose=True, num_modes=1601)
    model2.construct_powerspectrum()
    
    kvals = model2.powerspectrum.kvals
    real = model2.powerspectrum.spectrum
    interpolated = np.array([model.powerspectrum(k, Suppression.RAW) for k in kvals])

    df = pd.DataFrame([kvals, real, interpolated]).transpose()
    df.columns = ['k', 'P(k)', 'Interpolation']
    df.to_csv('models/test_ps1/pk_interptest.csv', index=False)

if __name__ == '__main__':
    main()
