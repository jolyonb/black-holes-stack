"""
peaks_test.py

Script for testing number density of peaks calculations.
"""
from stack import Model
from stack.peakdensity import PeakDensity

if __name__ == '__main__':
    gamma = 0.6
    sigma0 = 1
    sigma1 = 1
    outfile = 'peaks_test.csv'
    
    model = Model(model_name='peaks_test', n_fields=4, peakdensity_samples=int(1e5), nu_steps=50, verbose=True)
    peaks = PeakDensity(model)
    
    peaks._compute_data(gamma, sigma0, sigma1)
    peaks._save_data(outfile)
