from stack.peakdensity.peakdensity import signed_exact, number_density

import numpy as np

# Some test numbers that are giving me problems
n_fields = 4
sigma0 = 1.893113063873135
sigma1 = 6.975159671838396
gamma = 0.1458203796129053

samples = 1e5

nu_vals = np.linspace(0, 6, 50)

# Peak densities
min_vec = np.zeros_like(nu_vals)
saddleppm_vec = np.zeros_like(nu_vals)
saddlepmm_vec = np.zeros_like(nu_vals)
max_vec = np.zeros_like(nu_vals)
# Error estimates
min_err_vec = np.zeros_like(nu_vals)
saddleppm_err_vec = np.zeros_like(nu_vals)
saddlepmm_err_vec = np.zeros_like(nu_vals)
max_err_vec = np.zeros_like(nu_vals)
# Analytic results
signed_vec = np.zeros_like(nu_vals)

def assign_value(idx, values, err, signedval):
    min_vec[idx] = values[1]
    saddleppm_vec[idx] = values[2]
    saddlepmm_vec[idx] = values[3]
    max_vec[idx] = values[4]
    min_err_vec[idx] = err[1]
    saddleppm_err_vec[idx] = err[2]
    saddlepmm_err_vec[idx] = err[3]
    max_err_vec[idx] = err[4]
    signed_vec[idx] = signedval

for idx, nu in enumerate(nu_vals):
    print(f'    Computing {idx + 1}/{len(nu_vals)} at nu={nu}')
    if nu == 0:
        exact = signed_exact(n_fields, 0.0, sigma0, sigma1)
        values = np.array([0, exact, 0, 0, 0])
        err = np.array([0, 0, 0, 0, 0])
        assign_value(idx, values, err, exact)
    else:
        # Invoke the vegas routines
        values, err = number_density(int(n_fields), gamma, nu, sigma0, sigma1, int(samples))
        signedval = signed_exact(n_fields, nu, sigma0, sigma1)
        assign_value(idx, values, err, signedval)

if __name__ == '__main__':
    pass
