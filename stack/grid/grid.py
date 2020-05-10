"""
grid.py

Contains the Grid class, which construts a radial grid in physical space.
"""

# Use rho(r) = C(r) / sigma_0^2
# Compute on a reasonably fine grid, start at characteristic scale / 100, go to characteristic scale * 100
# Use persistence to output to file for easy plotting
# Make a MMA script that loads and plots
# Find the point at which rho(r) has fallen to 0.5 (estimate)
# This is the peak FWHM estimate (very rough, does the job)
# Use num_gridpoints and max_r_factor from model parameters to construct a grid
# r_max = FWRM * max_r_factor
# linear gridpoints at r_max / num_gridpoints spacing (start at 0, go to r_max)
# Save these parameters in the persistence data
# Persist the grid (save/load)

# Bonus points: separate PR to construct persistence model parameters nicely
