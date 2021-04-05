# Inputs to all bin_analysis files, including binfit_new.py,
# marg_new.py, compare_spectra.py. 

[DATA]
planet = l9859b
visit_number = visit02
scan_direction = both
transit = yes

# These dictate which residuals to use, but don't impact bin fitting at all.
# In MODEL section, limb_type will impact both.

# First: Use white light fit that:
ignored_exposures = yes
openar = no
inflated_errors = yes

# Second: Even when ignored in the white light fit, include
# in the spectral fit?
include_first_orbit = no
include_removed_points = no

# Size of bin, in pixels. 4-10 is typical range.
bin_width = 4
# Starting pixel. Useful for making sure any features are
# robust to exact bin placement.
shift = 2

[MODEL]

# Fixed nonlinear limb darkening, or fittable linear limb darkening?
# If linear, this also reads in residuals from linear-LD white light
# fit. Otherwise, residuals from fixed non-linear LD white light fit
# are used.
limb_type = nonlinear

# Use marginalization, or ramp? Ramp not yet functional.
method = marg

# Inflate flux errors such that reduced chi-squared = 1 (and so deflating
# the depth error, since reduced chi-squared can be less than 1)?
# Default is no.
include_error_inflation = no

# How to handle residuals. Use them? Include the adjustment
# to the depth at each model which accounts for the bias of
# that model's whitelight depth?
include_residuals = yes
# Always keep wl_adjustment on, except for testing.
include_wl_adjustment = yes

# Run MCMC on the best fitting model from the marginalization grid?
mcmc = no

# Determine what source to use for limb darkening coefficients. Claret2012.csv
# is preferred over claret2011.csv for cooler planets.
limb_source = claret2012.csv

# Visit long-slopes paramterizations. Default is linear, only use
# others if testing impact.
linear_slope = yes
# Do not use higher slopes.
quad_slope = no
exp_slope = no
log_slope = no

# For bi-directional scan, fit with one slope for both directions. Always
# set to "yes" unless testing. 
one_slope = yes

# Set fit_plots to yes to visualize best fit light curve for
# each systematic model. Program is much faster when set to
# no.
fit_plots = no
plot_best = no

[SAVE]

# Save details for each marginalization model in grid.
save_results = yes
save_spectrum = yes

# Save outputs of MCMC fit.
save_mcmc = no

[COMPARE SPECTRA]

compare_current = yes
visits = l9859b/visit00/both_no_first_exps;l9859b/visit01/both_no_first_exps;l9859b/visit03/both_no_first_exps;l9859b/visit02/both_no_first_exps
methods = marg00110;marg00110;marg00110;marg00110
bin_widths = 10;10;8;10
save_comparison = yes
save_comparison_name = l9859b_all
