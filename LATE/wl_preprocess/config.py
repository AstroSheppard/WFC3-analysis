[DATA]
planet = l9859c
visit_number = visit00
scan_direction = both
transit = yes
ignore_first_exposures = yes
# Only set inp_file to on if using pre-selected inputs.
# As a default, use "check" instead, which allows user to
# select orbits to include in analysis.
check = yes
inp_file = no
data_plots = no
wave_solution_plots = yes

[MODEL]
mcmc = no
include_error_inflation = yes
openinc = no
openar = no
fixtime = no
norandomt = yes
linear_slope = yes
quad_slope = no
exp_slope = no
log_slope = no
# For bi-directional scan, fit with one slope for both directions.
one_slope = yes
# Set fit_plots to yes to visualize best fit light curve for
# each systematic model. Program is much faster when set to
# no.
fit_plots = no
# Determine what source to use for limb darkening coefficients.
limb_source = claret2012.csv
# Only linear and nonlinear currently work.
limb_type = nonlinear

[SAVE]
# Necessary for spectral fits
save_processed_data = yes
# Necessary for wlresids.py
save_model_info = yes
save_mcmc = yes
