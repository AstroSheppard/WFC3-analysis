# Inputs to all wl_preprocess files, including: preprocess_whitelight.py,
# wlresids_new.py, wave_solution.py.

[DATA]
planet = l9859c
visit_number = visit00
scan_direction = both
# Is this a transit, or an eclipse?
transit = yes
# Ignore the first few exposures of each orbit to make
# polynomial approximation more accurate?
ignore_first_exposures = no
# Only set inp_file to on if using pre-selected inputs.
# As a default, use "check" instead, which allows user to
# select orbits to include in analysis.
check = yes
inp_file = no
# Show the processed data pre-fitting?
data_plots = no
# For wave_solution.py: show result of wavelength solution fit?
wave_solution_plots = yes

[MODEL]
# Run MCMC on the best fitting model from the marginalization grid?
mcmc = no

# Inflate flux errors such that reduced chi-squared = 1 (and so inflating
# the whitelight depth error)?
include_error_inflation = yes

# Physical parameters for which to fit.
openinc = no
openar = no
fixtime = no
# Do not change: norandomt always on.
norandomt = yes

# Visit long-slopes paramterizations. Default is linear, only use
# others if testing impact.
linear_slope = yes
# Do not use higher slopes
quad_slope = no
exp_slope = no
log_slope = no

# For bi-directional scan, fit with one slope for both directions. Always
# keep on. 
one_slope = yes

# Set fit_plots to yes to visualize best fit light curve for
# each systematic model. Program is much faster when set to
# no.
fit_plots = no

# Determine what source to use for limb darkening coefficients. Claret2012 is
# preferred over claret2011 for cooler planets.
limb_source = claret2012.csv

# Fixed nonlinear limb darkening, or fittable linear limb darkening?
limb_type = nonlinear

[SAVE]
# Save flux , error, shift, etc. for every pixel column.
# Necessary for spectral fits.
save_processed_data = yes

# Save details for each marginalization model in grid.
# Necessary for wlresids.py
save_model_info = yes

# Save outputs of MCMC fit.
save_mcmc = yes
