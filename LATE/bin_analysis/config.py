# Inputs to all bin_analysis files, including binfit_new.py,
# marg_new.py, compare_spectra.py. 

[DATA]
planet = l9859b
visit_number = visit02
scan_direction = both
transit = yes

# These dictate which residuals to use, but don't impact bin
# fitting assumptions at all.
# In MODEL section, however, limb_type will impact both.

# First: Use white light fit that:
ignored_exposures = yes
openar = no
inflated_errors = yes

# Second: Even when ignored in the white light fit, include
# in the spectral fit?
include_first_orbit = no
# Removed points include only the first few exposures ignored
# in the whitelight fit. It is independent from include_first_orbit.
include_removed_points = no

# Size of bin, in pixels. 4-12 is typical range.
bin_width = 4
# Starting pixel. Useful for making sure any features are
# robust to exact bin placement.
shift = 0

[MODEL]

# Fixed nonlinear limb darkening, or fittable linear limb darkening?
# If linear, this also reads in residuals from linear-LD white light
# fit. Otherwise, residuals from fixed non-linear LD white light fit
# are used.
limb_type = nonlinear

# Use marginalization, or ramp? Ramp not yet functional.
method = marg

# Inflate flux errors such that reduced chi-squared = 1 (and so
# often deflating  the depth error, since reduced chi-squared
# can be less than 1)?
# Default is no.
include_error_inflation = no

# How to handle residuals. Use them? Include the adjustment
# to the depth at each model which accounts for the bias of
# that model's whitelight depth?
include_residuals = yes
# Always keep wl_adjustment on, except for testing.
include_wl_adjustment = yes

# Run MCMC on the best fitting model from the marginalization grid?
# Not yet functional.
mcmc = no

# Determine what source to use for limb darkening coefficients.
# Claret2012.csv is preferred over claret2011.csv for cooler
# planets. Best practice to use same source as in white light fit.
limb_source = claret2012.csv

# Visit long-slopes paramterizations. Default is linear, and quad
# is also useful when includin the first orbit. 
linear_slope = yes
quad_slope = no
# These non-linear slopes are not recommended, still in testing phase.
exp_slope = no
log_slope = no

# For bi-directional scan, fit with one slope for both directions. Always
# set to "yes", unless testing. 
one_slope = yes

# Set fit_plots to yes to visualize best fit light curve for
# each systematic model. Program is much, much faster when set to
# no.
fit_plots = no
# To only view the best fit model (and it's autocorrelation
# function) for each bin, set plot_best to on. However, the
# figure must be closed after every bin fitting in order to
# continue the process.
plot_best = no

[SAVE]

# These will overwrite other saves, so if only testing
# be careful saving.

# Save details for each marginalization model in grid. These are
# useful for diagnostics and necessary for plotting the light curves
# and correlated noise figure.
save_results = no
# Save the spectrum, but not the model details. 
save_spectrum = yes

# Save outputs of MCMC fit. Not yet functional.
save_mcmc = no

[COMPARE SPECTRA]

# Works as input for both binfit_new.py and compare_spectra.py

# To compare spectrum derived by binfit_new to any other spectra, set
# compare_current to yes.
compare_current = yes
# To save comparison image to ./outputs/spectra_figures, set to yes.
save_comparison = yes
save_comparison_name = l9859b_all

# List of visits to compare. For binfit_new, these will all be compared
# to the derived spectrum. For compare_spectra.py, they will only be
# compared to one another. Separate with a semi-colon, no spaces.

# Format: visit = planet/visitXX/direction_whitelight_details
visits = l9859b/visit00/both_no_first_exps;l9859b/visit01/both_no_first_exps;l9859b/visit03/both_no_first_exps;l9859b/visit02/both_no_first_exps
# If comparing derived spectrum to only one other, do not use
# semi-colon. 
# visits = l9859b/visit00/both_no_first_exps

# Format: methods=margXXXXX, with each X corresponding to a
# spectral fitting assumption. See README.MD for meaning of each digit.
methods = marg00110;marg00110;marg00110;marg00110

# Bin size of each spectrum being compared.
bin_widths = 10;10;8;10

# If the spectra being compared all share one characteristic (e.g,
# all the same visit, or all the same bin size), you can list
# just one item. 

# For example, bin_widths = 10;10;10 and bin_widths = 10 are equivalent
# if 3 visits and methods are provided. 
