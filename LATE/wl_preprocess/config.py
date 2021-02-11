[DATA]
planet = l9859b
visit_number = visit02
scan_direction = both
transit = yes
check = yes
ignore_first_exposures = yes
inp_file = no
data_plots = yes

[MODEL]
mcmc = yes
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
# Only linear and nonlinear currently work.
limb_type = nonlinear

[SAVE]
# Necessary for spectral fits
save_processed_data = no
# Necessary for wlresids.py
save_model_info = no
save_mcmc = no
