[DATA]
planet = l9859b
visit_number = visit00
scan_direction = both
transit = yes
check = yes
ignore_first_exposures = yes
inp_file = no
data_plots = yes


[MODEL]
mcmc = no
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
fit_plots = yes

[SAVE]
save_processed_data = no
save_model_info = no
save_mcmc = no
