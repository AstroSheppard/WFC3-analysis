Module for preprocessing the data and fitting the white light data. All outputs are in "data_outputs/"

preprocess_whitelight.py preprocesses data and uses marg_mcmc.py to determine white light parameters.

> %run preprocess_whitelight.py
> %run wlresids_new.py gets white light residuals for each systematic model from those fits.
> %run wave_solution.py gets a wave_solution based on the fully reduced data.

First:
open config.py file and edit inputs. 
open ipython

Second:
> %run preprocess_whitelight.py
First preprocesses data (e.g, removes first orbit, gets whitelight fluxes, limb darkening), then
feeds it into marg_mcmc for actual fitting. Saves preprocessing info to preprocess_info.csv, and
data used in fit is saved in processed_data.csv with a flux at every pixel, as well as other
inputs to spectra fit. Saves input values (basically same as planets/inputs.dat, used as priors
for mcmc) to system_params.csv.
Fitting and plotting info saved in wl_data.csv. wl_models_info.csv has specific parameters
and info for each grid model. When mcmc and save_mcmc are true, diagnostics, best_params.csv,
corner plots are saved in data_outputs/emcee_runs.

Third:
> %run wlresids_new.py
This gets extrapolates the model to the first orbit and gets the residuals for each exposure in
the time series for each model in the grid. Outputs the residuals for the most complex model.
Should be an almost exact match for all but first orbit, which will typically look like an upside-down
hook. Residuals should be centered on zero and pretty structure-free. Saves outputs to wlresiduals.csv.

Fourth:
>%run wave_solution.py
This fits an out-of-transit spectrum to an ATMO model to get the corresponding wavelength for each
pixel (wavelength_solution). Saves to data_outputs/wave_sol/wave_solution.csv.


All set for bin fitting now! Go to >cd ../bin_analysis/
------------------------------

# RECTE physical charge-trap model info and background.

HST/WFC3 ramp effect model based on charge trapping. Journal references: [Zhou et al. (2017)](http://adsabs.harvard.edu/abs/2017AJ....153..243Z)

## Usage
The first three input parameters are the intrinsic count rate (in **e/s**)
time series, the time stamps of the exposures, and the exposure time (in seconds). For calculating
ramps for a group of pixels, `crates` should be the average count rates
of the pixels in the group. The next four parameters are the ones that need to be
optimized. `trap_pop_s` and `trap_pop_f` are the initial numbers of occupied
traps and `dTrap_s` and `dTrap_f` are charges trapped during the gaps between
orbits. `trap_pop_s/trap_pop_f` normally ranges from 0 to 100. `dTrap_s`
and `dTrap_f` are often close to 0 and normally less than 50. It will
return the count time sieries (in **e**) with ramp applied. You may also
check the doc-string for additional information.