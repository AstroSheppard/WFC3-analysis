Module for preprocessing the data and fitting the white light data.

preprocess_whitelight.py preprocesses data and uses marg_mcmc.py to determine white light parameters.

%run preprocess_whitelight.py [planet] [visit] [direction]

wlresids_new.py gets white light residuals for each systematic model from those fits. Wave_solution.py gets a wave_solution based on the fully reduced data.


# RECTE

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
