First, active "to-dos" in this directory.

Marg_mcmc.py is copied as template for applying MCMC to spectral marginalization curves.

Both reference RECTE/ramp, but not totally clear what bin_mcmc_fixed, fixed over bin_mcmc? Or how bin_ramp_mcmc fits in? But will confirm. Binramp does not use mcmc, but should not require a whole other program. So, combine these 4.I believe bin_ramp_mcmc is the default, as it saves files correctly.


Binfit_new.py sets uses fitter in marg_new.py to derive spectrum for marginalization. Filt_waves.csv gives a list of wavelengths for common filters in order to interpolate limb-darkening to bin's wavelength. Emcee_runs contain diagnostics for MCMC (visualization of model fits, pickle file of sampler, autocorrelation estimate and residuals for chains, visualization of chains, corner plot, 3 sigma ranges of parameters). 
