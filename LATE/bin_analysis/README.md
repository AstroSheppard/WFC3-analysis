To use:
Step 1: Edit config.py to desired inputs. Config.py contains descriptions for each input. 
Step 2: %run bifit_new.py
This calls on marg_new.py to fit spectra.
Step 3 (optional): %run compare_spectra.py. If 2+ visits are provided, it will compare their spectra. See outputs/spectra.csv for visit names to use.

Spectral fits save info: For marginalization, there are **4** important parts to a save file:
visit_info, whitelight fitting assumptions, spectral fitting assumptions, bin size (in pixels).
The 5 possible spectral fitting assumptions are stored as 10101. In order, these are: include first orbit, include ignored first exposures, include white light residuals, include include whitelight adjustment, and include error inflation. The default is 00110. 

For example, a spectrum from both scan directions of the first visit of planet b binned to 6 pixels, which ignored the first few exposures of each orbit in white light fitting, but included them in spectral fitting would be saved as:
>visit, method, binsize
>l9859b/visit00/both_ignore_first_exps, marg01110, 6

----------------------------------------------------------------------------------------------------
First, active "to-dos" in this directory.

Marg_mcmc.py is copied as template for applying MCMC to spectral marginalization curves.

Both reference RECTE/ramp, but not totally clear what bin_mcmc_fixed, fixed over bin_mcmc? Or how bin_ramp_mcmc fits in? But will confirm. Binramp does not use mcmc, but should not require a whole other program. So, combine these 4.I believe bin_ramp_mcmc is the default, as it saves files correctly.


Binfit_new.py sets uses fitter in marg_new.py to derive spectrum for marginalization. Filt_waves.csv gives a list of wavelengths for common filters in order to interpolate limb-darkening to bin's wavelength. Emcee_runs contain diagnostics for MCMC (visualization of model fits, pickle file of sampler, autocorrelation estimate and residuals for chains, visualization of chains, corner plot, 3 sigma ranges of parameters). 
