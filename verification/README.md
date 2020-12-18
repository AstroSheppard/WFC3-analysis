wlpaper.py read in data to make white light analysis figures potentially useful in papers. For example, it searches for red noise, checks normality of residuals, and plots raw and de-trended light curves with residuals.

get_stis_curves.py converts Nikolay's sav files into machine readable files for data product for journal.

binpaper_.py: These are the same as wlpaper, but for spectral bin analyses. Notably, this includes the visualization of each de-trended light curve over the best fit model for each wavelength bin all in one figure. There are two versions here:


binpaper.py is from original submission of hatp41 manuscript (and then edits in july 24 2020). This should be the version that works with old parameterization (i.e, one scan direction, and slightly different parameter order). NEW EDIT 12/17: This had been updated to include autocorrelation figures as well as uncertainty contours for red noise figure. Unclear if this still only works for old parameters, will determine.


binpaper2020.py is september/october updated version that works with new parameters. However, it may now be outdated. See previous comment.

MCCubed contains black-box version of red noise curve that should appropriately set error bars (instead of poor approximations for small N). It is not immediately clear that this error is correct, but will it is published and we can verify.