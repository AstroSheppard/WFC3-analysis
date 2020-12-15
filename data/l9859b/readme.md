inputs.dat: orbital parameters used in transit light curve fitting. These values set priors.

kurcz.dat, continuum.dat, wave.dat: These are stellar spectrum files from ATLAS stellar models (Kurucz.harvard.edu/grids.html). For the metallicity of the star (see below) I pick the most appropriate odfnew grid (odfnew typically gives the best fit and is the most up to date). I then find the closest Teff (increments of 500K) and logg (increments of 0.5). Wave.dat contains the model wavelengths. kurucz.dat contains the total line opacities at each wavelength. Continuum.dat contains the total continuum opacity at each wavelength. Specifics aren't super important, as these are used with sensitivity.fits to get a wavelength solution.


Limb-darkening breakdown: 

There are 2 sources. The first is Claret 2011, which is based on ATLAS and allows for different metallicities. This is denoted "claret2011.csv"

The second is from Claret 2012 (2013 for planet >5000K). This is only Phoenix, so no metallicity flexibility, but the opacities are more updated and the differences seem significant. Not sure which to favor, so test both.

Finally, as a reminder, the preferred LD values are the 3d ones from Magic 2015. However, this only goes down to 4000K, and for these Earth and neptune planets the stellar temp is closer to 3500K.}

The inputs for L98-59 are logg=5.0, Teff=3400K (from vizier TIC-v8 database --- which itself is from the cool dwarf spectroscopy. Also Teff=3400K for discovery paper).

For Claret 2011, inputs are logg=5.0, Teff=3500K, Metallicity -0.5 (from disc paper), x=2.0. The Teff and x are because different metallicities were only allowed for Atlas, which ionly did 500K increments.

