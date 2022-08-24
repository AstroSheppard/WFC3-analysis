DEFLATE:
**D**ata **E**xtraction and **F**lexible **L**ight curve **A**nalysis for **T**ransits and **E**clipses

This package is a highly-customizable Python 3 analysis pipeline for Hubble Space Telescope Wide Field Camera-3 (HST WFC3) exoplanet light curve data. Currently, it deals specifically with spatial scan observations with the G141 grism. It consists of three main components:

1. Fits image files -> physical flux time series (Data extraction and reduction: custom routines to derive spectral and broadband light curves from "\_ima.fits" files)
2. Time series -> transit spectra
3. A suite of light curve fit quality diagnostics, including red noise analyses.

Specifics are described in more detail in Sheppard et al, 2021. At it's core, this package marginalizes over instrumental models to retrieve physical transit depths (see Gibson 2014 and Wakeford 2016). It utilizes many great astronomy python packages, such as BATMAN (Kreidberg), Emcee (Foreman-Mackey), MC3 (Cubillos), KMPFIT, Astropy, and others. Additional credit goes to Dr. Hannah Wakeford, who got me started with IDL light curve fitting program templates back in 2016.

This is a beta version. Currently, you must clone the repository to use it.

