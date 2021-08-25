DEFLATE:
**D**ata **E**xtraction and **F**lexible **L**ight curve **A**nalysis for **T**ransits and **E**clipses

This package is a highly-customizable analysis pipeline for Hubble Space Telescope Wide Field Camera-3 (HST WFC3) exoplanet light curve data. Currently, it deals specifically with spatial scan observations with the G141 grism. It consists of three main components:

1. Fits image files -> physical flux time series (Data extraction and reduction: custom routines to derive spectral and broadband light curves from "\_ima.fits" files)
2. Time series -> transit spectra
3. A suite of light curve fit quality diagnostics, including red noise analyses.

Specifics are described in more detail in Sheppard et al, 2021. 

This is a beta version which will be actively developed to make more user-friendly.

