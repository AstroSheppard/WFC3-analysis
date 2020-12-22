Series of programs to prepare observed exposures for science, i.e. reduce them. Below they are explained, in order.

separate.py: %run separate.py [planet]

This takes all the observations for a planet, orders them chronologically,  and separates them by visit.


bkg.py: %run bkg.py [planet] [visit##]

This organizes the data by scan direction, zooms in on the 2D spectrum, and removes background contamination, and sorts it into the reduced/bkg directory.

reduction.py: %run reduction.py [planet] [visit##] [direction] [transit] [plotting]

direction = forward, reverse, or both
transit = 0 for eclipse, 1 for transit

This first finds an approximate wavelength solution, and uses that to remove remaining flats (flat.fits). It then uses data quality array to mask pixels which are bad in every image. Finally, it corrects for cosmic rays, and saves the reduced data to reduced/final.

wave_solution.py: Uses stellar spectrum to fit for a wavelength solution

fullzap.py: Cosmic ray correction module, imported by reduction.py

mpfit.py: Fitter used for wavelength solution

plot_times_series and pat_reduction.py: simplified versions used for a quick gauge on non-detrended (trended?) light curve.



