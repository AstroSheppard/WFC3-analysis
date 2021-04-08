# After cloning git, copy this file and rename it "config.py".
# You can then change the config.py file without creating
# conflicts for future git pulls.

[DATA]
planet = l9859c
visit_number = visit00
scan_direction = reverse
transit = yes
data_plots = yes
# Set test_bkg to yes to save fits files to see if
# background window size is appropriate.
test_bkg = no
# Typically, positive or negative position in header indicates scan
# direction. Sometimes this is slightly not true, and an adjustment
# is necessary to properly classify scan directions. 
scan_adjustment = 0.0

[COSMIC_RAY]
first_sigma_cut = 8
second_sigma_cut = 5

[IMAGE_VIEWER]
# Image location: bkg (background removal only)
# or final (complete reduction)
reduction_level = bkg
exposure_number = 045
