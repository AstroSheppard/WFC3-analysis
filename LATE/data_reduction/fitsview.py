import sys
import glob
import configparser

from astropy.io import fits
import matplotlib.pyplot as plt

def fitsviewer(config_file):

  config = configparser.ConfigParser()
  config.read(config_file)
  planet = config.get('DATA', 'planet')
  visit_number = config.get('DATA', 'visit_number')
  direction = config.get('DATA', 'scan_direction')
  reduction_level = config.get('IMAGE_VIEWER', 'reduction_level')
  exposure_number = config.get('IMAGE_VIEWER', 'exposure_number')
  visit = planet + '/' + visit_number + '/' + direction + '/' \
          + reduction_level
  exp_file = exposure_number + '.fits'
  img_file = './reduced/' + visit + '/' + exp_file
  with fits.open(img_file) as exp:
    test = exp[0].data
    plt.imshow(test)
    plt.show()

  plt.clf()
  plt.close()

if __name__ == '__main__':

  fitsviewer('config.py')

