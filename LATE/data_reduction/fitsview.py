import sys
from astropy.io import fits
import matplotlib.pyplot as plt
if __name__ == '__main__':
  visit = sys.argv[1] + '/' + sys.argv[2] + '/' + sys.argv[3] + '/' + sys.argv[4]
  exp_file = sys.argv[5] + '.fits'
  img_file = './reduced/' + visit + '/' + exp_file
  with fits.open(img_file) as exp:
    test = exp[0].data
    plt.imshow(test)
    plt.show()
