import configparser
import sys
sys.path.insert(0, '../wl_preprocess')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav

import marg_new
from get_limb import get_limb


def compare_spectra(bin_widths
                    , visits
                    , methods
                    , save_comp=False
                    , save_name='spectra_comp'):
  """ Function to plot spectra from different methods for comparison.
  INPUTS:
  **visit_dict is a dictionary that contains the planet/visit/direction
  and method to search for. E.g, {'l9859c/visit00/both': 'marg0011', 
  'l9859c/visit00/both': 'marg0000', 'l9859c/visit00/reverse': 'ramp'}

  OUTPUTS: Figure comparing the spectra.
  """

  # Find number of comparisons.
  max_len = len(bin_widths)
  if len(visits) > max_len:
    max_len = len(visits)
  if len(methods) > max_len:
    max_len = len(methods)

  if len(bin_widths)==1:
    bin_widths = bin_widths * max_len
  if len(visits)==1:
    visits = visits * max_len
  if len(methods)==1:
    methods = methods * max_len
  assert(len(bin_widths) == len(visits))
  assert(len(bin_widths) == len(methods))
  spec = pd.read_csv('./outputs/spectra.csv', index_col=[0,1,2])
  for i in range(max_len):
    cen = spec.loc[(visits[i], methods[i], int(bin_widths[i]))]
    # all_visit = spec.loc[visits[i]]
    center=cen['Central Wavelength'].values
    spread=cen['Wavelength Range'].values
    depth=cen['Depth'].values
    error=cen['Error'].values
    # if i==max_len-1:
    #   depth+=975
    label = '%s %s' % (visits[i], methods[i])
    plt.errorbar(center, depth, error, xerr=spread, fmt='o'
                 , ls='', label=label )

  plt.legend()
  planet = visits[0].split('/')[0]
  plt.title('%s transit' % (planet))
  plt.xlabel(r'Wavelength [$\mu$m]')
  plt.ylabel(r'$(R_p/R_s)^2$ [ppm]')

  # # Renyu Hu Spectrum approximation for l98c
  # hu = [1.12,1.16, 1.19, 1.21, 1.24, 1.27, 1.295, 1.32, 1.35,1.38, 1.4,
  #       1.42, 1.45, 1.48, 1.51,1.55, 1.58, 1.62]
  # hud = np.array([1600, 1675, 1720, 1635, 1725, 1600, 1660, 1640, 1670, 1700,
  #        1690, 1750, 1730, 1800, 1625, 1700, 1710, 1710]) - 40
  # hue = [40]*18
  # plt.errorbar(hu, hud, hue, fmt='o', color='r', ecolor='r'
  #              , ls='', label='Renyu', alpha=.5)

  if save_comp==True:
    plt.savefig('./outputs/spectra_figures/' + save_name + '.png')
  else:
    plt.show()

  return 1
    


  # plt.close()


  # avi = pd.DataFrame()
  # index = ['Marg + resids - first orbit']*len(cr) + ['Renyu']*18
  # avi['Wavelength [microns]'] = np.append(cr/1e4, hu)
  # avi['Width'] = np.append(rs/1e4, np.zeros_like(hu))
  # avi['Depths'] = np.append(dr, hud)
  # avi['Errors'] = np.append(er, hue)
  # avi.index=index
  # #avi.to_csv('../../l9859c_spectra_0925_10pixel.csv')



  # #plt.errorbar(center5, depth5, error5,xerr=spread5, fmt='o', color='r', ecolor='r'
  # #             , ls='', label='4 pixel no resids')
  # #plt.errorbar(center4*1e4, depth4, error4,xerr=spread4*1e4, fmt='o', color='r', ecolor='r'
  # #             , ls='', label='Old', alpha=.5)
  # # plt.errorbar(center3, depth3, error3,xerr=spread3, fmt='o', color='g'
  # #             , ecolor='g', ls='', label='Normal', alpha=.5)
  # #plt.errorbar(center1, depth1, error1,xerr=spread1, fmt='o', color='orange', ecolor='orange'
  # #             , ls='', label='First orbit + resids', alpha=.5)
  # plt.errorbar(center4, depth4, error4,xerr=spread4, fmt='o', color='b', ecolor='b'
  #              , ls='', label='Paper', alpha=.5)
  plt.errorbar(hu, hud, hue, fmt='o', color='r', ecolor='r'
               , ls='', label='Renyu', alpha=.5)

  # #adjust= (np.median(hud)-np.median(dr) + np.mean(hud) - np.mean(dr)) / 2
  # #print adjust
  # plt.errorbar(cr/1e4, dr, er, xerr=rs/1e4, fmt='o', color='g', ecolor='g'
  #              , ls='', label='No first orbit + resids', alpha=.5)

  # #print 'Difference between current run and saved spectrum: %.2f' % ((dr-depth1).mean())
  
  # #sss
  # #sys.exit()
  # #centerramp=centerramp[1:-2]
  # #depthramp=depthramp[1:-2]
  # #errorramp=errorramp[1:-2]
  # #plt.errorbar(centerramp, depthramp, errorramp,xerr=rampspread, fmt='o', color='g'
  # #             , ecolor='g', ls='', label='Zhou Ramp Method')


  # sav=readsav('nik.sav')
  # lamb=sav.WAV_CEN_ANG*1e4
  # lamb_err=sav.ERR_WAV_ANG*1e4
  # depth=sav.TRANSIT_RPRS**2*1e6
  # error=sav.TRANSIT_RPRS_ERR*sav.TRANSIT_RPRS*1e6*2
  # plt.errorbar(lamb, depth, yerr=error,xerr= lamb_err, fmt='o', color='orange'
  #              , ecolor='orange', ls='', label='Nikolay')
  # df=pd.read_csv('tsarias2.csv')
  # twave=df['wave'].values*1e4
  # tdepth=df['depth'].values
  # terror=df['error'].values
  # plt.errorbar(twave, tdepth, terror, fmt='o', color='purple'
  #              , ecolor='purple', ls='', label='Tsarias')

  # plt.legend(numpoints=1)
  # plt.show()
  # sys.exit()
  # #plt.savefig('avi.png')
  # plt.show()

if __name__=='__main__':
  config = configparser.ConfigParser()
  config.read('./config.py')

  # Read in data.
  visits = config.get('COMPARE SPECTRA', 'visits').split(';')
  methods = config.get('COMPARE SPECTRA', 'methods').split(';')
  bin_widths = config.get('COMPARE SPECTRA', 'bin_widths').split(';')
  save_comp = config.getboolean('COMPARE SPECTRA', 'save_comparison')
  save_name = config.get('COMPARE SPECTRA', 'save_comparison_name')
  compare_spectra(bin_widths
                  , visits
                  , methods
                  , save_comp=save_comp
                  , save_name=save_name)
