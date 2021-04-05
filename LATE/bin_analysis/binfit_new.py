import configparser
import sys
sys.path.insert(0, '../wl_preprocess')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav

import marg_new
# import binramp as ramp
# from preprocess_ramp import intrinsic, get_data
from get_limb import get_limb
from compare_spectra import compare_spectra
#import bin_mcmc_fixed as mcmc
# This one is newer for ramp mcmc
#import bin_ramp_mcmc as mcmc

def binfit(visit
           , width=6
           , method='marg'
           , shift=0
           , inflated_errors=True
           , ignored_exposures=False
           , openar=False
           , include_first_orbit=True
           , include_removed_points=True
           , include_residuals=True
           , include_wl_adjustment=True
           , include_error_inflation=False
           , ld_type='nonlinear'
           , linear_slope=True
           , quad_slope=False
           , exp_slope=False
           , log_slope=False
           , one_slope=False
           , mcmc=False
           , save_mcmc=False
           , save_spectrum=False
           , save_results=False
           , betas=None
           , transit=True
           , fit_plots=False
           , plot_best=True):


  # Read in appropriate white light fit residuals. 
  planet_index = visit
  if inflated_errors == False:
    visit = visit + '_no_inflation'
  if ld_type == 'linear':
    visit = visit + '_linearLD'
  if ignored_exposures == True:
    visit = visit + '_no_first_exps'
  if openar == True:
    visit = visit + '_openar'
  if quad_slope == True:
    visit = visit + '_quad'
  
  # Read in all relevant whitelight and system values
  # to set up bin fit.

  # First, get time series spectrum and errors
  white='../wl_preprocess/data_outputs/'
  if method=='marg':
    proc_file = white + 'processed_data.csv'
    pre = pd.read_csv(white + 'preprocess_info.csv', index_col=[0, 1])
    pre = pre.loc[(planet_index, ignored_exposures)]
    proc = pd.read_csv(proc_file, index_col=[0,1]).loc[planet_index]
    syst = pd.read_csv(white + 'wl_data.csv', index_col=[0,1]).loc[visit]

    first = pre['User Inputs'].values[-2].astype(int)
    last = pre['User Inputs'].values[-1].astype(int)      
    transit = proc['Transit'].values[0]
    date = proc.loc['Value','Date'].values
    spectra = proc.loc['Value'].drop(['Date', 'sh'
                                      , 'Mask', 'Transit'
                                      , 'Scan Direction']
                                     , axis=1).dropna(axis=1).values
    spec_err = proc.loc['Error'].drop(['Date', 'sh'
                                       , 'Mask', 'Transit'
                                       , 'Scan Direction']
                                      , axis=1).dropna(axis=1).values
    sh = proc.loc['Value', 'sh'].values
    dir_array = proc.loc['Value', 'Scan Direction'].values

    # First = the cutoff point for the first orbit. 0 if orbit included.
    # Read in residuals [nExposure x nModels]
    resids_df = pd.read_csv(white+'wlresiduals.csv', index_col=0).loc[visit]
    resids_df.drop(['Transit', 'Scan Direction'], axis=1, inplace=True)
    # iloc work?

    # Fillna was used when the earlier wl (50 models) was combined with new wl (125 models) and it kept
    # arrays same dimension. Dropna was used when both 50/50 and 125/125 were possible. I'll keep
    # for now, but might be unnecessary in future.
    # resids_df = resids_df.fillna(0)
    # This does nothing since unused models are value=0 now and not value=NaN. Still, doesn't hurt.
    resids_df = resids_df.dropna(axis=1)

    ### Get numeric columns (only needed to do once, when columns were out of order)
    #cols = [int(x[-2:]) for x in resids_df.columns]
    #resids_df.columns = cols
    # Sort columns to appropriate order
    #resids_df = resids_df.sort_index(axis=1)
    ###

    # Now, finally get values. This should match date and everything in length.
    wlresiduals = resids_df.values
    # WLresiduals = WLresiduals[first:]

    ### code to correctly order file:
    #cur = pd.read_csv('wlresiduals_orig.csv', index_col=[0,1])
    #cols = [int(x[6:]) for x in cur.columns]
    #cur.columns = cols
    #cur = cur.sort_index(axis=1)
    #new_cols = ['Model ' + str(x) for x in cur.columns]
    #cur.columns = new_cols
    #cur.to_csv('./wlresiduals.csv', index_label=['Obs', 'Transit'])

    # Array which holds which exposures were ignored in whitelight analysis, if any.
    mask = proc.loc['Value', 'Mask'].values
    
    # First, get same HST phase as whitelight fit by using the same HST
    # reference time.  
    HSTmidpoint = syst.loc['HST midpoint', 'Values'].astype(int)
    if ignored_exposures == False:
      mask = np.ones_like(mask)
      
    HST_phase_ref = date[mask][first+HSTmidpoint]

    # Now, set mask to exclude points. If first exposures are ignored, then keep mask as is. Else,
    # make mask useless by using every point. Second, if excluding the first orbit, set mask at points
    # before first included point to false.

    # this doesnt work
    
    if include_removed_points==True:
      adj = np.where(mask)[0][0]
      first = first + adj
      mask = np.ones_like(mask)
    # Get HST phase using same 0 point as the whitelight curve. If not
    # including the first orbit, this will be the first exposure.
    # Else, this will be the first exposure in the second orbit.
    
    if include_first_orbit==True:
      first = 0

    # I now have all the "data", or inputs into the model at every exposure included in the analysis.
    date = date[mask][first:]
    spectra = spectra[mask][first:]
    spec_err = spec_err[mask][first:]
    sh = sh[mask][first:]
    dir_array = dir_array[mask][first:]
    WLresiduals = wlresiduals[mask][first:]

    # Get HSTphase (last component) at each exposure.
    # Calculate HST phase for each orbit based on same zero-point as
    # the whitelight case.
    HSTper = 96.36 / (24.*60.)
    HSTphase = (date - HST_phase_ref) / HSTper
    HSTphase = HSTphase - np.floor(HSTphase)
    HSTphase[HSTphase > 0.5] = HSTphase[HSTphase > 0.5] - 1.0 
    
    # Second, read in system parameters that are possibly determined by whitelight fit.
    depth_start = syst.loc['Marg Depth', 'Values']
    best_tcenter = syst.loc['Marg Epoch', 'Values']
    #best_tcenter=syst.loc['Marg Epoch'].dropna(axis=0).values[0]
    best_inc = syst.loc['Inc','Values']
    best_ar = syst.loc['ar', 'Values']
    tc_err = syst.loc['Epoch err', 'Values']
    inc_err = syst.loc['Inc err', 'Values']
    ar_err = syst.loc['ar err', 'Values']

    scale = syst.loc['Error Scaling', 'Values']
    y = spectra.sum(axis=1)
    norm = syst.loc['Flux Norm Value', 'Values']
    wlerrors = scale*np.sqrt(np.sum(spec_err*spec_err, axis=1)) / norm

    params = pd.read_csv(white+'system_params.csv'
                         , index_col=0).loc[planet_index, 'Properties'].values
    period = params[4]

    if transit == False:
      rprs = params[0]
    else:
      rprs = np.sqrt(depth_start)

  if method == 'ramp' or method == 'mcmc':
    proc=white+'processed_ramp.csv'
    pre=pd.read_csv(white+'preprocess_ramp_info.csv', index_col=0).loc[visit]
    df=pd.read_csv(proc, index_col=[0,1]).loc[visit]

    transit=df['Transit'].values[0]
    date=df.loc['Value','Date'].values
    spectra=df.loc['Value'].iloc[:,1:-1].values
    spec_err=df.loc['Error'].iloc[:,1:-1].values

    # Second, read in system parameters that are possibly determined by whitelight fit
    syst=pd.read_csv(white+'wl_ramp_params.csv', index_col=[0,1]).loc[visit,'Values']
    depth_start=syst['Depth']
    best_tcenter=syst['Event time']
    best_inc=syst['inc']
    best_ar=syst['ar']
    # Ramp method will never ignore first orbit or first point, so
    # no adjustments needed.
    norm1 = syst['Norm index1'].astype(int)
    norm2 = syst['Norm index2'].astype(int)
    #c1 = syst['c1']
    #c2 = syst['c2']
    period = syst['Period']
    if transit==True:
      rprs = syst['rprs']
    else:
      rprs=pd.read_csv(white+'system_params.csv', index_col=0).loc[visit,'Properties'].values[0]


  # Set all inputs to appropriate priors
  inputs = np.zeros((10, 2))
  inputs[0, 0] = rprs
  inputs[1, 0] = best_tcenter
  inputs[2, 0] = best_inc
  inputs[3, 0] = best_ar
  inputs[4, 0] = period
  inputs[5, 0] = depth_start
  inputs[0, 1] = 0
  inputs[1, 1] = tc_err
  inputs[2, 1] = inc_err
  inputs[3, 1] = ar_err
  inputs[4, 1] = 0
  inputs[5, 1] = 0

  # Fourth, read in wave solution
  wave_file = planet_index
  wave_dir = white + 'wave_sol/wave_solution.csv'
  #wave_file = 'l9859c/visit00/forward'
  wavelength=pd.read_csv(wave_dir
                         , index_col=0).loc[wave_file, 'Wavelength Solution [A]'].values

  ##### Now we have all necessary data #####
  # Determine bin size an allocate pixels to appropriate wavelength bins
  index = np.where((wavelength>11200.) * (wavelength<16600.) == True)[0]
  size = len(index[shift:])
  nbins = size // width

  start = index[0+shift]
  nexposure = len(spectra[:, 0])
  bins = np.zeros((nbins, nexposure, width))
  bins_error = np.zeros((nbins, nexposure, width))
  center = np.zeros(nbins)

  for i in range(nbins):
    n1 = start + width*i
    n2 = n1 + width
    center[i] = (wavelength[n1]+wavelength[n2]) / 2
    bin = spectra[:, n1:n2]
    err = spec_err[:, n1:n2]
    bins[i, :, :] = bin
    bins_error[i, :, :] = err
    print('%.4f--%.4g microns' % (wavelength[n1]/1e4, wavelength[n2]/1e4))

  ### get limb darkening non-linear coeff to be fixed for each bin
  planet = visit.split('/')[0]
  # Load is for externally determined LD coefficients saved in a csv file.
  load = False
  if ld_type=='nonlinear':
    a1 = get_limb(planet, center, 'a1', source=ld_source)
    a2 = get_limb(planet, center, 'a2', source=ld_source)
    a3 = get_limb(planet, center, 'a3', source=ld_source)
    a4 = get_limb(planet, center, 'a4', source=ld_source)
  elif ld_type=='linear':
    a1 = get_limb(planet, center, 'u', source=ld_source)
    a2 = np.zeros_like(a1)
    a3 = np.zeros_like(a1)
    a4 = np.zeros_like(a1)
  else:
    raise ValueError('Error: Must choose nonlinear (fixed) ' \
                     'and or linear (open) limb-darkening model.')

  # depending on method and save, I go through each bin, get the count, get the photon error
  # get depth, error, photon error, stddev(resids), count
  residuals = np.zeros(4)
  resids = np.zeros(nbins)
  depth = np.zeros(nbins)
  error = np.zeros(nbins)
  count = np.zeros(nbins)
  flux_error = np.zeros(nbins)
  flux_error_test = np.zeros(nbins)

  if betas is None: betas=np.ones(nbins)
  for i in range(nbins):
    binned_spectra = bins[i, :, :]
    binned_error = bins_error[i, :, :]
    count[i] = np.median(np.sum(binned_spectra, axis=1))
    flux_error[i] = np.median(np.sqrt(np.sum(binned_error*binned_error, axis=1))/
                              np.median(np.sum(binned_spectra, axis=1)))*1e6
    flux_error_test[i] = np.median(np.sqrt(np.sum(binned_error*binned_error, axis=1))/
                                   np.sum(binned_spectra, axis=1))*1e6
    beta = betas[i]

    if method == 'ramp':
      inputs[6:] = a1[i],a2[i],a3[i],a4[i]
      light=np.sum(binned_spectra, axis=1)
      raw, exptime = get_data(visit, 0, 0, get_raw=1) # 0, 0 are x,y aperture. May need to fix in future
      p1=start+width*i
      p2=p1+width
      pix=(p1,p2)
      intrinsic_count = intrinsic(date, light, raw, pixels=pix)/exptime
      depth[i], error[i], resids[i]=ramp.binramp(inputs, date, binned_spectra, binned_error
                                                 , intrinsic_count, exptime, visit, width, beta
                                                 , plotting=False, transit=transit, save=save
                                                 , nbin='%02d' % i)
    if method == 'mcmc':
      inputs[6:] = a1[i],a2[i],a3[i],a4[i]
      light=np.sum(binned_spectra, axis=1)
      raw, exptime = get_data(visit, 0, 0, get_raw=1) # 0, 0 are x,y aperture. May need to fix in future
      p1=start+width*i
      p2=p1+width
      pix=(p1,p2)
      intrinsic_count = intrinsic(date, light, raw, pixels=pix)/exptime

      depth[i], error[i], resids[i]=mcmc.binramp(inputs, date, binned_spectra, binned_error
                                                 , intrinsic_count, exptime, visit, width, beta
                                                 , plotting=False, transit=transit, save=save_model_info, nbin='%02d' % i)

    if method == 'marg':

      specs = 'marg' + str(int(include_first_orbit)) + str(int(include_removed_points)) \
              + str(int(include_residuals)) + str(int(include_wl_adjustment)) \
              + str(int(include_error_inflation))
      inputs[6:, 0] = a1[i], a2[i], a3[i], a4[i]
      inputs[6:, 1] = [0, 0, 0, 0]
      #depth[i], error[i], resids[i] = marg.marg(inputs, date, binned_spectra, binned_error
      #                                          , norm1, norm2, visit, width, beta, wlerrors
      #                                          , dir_array=dir_array, sh=sh
      #                                          , plotting=False, transit=transit, save=False
      #                                          , nbin='%02d' % i)


      # inputs is ( +4 LD at the end)
      # inputs[0] = rprs

      # wl_data.csv
      # inputs[1] = best_tcenter
      # inputs[2] = best_inc
      # inputs[3] = best_ar
      # system_params.csv
      # inputs[4] = period
      # inputs[5] = depth_start

      # date, dir_array, sh,  and base binned spec and error are from processed data.
      # Set sh to 1 or false or something to test wavelength independent sh.
      # visit is for residuals, wl depth adjustment (wl_data), and saving
      # width is for saving
      # beta is relic of error scaling, but I avoid it
      # wlerrors is the flux errors on the white light curve for fitting
      # HSTphase is used for fitting, calculated above for each exposure used in fit
      # first is for residuals I think, and used for residuals. Could be replaced by residuals...
      # plotting=False: copy from marg_mcmc
      # transit=transit
      # save is for saving what, bin_params, bin_data, and bin_smooth, which are for verification.
      # include wl/residuals/inflation determines fitting stuff. Ld_type should be like this

      depth[i], error[i], resids[i] = marg_new.marg(inputs
                                                    , date
                                                    , binned_spectra
                                                    , binned_error
                                                    , visit
                                                    , width
                                                    , beta
                                                    , wlerrors
                                                    , dir_array
                                                    , HSTphase
                                                    , WLresiduals
                                                    , sh=sh
                                                    , method=specs
                                                    , include_wl_adjustment=include_wl_adjustment
                                                    , include_residuals=include_residuals
                                                    , include_error_inflation=include_error_inflation
                                                    , ld_type=ld_type
                                                    , linear_slope=linear_slope
                                                    , quad_slope=quad_slope
                                                    , exp_slope=exp_slope
                                                    , log_slope=log_slope
                                                    , one_slope=one_slope
                                                    , transit=transit
                                                    , fit_plots=fit_plots
                                                    , plot_best=plot_best
                                                    , save_results=save_results
                                                    , mcmc=mcmc
                                                    , save_mcmc=save_mcmc
                                                    , nbin='%02d' % i)


  photon_err = 1e6 / np.sqrt(count)
  residuals[0] = np.median(resids)
  residuals[1] = np.median(photon_err)
  # Convert to PPM
  error *= 1e6
  depth *= 1e6
  residuals[3], residuals[2] = wmean(depth, error)
  spread = np.zeros_like(center) + (center[1] - center[0])/2.

  label = '%d pixel %s' % (width, specs)
  plt.errorbar(center, depth, error, xerr=spread, fmt='o'
               , ls='', label=label )

  plt.legend()
  planet = visit.split('/')[0]
  plt.title('%s transit' % (planet))
  plt.xlabel(r'Wavelength [$\mu$m]')
  plt.ylabel(r'$(R_p/R_s)^2$ [ppm]')

  print("Ratio of resids to photon error:", (resids/photon_err))
  print("Median ratio of resids to photon error: %.2f" % np.median(resids/photon_err))
  print("Mean ratio of resids to photon error: %.2f" % np.mean(resids/photon_err))
  print("Ratio of resids to theoretical limit flux  error:", (resids/flux_error))
  print("Median ratio of resids to theoretical limit flux  error: %.2f" % np.median(resids/flux_error))
  print("Mean ratio of resids to theoretical limit flux  error: %.2f" % np.mean(resids/flux_error))
  plt.clf()
  plt.errorbar(center, depth, error, fmt='o', ls='', color='b', ecolor='b')
  #plt.savefig()
  plt.show()
  #plt.errorbar(center[1:-1], depth[1:-1], error[1:-1], fmt='o', ls='', color='b', ecolor='b')
  #plt.savefig()
  #plt.show()
  if save_spectrum == True:

    #  if method == 'ramp':
    #    savename2='./spectra_april/'+inp_file

    #  if method == 'marg':
    #    savename2='./spectra_april/'+inp_file + 'adj'

    #ratio=residuals[0]/photon_err
    # index=np.where(ratio < 1.5)
    # ;   center=center[index]
    # ;   depth=depth[index]
    # ;   error=error[index]

    #plt.errorbar(center, depth, error, fmt='o', ls='', color='b', ecolor='b')
    #plt.show()
    # save, filename='resids.sav', r, photon_err
    #SAVE, filename=savename2+'.sav', wavelength, center, width, nbins, start, depth, error, count
    #e.save, savename2+'.pdf'

    # THINGS TO SAVE HERE FOR LATER: for each bin
    # THEORY ERROR, PHOTON ERROR, RMS,  RMS/THEORY ERROR, center wavelength, depth, depth error
    # , start wavelength, end wavelength

    # Index will be visit, bin size, and method.  Columns are depth, depth error, wavelenth range (2),
    # rms/theory err, rms, theory err, photon-limited error  9 x nbins, labeled my binsize, method, obs


    cols = ['Central Wavelength', 'Wavelength Range', 'Depth', 'Error'
            , 'RMS/Theory', 'RMS', 'Theory', 'Photon Error']

    if method == 'marg':
      method = 'marg' + str(int(include_first_orbit)) + str(int(include_removed_points)) \
               + str(int(include_residuals)) + str(int(include_wl_adjustment)) \
               + str(int(include_error_inflation))


    data = np.vstack((center/1e4, spread/1e4, depth, error
                      , resids/flux_error, resids, flux_error
                      , photon_err)).T
    spec = pd.DataFrame(data, columns=cols)
    spec['Visit'] = visit
    spec['Method'] = method
    spec['Bin Size'] = width
    spec['Beta Max'] = 1
    spec = spec.set_index(['Visit', 'Method', 'Bin Size'])
    try:
      cur = pd.read_csv('./outputs/spectra.csv', index_col=[0, 1, 2])
      cur = cur.drop((visit, method, width), errors='ignore')
      cur = pd.concat((cur, spec))
      #cur=cur[~cur.index.duplicated(keep='first')]
      cur.to_csv('./outputs/spectra.csv', index_label=['Obs', 'Method', 'Bin Size'])
    except IOError:
      spec.to_csv('./outputs/spectra.csv', index_label=['Obs', 'Method', 'Bin Size'])

  return center, depth, error, resids, spread


def wmean(val, dval):
  dw = np.sum(1. / (dval*dval))
  wsum = np.sum(val/dval/dval) / dw
  error = np.sqrt(1./dw)
  results = wsum, error
  return results


if __name__ == '__main__':

  config = configparser.ConfigParser()
  config.read('./config.py')

  # Read in data.
  planet = config.get('DATA', 'planet')
  visit_number = config.get('DATA', 'visit_number')
  direction = config.get('DATA', 'scan_direction')
  visit = planet + '/' + visit_number + '/' + direction
  transit = config.getboolean('DATA', 'transit')
  bin_width = config.getint('DATA', 'bin_width')
  shift = config.getint('DATA', 'shift')

  # Which white-light fit to get residuals from.
  ignored_exposures = config.getboolean('DATA', 'ignored_exposures')
  inflated_errors = config.getboolean('DATA', 'inflated_errors')
  openar = config.getboolean('DATA', 'openar')
  include_first_orbit = config.getboolean('DATA', 'include_first_orbit')
  include_removed_points = config.getboolean('DATA', 'include_removed_points')

  # Specify spectral model inputs.
  method = config.get('MODEL', 'method')
  mcmc = config.getboolean('MODEL', 'mcmc')  
  include_error_inflation = config.getboolean('MODEL', 'include_error_inflation')
  include_wl_adjustment = config.getboolean('MODEL', 'include_wl_adjustment')
  include_residuals = config.getboolean('MODEL', 'include_residuals')
  ld_type = config.get('MODEL', 'limb_type')
  ld_source = config.get('MODEL', 'limb_source')
  linear_slope = config.getboolean('MODEL', 'linear_slope')
  quad_slope = config.getboolean('MODEL', 'quad_slope')
  exp_slope = config.getboolean('MODEL', 'exp_slope')
  log_slope = config.getboolean('MODEL', 'log_slope')
  one_slope = config.getboolean('MODEL', 'one_slope')
  fit_plots = config.getboolean('MODEL', 'fit_plots')
  plot_best = config.getboolean('MODEL', 'plot_best')
  beta = None

  # Specify save info.
  save_mcmc = config.getboolean('SAVE', 'save_mcmc')
  save_results = config.getboolean('SAVE', 'save_results')
  save_spectrum = config.getboolean('SAVE', 'save_spectrum')

  # If beta is set to true as the 5th variable, then get beta values from a previous run
  if beta is not None:
    spectra = pd.read_csv('../bin_analysis/spectra.csv', index_col=[0,1,2]).sort_index()
    beta = spectra.loc[(visit, method, bin_width), 'Beta Max'].values


  cr, dr, er, rr, rs = binfit(visit
                              , width=bin_width
                              , shift=shift
                              , method=method
                              , betas=beta
                              , transit=transit
                              , openar=openar
                              , ignored_exposures=ignored_exposures
                              , inflated_errors=inflated_errors
                              , include_first_orbit=include_first_orbit
                              , include_removed_points=include_removed_points
                              , include_residuals=include_residuals
                              , include_wl_adjustment=include_wl_adjustment
                              , include_error_inflation=include_error_inflation
                              , ld_type=ld_type
                              , linear_slope=linear_slope
                              , quad_slope=quad_slope
                              , exp_slope=exp_slope
                              , log_slope=log_slope
                              , one_slope=one_slope
                              , fit_plots=fit_plots
                              , plot_best=plot_best
                              , mcmc=mcmc
                              , save_mcmc=save_mcmc
                              , save_spectrum=save_spectrum
                              , save_results=save_results)

  # Compare spectrum to other saved ones.
  compare_current = config.getboolean('COMPARE SPECTRA', 'compare_current')
  if compare_current==True:
    if inflated_errors == False:
      visit = visit + '_no_inflation'
    if ld_type == 'linear':
      visit = visit + '_linearLD'
    if ignored_exposures == True:
      visit = visit + '_no_first_exps'
    if openar == True:
      visit = visit + '_openar'
    if quad_slope == True:
      visit = visit + '_quad'
    save_comp = config.getboolean('COMPARE SPECTRA', 'save_comparison')
    save_comparison_name = config.get('COMPARE SPECTRA', 'save_comparison_name')
    visits = config.get('COMPARE SPECTRA', 'visits').split(';')
    methods = config.get('COMPARE SPECTRA', 'methods').split(';')
    bin_widths = config.get('COMPARE SPECTRA', 'bin_widths').split(';')

    visits.append(visit)
    method = 'marg' + str(int(include_first_orbit)) + str(int(include_removed_points)) \
             + str(int(include_residuals)) + str(int(include_wl_adjustment)) \
             + str(int(include_error_inflation))
    methods.append(method)
    bin_widths.append(str(bin_width))
  
    compare_spectra(bin_widths
                    , visits
                    , methods
                    , save_comp = save_comp
                    , save_name = save_comparison_name)
  print('This is outside the function, at the end of the main function')



