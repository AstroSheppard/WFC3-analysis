import sys
sys.path.insert(0, '../wl_preprocess')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav

import marg_new
#import marg_no_adjust
#import marg_no_adjust_no_orbit
#import marg_no_resids
#import marg_no_orbit
import binramp as ramp
from preprocess_ramp import intrinsic, get_data
from get_limb import get_limb
#import bin_mcmc_fixed as mcmc
import bin_ramp_mcmc as mcmc

def binfit(visit
           , width = 6
           , save = False 
           , method = 'marg'
           , shift = 0
           , include_first_orbit = True
           , include_residuals = True
           , include_wl_adjustment = True
           , include_error_inflation = False
           , betas = None):

  # calculate HST phase
  if include_error_inflation == False:
    orig = visit
    planet = visit.split('/')[0]
    visit = 'no_inflation_'+ planet
  else:
    orig = visit

  ########## Read in all relevant whitelight and system values to set up bin fit ##########

  # First, get time series spectrum and errors
  white='../wl_preprocess/'
  if method=='marg':
    proc=white+'processed_data.csv'
    pre=pd.read_csv(white+'preprocess_info.csv', index_col=0).loc[orig]
    df=pd.read_csv(proc, index_col=[0,1]).loc[orig] 

    first = pre['User Inputs'].values[-2].astype(int)
    last = pre['User Inputs'].values[-1].astype(int)
    adj = 0
    
    if include_first_orbit == True:
      adj = first
      first = 0
    
    transit=df['Transit'].values[0]
    date=df.loc['Value','Date'].values
    spectra=df.loc['Value'].iloc[:,1:-2].dropna(axis=1).values
    spec_err=df.loc['Error'].iloc[:,1:-2].dropna(axis=1).values
    sh=df.loc['Value','sh'].values
    dir_array=df.loc['Value','Scan Direction'].values
    
    date = date[first:]
    spectra = spectra[first:]
    spec_err = spec_err[first:]
    sh = sh[first:]
    dir_array=dir_array[first:]
    
    
    # Second, read in system parameters that are possibly determined by whitelight fit
    syst=pd.read_csv(white+'wl_data.csv', index_col=[0,1]).loc[visit]
    depth_start=syst.loc['Marg Depth', 'Values']
    best_tcenter=syst.loc['Marg Epoch', 'Values']
    #best_tcenter=syst.loc['Marg Epoch'].dropna(axis=0).values[0]
    best_inc=syst.loc['Inc','Values']
    best_ar=syst.loc['ar', 'Values']
    #best_ar=5.44
    norm1 = syst.loc['Norm index1', 'Values'].astype(int) + adj 
    norm2 = syst.loc['Norm index2', 'Values'].astype(int) + adj
    scale= syst.loc['Error Scaling', 'Values']
    y=spectra.sum(axis=1)
    norm=np.median(y[norm1:norm2])
    wlerrors=scale*np.sqrt(np.sum(spec_err*spec_err, axis=1))/norm


    params=pd.read_csv(white+'system_params.csv', index_col=0).loc[orig,'Properties'].values 
    period=params[4]
    #period = params[]
    #period = limbs[4]
    #c1, c2 = limbs[6:8]
    
    if transit == False:
      rprs = params[0]
    else:
      rprs=np.sqrt(depth_start)
    
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
  inputs=np.zeros(10)
  inputs[0]=rprs
  inputs[1]=best_tcenter
  inputs[2]=best_inc
  inputs[3]=best_ar
  inputs[4]= period
  inputs[5]=depth_start
  #inputs[6:]=c1, c2

  # Fourth, read in wave solution
  wave_file = orig
  #wave_file = 'l9859c/visit00/forward'
  wavelength=pd.read_csv(white + 'wave_sol/wave_solution.csv'
                         , index_col=0).loc[wave_file,'Wavelength Solution [A]'].values 

  ##### Now we have all necessary data #####
  # Determine bin size an allocate pixels to appropriate wavelength bins
  index=np.where((wavelength>11200.) * (wavelength<16600.) == True)[0]
  size=len(index[shift:])
  nbins=size/width

  start= index[0+shift]
  nexposure=len(spectra[:,0])
  bins=np.zeros((nbins,nexposure, width))
  bins_error=np.zeros((nbins,nexposure, width))
  center=np.zeros(nbins)

  for i in range(nbins):
    n1=start+width*i
    n2=n1+width
    center[i]=(wavelength[n1]+wavelength[n2])/2
    bin=spectra[:,n1:n2]
    err=spec_err[:,n1:n2]
    bins[i,:,:]=bin
    bins_error[i,:,:]=err
    print wavelength[n1]/1e4, wavelength[n2]/1e4


  
  ### get limb darkening non-linear coeff to be fixed for each bin
  # planet=orig[:-16] # change back to visit from orig
  planet=orig.split('/')[0]
  # Load was true before, and was loading incorrect LD values
  load = True
  a1=get_limb(planet, center, 'a1', load=load)
  a2=get_limb(planet, center, 'a2', load=load)
  a3=get_limb(planet, center, 'a3', load=load)
  a4=get_limb(planet, center, 'a4', load=load)

  # depending on method and save, I go through each bin, get the count, get the photon error
  # get depth, error, photon error, stddev(resids), count 
  residuals=np.zeros(4)
  resids=np.zeros(nbins)
  depth=np.zeros(nbins)
  error=np.zeros(nbins)
  count=np.zeros(nbins)
  flux_error=np.zeros(nbins)
  flux_error_test=np.zeros(nbins)

  #if save == True:
    #print 'a'
    ### ignore for now 
    # for i in range(nbins):
    #     print, 'bin' + i
    #     binned_spectra=bins[i,:,:]
    #     count[i]=np.median(np.sum(binned_spectra, axis=1))
    #     if method == 'ramp':
    #       savename='./spectra_april/models/resids/' + inp_file + STRING(i, format='(I02)') 
    #     if method == 'marg':
    #       cen=center[i]
    #       savename='./paper/bincurve/' + inp_file + STRING(i, format='(I02)') 
    #       results = marg(inputs, date, binned_spectra, first_orbit_size
    #                                 , inp_file, first, cc, hst , cen, SAVEFILE=savename)
      
    #     depth_array2[i,0]=results[0]
    #     depth_array2[i,1]=results[1]
    #     r[i]=results[2]
    ###
    #sys.exit('hello')
  #else:
  
  if betas is None: betas=np.ones(nbins)
  for i in range(nbins):
    binned_spectra=bins[i,:,:]
    binned_error=bins_error[i,:,:]
    count[i]=np.median(np.sum(binned_spectra, axis=1))
    flux_error[i]=np.median(np.sqrt(np.ma.sum(binned_error*binned_error, axis=1))/
                            np.median(np.sum(binned_spectra, axis=1)))*1e6
    flux_error_test[i]=np.median(np.sqrt(np.ma.sum(binned_error*binned_error, axis=1))/
                                 np.sum(binned_spectra, axis=1))*1e6
  
  
    beta=betas[i]
  
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
                                                 , plotting=False, transit=transit, save=save, nbin='%02d' % i)
                                                
    if method == 'marg':
      
      inputs[6:] = a1[i],a2[i],a3[i],a4[i]
      #depth[i], error[i], resids[i] = marg.marg(inputs, date, binned_spectra, binned_error
      #                                          , norm1, norm2, visit, width, beta, wlerrors
      #                                          , dir_array=dir_array, sh=sh
      #                                          , plotting=False, transit=transit, save=False
      #                                          , nbin='%02d' % i)

      depth[i], error[i], resids[i] = marg_new.marg(inputs, date, binned_spectra, binned_error
                                                    , norm1, norm2, visit, width, beta, wlerrors
                                                    , dir_array , sh=sh
                                                    , first = first, plotting=False
                                                    , include_wl_adjustment = include_wl_adjustment
                                                    , include_residuals = include_residuals
                                                    , include_error_inflation = include_error_inflation
                                                    , transit=transit, save=save, nbin='%02d' % i)
  

  photon_err=1e6/np.sqrt(count)
  residuals[0]=np.median(resids)
  residuals[1]=np.median(photon_err)
  # Convert to PPM
  error*=1e6
  depth*=1e6
  residuals[3], residuals[2]=wmean(depth, error)
  spread=np.zeros_like(center) + (center[1] - center[0])/2.
  
  print "Ratio of resids to photon error:", (resids/photon_err)
  print "Median ratio of resids to photon error: %.2f" % np.median(resids/photon_err)
  print "Mean ratio of resids to photon error: %.2f" % np.mean(resids/photon_err)
  print "Ratio of resids to theoretical limit flux  error:", (resids/flux_error)
  print "Median ratio of resids to theoretical limit flux  error: %.2f" % np.median(resids/flux_error)
  print "Mean ratio of resids to theoretical limit flux  error: %.2f" % np.mean(resids/flux_error)
  plt.clf()
  plt.errorbar(center, depth, error, fmt='o', ls='', color='b', ecolor='b')
  #plt.savefig()
  plt.show()
  #plt.errorbar(center[1:-1], depth[1:-1], error[1:-1], fmt='o', ls='', color='b', ecolor='b')
  #plt.savefig()
  #plt.show()
  if save == True:

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

    
    cols=['Central Wavelength', 'Wavelength Range', 'Depth', 'Error'
          , 'RMS/Theory', 'RMS', 'Theory', 'Photon Error']

    if method == 'marg':
      method = 'marg' + str(int(include_first_orbit)) + str(int(include_residuals)) \
               + str(int(include_wl_adjustment)) + str(int(include_error_inflation))
      
    
    data=np.vstack((center/1e4, spread/1e4, depth, error,resids/flux_error, resids, flux_error, photon_err)).T
    spec=pd.DataFrame(data, columns=cols)
    spec['Visit']=visit
    spec['Method']=method
    spec['Bin Size']=width
    spec['Beta Max']=1
    spec=spec.set_index(['Visit', 'Method', 'Bin Size'])
    try:
      cur=pd.read_csv('./spectra.csv', index_col=[0, 1, 2])
      cur=cur.drop((visit, method, width), errors='ignore')
      cur=pd.concat((cur,spec))
      #cur=cur[~cur.index.duplicated(keep='first')]
      cur.to_csv('./spectra.csv', index_label=['Obs', 'Method', 'Bin Size'])
    except IOError:
      spec.to_csv('./spectra.csv', index_label=['Obs', 'Method', 'Bin Size'])

    
  return center, depth, error, resids, spread


def wmean(val, dval):
  dw=np.sum(1./(dval*dval))
  sum=np.sum(val/dval/dval)/dw
  error=np.sqrt(1./dw)
  results=sum, error
  return results

if __name__ == '__main__':
    if len(sys.argv) not in [4,5,6]:
        sys.exit()
    planet=sys.argv[1]
    obs = sys.argv[2]
    direction=sys.argv[3]
    visit=planet+'/'+obs+'/'+direction
    width = int(sys.argv[4])
    beta=None
    # If beta is set to true as the 5th variable, then get beta values from a previous run
    if len(sys.argv)==6:
      binsize=int(sys.argv[4])
      spectra=pd.read_csv('../bin_analysis/spectra.csv', index_col=[0,1,2]).sort_index()
      beta=spectra.loc[(visit, 'marg', binsize), 'Beta Max'].values

    #center4, depth4, error4, residuals4, spread4 = binfit(visit, width=16, method='marg3'
    #                                             , save=False, betas=beta)

    #centerramp, depthramp, errorramp, residualsramp, rampspread = binfit(visit, width=4, method='ramp'
    #                                                                     , save=True, betas=beta)

    cr, dr, er, rr, rs = binfit(visit, width=width, method='marg'
                                , betas=beta
                                , include_first_orbit = False
                                , include_residuals = False
                                , include_wl_adjustment = True
                                , include_error_inflation= True
                                , save=True)
    
    """cen=pd.read_csv('spectra.csv', index_col=[0,1,2]).loc[(visit,'marg3',4)]
    center5=cen['Central Wavelength'].values
    spread5=cen['Wavelength Range'].values
    depth5=cen['Depth'].values
    error5=cen['Error'].values
    cen=pd.read_csv('spectra.csv', index_col=[0,1,2]).loc[(visit,'marg3',10)]
    center2=cen['Central Wavelength'].values
    spread2=cen['Wavelength Range'].values
    depth2=cen['Depth'].values
    error2=cen['Error'].values

    cen=pd.read_csv('spectra.csv', index_col=[0,1,2]).loc[(visit,'marg',4)]
    center3=cen['Central Wavelength'].values
    spread3=cen['Wavelength Range'].values
    depth3=cen['Depth'].values
    error3=cen['Error'].values"""
    
    cen=pd.read_csv('spectra.csv', index_col=[0,1,2]).loc[(visit,'marg0011',10)]
    center3=cen['Central Wavelength'].values
    spread3=cen['Wavelength Range'].values
    depth3=cen['Depth'].values
    error3=cen['Error'].values

    #visit = 'l9859c/visit00/forward'
    cen=pd.read_csv('spectra.csv', index_col=[0,1,2]).loc[(visit,'marg1111',4)]
    center4=cen['Central Wavelength'].values
    spread4=cen['Wavelength Range'].values
    depth4=cen['Depth'].values
    error4=cen['Error'].values
    #cen=pd.read_csv('spectra.csv', index_col=[0,1,2]).loc[(visit,'marg1111',10)]
    #center1=cen['Central Wavelength'].values
    #spread1=cen['Wavelength Range'].values
    #depth1=cen['Depth'].values
    #error1=cen['Error'].values
    #print np.median(errorramp)/np.median(error)
    #center=center[1:-2]
    #depth=depth[1:-2]
    #error=error[1:-2]

    hu = [1.12,1.16, 1.19, 1.21, 1.24, 1.27, 1.295, 1.32, 1.35,1.38, 1.4,
          1.42, 1.45, 1.48, 1.51,1.55, 1.58, 1.62]
    hud = [1600, 1675, 1720, 1635, 1725, 1600, 1660, 1640, 1670, 1700,
           1690, 1750, 1730, 1800, 1625, 1700, 1710, 1710]
    hue = [40]*18
    plt.close()


    avi = pd.DataFrame()
    index = ['Marg + resids - first orbit']*len(cr) + ['Renyu']*18
    avi['Wavelength [microns]'] = np.append(cr/1e4, hu)
    avi['Width'] = np.append(rs/1e4, np.zeros_like(hu))
    avi['Depths'] = np.append(dr, hud)
    avi['Errors'] = np.append(er, hue)
    avi.index=index
    #avi.to_csv('../../l9859c_spectra_0925_10pixel.csv')


    
    #plt.errorbar(center5, depth5, error5,xerr=spread5, fmt='o', color='r', ecolor='r'
    #             , ls='', label='4 pixel no resids')
    #plt.errorbar(center4*1e4, depth4, error4,xerr=spread4*1e4, fmt='o', color='r', ecolor='r'
    #             , ls='', label='Old', alpha=.5)
    # plt.errorbar(center3, depth3, error3,xerr=spread3, fmt='o', color='g'
    #             , ecolor='g', ls='', label='Normal', alpha=.5)
    #plt.errorbar(center1, depth1, error1,xerr=spread1, fmt='o', color='orange', ecolor='orange'
    #             , ls='', label='First orbit + resids', alpha=.5)
    plt.errorbar(center4, depth4, error4,xerr=spread4, fmt='o', color='b', ecolor='b'
                 , ls='', label='Paper', alpha=.5)
    #plt.errorbar(hu, hud, hue, fmt='o', color='r', ecolor='r'
    #             , ls='', label='Renyu', alpha=.5)

    #adjust= (np.median(hud)-np.median(dr) + np.mean(hud) - np.mean(dr)) / 2
    #print adjust
    plt.errorbar(cr/1e4, dr, er, xerr=rs/1e4, fmt='o', color='g', ecolor='g'
                 , ls='', label='No first orbit + resids', alpha=.5)

    #print 'Difference between current run and saved spectrum: %.2f' % ((dr-depth1).mean())
    plt.legend()
    plt.title('L9859c transit')
    plt.xlabel(r'Wavelength [$\mu$m]')
    plt.ylabel(r'$(R_p/R_s)^2$ [ppm]')
    #plt.title('L98-59c Transit Spectrum from Marginalization')
    #plt.show()
    #sss
    #sys.exit()
    #centerramp=centerramp[1:-2]
    #depthramp=depthramp[1:-2]
    #errorramp=errorramp[1:-2]
    #plt.errorbar(centerramp, depthramp, errorramp,xerr=rampspread, fmt='o', color='g'
    #             , ecolor='g', ls='', label='Zhou Ramp Method')
    # drake=np.genfromtxt('HAT41_transit_drake_spec.dat')
    # centerdrake=drake[1:-2,0]*1e4
    # depthdrake=drake[1:-2,1]*1e6
    # errordrake=drake[1:-2,2]*1e6
    # plt.errorbar(centerdrake, depthdrake, errordrake, fmt='o', color='r'
    #              , ecolor='r', ls='', label='Drakes: Differential')

    #mcmc=pd.read_csv('./spectra.csv', index_col=[0, 1, 2]).loc[visit, 'mcmc', 6]
    #wave=mcmc['Central Wavelength'].values*1e4
    #waveerr=mcmc['Wavelength Range'].values*1e4
    #mdepth=mcmc['Depth'].values
    #merror=mcmc['Error'].values
    #plt.errorbar(wave[1:-2], mdepth[1:-2], yerr=merror[1:-2],xerr= waveerr[1:-2], fmt='o', color='red'
    #             , ecolor='red', ls='', label='MCMC')
    #plt.errorbar(center, depth, error,xerr=spread, fmt='o', color='b', ecolor='b'
    #             , ls='', label='Inflation, no adjustment')
    #plt.errorbar(center2, depth2, error2,xerr=spread2, fmt='o', color='orange', ecolor='orange'
    #             , ls='', label='No first orbit, no adjustment')
    #plt.errorbar(center1, depth1, error1,xerr=spread1, fmt='o', color='r', ecolor='r'
    #             , ls='', label='Inflation')
    #plt.errorbar(centerramp, depthramp, errorramp,xerr=rampspread, fmt='o', color='g'
    #             , ecolor='g', ls='', label='No inflation', alpha=.5)
    # plt.errorbar(cr, dr, er,xerr=rs, fmt='o', color='k', ecolor='k'
    #             , ls='', label='No first orbit, no inflation', alpha=.5)

    plt.legend(numpoints=1)
    plt.show()
    sys.exit()
    
    sav=readsav('nik.sav')
    lamb=sav.WAV_CEN_ANG*1e4
    lamb_err=sav.ERR_WAV_ANG*1e4
    depth=sav.TRANSIT_RPRS**2*1e6
    error=sav.TRANSIT_RPRS_ERR*sav.TRANSIT_RPRS*1e6*2
    plt.errorbar(lamb, depth, yerr=error,xerr= lamb_err, fmt='o', color='orange'
                 , ecolor='orange', ls='', label='Nikolay')
    df=pd.read_csv('tsarias2.csv')
    twave=df['wave'].values*1e4
    tdepth=df['depth'].values
    terror=df['error'].values
    plt.errorbar(twave, tdepth, terror, fmt='o', color='purple'
                 , ecolor='purple', ls='', label='Tsarias')
    
    plt.legend(numpoints=1)
    plt.show()
    sys.exit()
    #plt.savefig('avi.png')
    plt.show()
    
    
