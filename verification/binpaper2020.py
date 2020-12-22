from __future__ import print_function
import sys

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter

sys.path.append('./MCcubed')
import MCcubed as mc3

def comp_methods(visit, binsize, bin, wave):
    
    modelfile='../bin_analysis/bin_smooth2.csv'
    datafile='../bin_analysis/bin_data2.csv'
    pfile='../bin_analysis/bin_params2.csv'
    params=pd.read_csv(pfile, index_col=[0,1,2,3]).sort_index()
    data=pd.read_csv(datafile, index_col=[0,1,2]).sort_index()
    data=data.loc[(visit, binsize, bin)]
    params=params.loc[(visit, binsize, bin, 'Values')]
    smooth=pd.read_csv(modelfile, index_col=[0,1,2]).sort_index()
    smooth=smooth.loc[(visit, binsize, bin)]
   
    margdepth=params['Depth']*1e6
    margflux=data['Norm Flux'].values
    margerror=data['Norm Flux Error'].values
    margnflux=data['Flux'].values
    margnerror=data['Flux Error'].values  
    margresids=data['Residuals'].values*1e6
    margbinphase=data['Model Phase'].values
    margcor=data['Corrected Flux'].values
    margcorerr=data['Corrected Flux Error'].values

    margslope=(params['Slope'].values*margbinphase+1.0)*params['Zero-flux'].values
    margmodel=smooth['Model'].values
    margmodelx=smooth['Phase'].values
    try:
        margnorm=params['Zero-flux'].values[0]
    except AttributeError:
        margnorm=params['Zero-flux']
    try:
        margphoton_error=params['Photon Error'].values[0]
    except AttributeError:
        margphoton_error=params['Photon Error']
    marg_sys_model=margflux/margcor/margnorm
    marg_full_model=margflux-margresids*margnorm/1e6
    xmin=np.min(margbinphase)-0.02
    xmax=np.max(margbinphase)+.02

    margmod=data['Model'].values
    print(params.T)
    dfile='../bin_analysis/binmcmc_data.csv'
    pfile='../bin_analysis/binmcmc_params.csv'
    sfile='../bin_analysis/binmcmc_smooth.csv'


    binsmooth=pd.read_csv(sfile, index_col=[0,1,2]).sort_index()
    smooth=binsmooth.loc[(visit, binsize, bin)]
    data=pd.read_csv(dfile, index_col=[0,1,2]).sort_index()
    data=data.loc[(visit, binsize, bin)]
    params=pd.read_csv(pfile, index_col=[0,1,2,3]).sort_index()
    params=params.loc[(visit,binsize, bin, 'Values')]
    mcmcdepth=params['Depth']*1e6
    
    mcmcflux=data['Norm Flux'].values
    mcmcerror=data['Norm Flux Error'].values
    mcmcnflux=data['Flux'].values
    mcmcnerror=data['Flux Error'].values
    mcmcresids=data['Residuals'].values*1e6
    mcmcbinphase=data['Model Phase'].values
    mcmccor=data['Corrected Flux'].values
    mcmccorerr=data['Corrected Flux Error'].values
    mcmcslope=(params['Slope']*mcmcbinphase+1.0)*params['Zero-flux']
    mcmcmod=data['Model'].values
    print(params)
    try:
        mcmcphoton_error=params['Photon Error'].values[0]
    except AttributeError:
        mcmcphoton_error=params['Photon Error']
    try:
        mcmcnorm=params['Zero-flux'].values[0]
    except AttributeError:
        mcmcnorm=params['Zero-flux']
   
    mcmcmodel=smooth['Model'].values
    mcmcmodelx=smooth['Phase'].values
    mcmc_sys_model=mcmcflux/mcmccor
    mcmc_full_model=mcmcflux#-mcmcresids*mcmcnorm/1e6
    """f=plt.figure(figsize=(8,12))
    plt.subplot(311)
    plt.errorbar(margbinphase, margflux, margerror, color='b', ls='', marker='o',
                 ecolor='g', label='Error ')
    plt.plot(mcmcbinphase, mcmc_full_model, label='MCMC', color='g')
    plt.plot(margbinphase, marg_full_model, label='Marg', color='r')
    plt.xlim([xmin, xmax])
    plt.ylabel('Normalized Flux')
    plt.text(-0.02, 1.0, 'Raw Light Curve')
    #plt.text(0.0,.99, 'Error ')"""

    #plt.subplot(312)
    plt.errorbar(mcmcbinphase, mcmcmod, mcmccorerr, color='r', ls='', marker='o', ecolor='r', label='mcmc')
    #plt.errorbar(mcmcbinphase, mcmcresids/1e6, mcmccorerr, color='r', ls='', marker='o', ecolor='r', label='MCMC')
    #plt.plot(mcmcmodelx, mcmcmodel, color='b', label='MCMC')
    #plt.plot(mcmcbinphase, mcmc_sys_model, 'go', label='MCMC', ls='')
    #plt.plot(mcmcbinphase, mcmcslope, 'go', label='MCMC',ls='')
    plt.xlim([mcmcbinphase[0]-.01, mcmcbinphase[-1]+.01])
    #plt.plot(margmodelx, margmodel, color='r', label='Marg')
    #plt.plot(margbinphase, marg_sys_model, 'ro', label='Marg',ls='')
    #plt.plot(margbinphase, margslope, 'ro', label='Marg',ls='')
    plt.errorbar(margbinphase, margmod, margerror, color='g', ls='', marker='o',
                 ecolor='g', label='Marg')
    #plt.plot(margbinphase, np.zeros_like(margbinphase))
    plt.ylabel('Normalized Flux')
    #plt.text(-.02, 1.0, 'Systematics removed')
    #plt.text(.0, 0.000, 'Marg: %.1f' % (np.median(margresids[36:55])))
    #plt.text(.0, -0.001, 'MCMC: %.1f' % (np.median(mcmcresids[36:55])))
    plt.text(.0, 1-0.002, 'Marg-MCMC: %.1f' % (margdepth-mcmcdepth))
    plt.text(.0, 1-0.003, 'Norm dif: %.1f' % ((margnorm-mcmcnorm)*margdepth))
    plt.errorbar(margbinphase, margflux, margerror, color='b', ls='', marker='o',
                 ecolor='b', label='Data')
    #plt.text(-.2,.998, 'Error ')  
    plt.legend()
    plt.show()
    
    
    """flat=np.zeros_like(mcmcresids)
    mcmccorerr*=1e6
    margcorerr*=1e6
    plt.subplot(313)
    p3=plt.errorbar(mcmcbinphase, mcmcresids, mcmccorerr, color='g', ls='', marker='o'
                    , ecolor='g', label='Residuals')
    p3=plt.errorbar(margbinphase, margresids, margcorerr, color='r', ls='', marker='o'
                    , ecolor='r', label='Residuals')
    plt.xlim([xmin, xmax])
    plt.xlabel('Phase')
    plt.ylabel('Obs - Model [ppm]')
    plt.plot(mcmcbinphase, flat)
    plt.text(-.23, np.max(mcmcresids), 'Residuals')
    plt.legend()
    plt.show()"""

    


    
def bin_op(input, size, op='mean'):
    nbins=len(input)/size
    out=np.zeros(nbins)
    for i in range(nbins):
        start=size*i
        fin=size*(i+1)
        if op=='mean':
            out[i]=np.mean(input[start:fin])
        elif op=='sum':
            out[i]=np.sum(input[start:fin])
        elif op=='sqsum':
            out[i]=np.sqrt(np.sum(np.square(input[start:fin])))
    return out
def correlated(resids, wave, axes, color):
    #n=resids.shape[0]/9
    n=26
    rms=np.zeros(n)
    error=np.zeros(n)
    binsize=np.arange(n)+1
    nbins=np.zeros(n)
    #for i in range(n):
        #r=bin_op(resids, i+1)
        #nbins[i]=len(r)
        #rms[i]=np.std(r)
        #error[i]=rms[i]/np.sqrt(2*nbins[i])
        
    #expected=rms[0]/np.sqrt(binsize)*np.sqrt(nbins/(nbins-1))

    rms, rmslo, rmshi, expected, binsize=mc3.rednoise.binrms(resids,n)

    
    significant=np.where(rms/expected -1 > 2*rmslo/expected)[0]
    print(significant)
    if len(significant) == 0:
        max_beta=1.0
    else:
        max_beta = np.max(rms[significant]/expected[significant])
        ind = np.argmax(rms[significant]/expected[significant])
        print (rms[significant[ind]]/expected[significant[ind]]- 
               rmslo[significant[ind]]/expected[significant[ind]])

    ax.plot(binsize, expected/rms[0],color='black', label='Expected')
    ax.errorbar(binsize, rms/rms[0], yerr=[rmslo/rms[0], rmshi/rms[0]]
                , color=color, label='Data RMS')
    ax.set_xscale('log')
    ax.set_yscale('log')
   
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.get_xaxis().get_major_formatter().labelOnlyBase = False
    ax.yaxis.set_major_formatter(ScalarFormatter())
    #ax.minorticks_off()
    ax.set_yticks([.1,1])
    ax.set_xticks([1,2,3,4,5,6,7,8,9])
    #a.xscale('log')
    #plt.yscale('log')
    label = '%.03f $\mu$m' % wave
    if wave ==0:
        label = 'Band-integrated'
    ax.text(.05, .2, label, transform=ax.transAxes)
    #ax.text(1, .2, r'$\beta_{max}$ = %.03f' % max_beta)
    #max_beta=rms[0]/expected[0]
    return max_beta

def adtest(resids, photon_error, norm=False):

  """ Eventually save AD to params? Or something. Maybe it's own CSV with all of this info."""
  if norm:
      st.probplot(resids, plot=plt)
      plt.show()
  shapiro=st.shapiro(resids)
  pearson= st.normaltest(resids)
  #A-D test

  # First, get CDF of data
  nres=len(resids)
  res=np.sort(resids)
  num=np.ones(nres)/nres
  cdf1=np.cumsum(num)
  
  # Case 3: we do not know mean or sigma of distribution. Determine from residuals
  # Test if it is gaussian, just with inflated errors (aka no red noise)
  avg_3=np.mean(res)
  sig_3=np.std(res)
    
  # Case 0: We "know" the distribution is photon noise gaussian centered on 0. Test for accurate noise
  avg_0=0
  sig_0=photon_error

  # Normalize (see wikipedia)
  data_0=(res-avg_0)/sig_0
  data_3=(res-avg_3)/sig_3
  
  # Get gaussian CDFs with corresponding mean and sigma
  cdf_0=st.norm.cdf(data_0)
  cdf_3=st.norm.cdf(data_3)

  # Get continuous gaussian CDFs for plotting
  gauss=np.arange(180)/30. - 3
  gauss_cdf=st.norm.cdf(gauss)
  # Get continuous, unnormalized perfectly gaussian residuals for plotting
  gauss_resids_0 = sig_0*gauss + avg_0
  gauss_resids_3 = sig_3*gauss + avg_3
   
    
  # Calculate A-D number
  sum_0=0
  sum_3=0
  for j in range(1, nres):
    # First quoted number vs photon error
    sum_0+=(2*j-1)*(np.log(cdf_0[j-1])+np.log(1.0-cdf_0[nres-j]))
    sum_3+=(2*j-1)*(np.log(cdf_3[j-1])+np.log(1.0-cdf_3[nres-j]))
    # Then comparison to any gaussian
    
  ad_0=-nres-sum_0/nres
  ad_3=-nres-sum_3/nres
  ad_3*=(1+4./nres-25./nres/nres)

  # Save all plotting stuff to somewhere? probably same place as residuals?
  print('Compared to theory-limited: %f' % ad_0)
  print('Compared to Gaussian: %f' % ad_3)
  print('Shapiro p-value: %f' % shapiro[1])
  print('Pearson p-value: %f' % pearson[1])
  
  return res, cdf1, gauss_resids_0, gauss_resids_3, gauss_cdf


def binpaper(visit, binsize, bin, wave, method='marg'):

  """ This puts all information about quality of fit for 
  one visit into a few nice figures.

  Right now I toggle correlated/adtest/binpaper plot. In future make these inputs."""
  if method=='marg':
    modelfile='../bin_analysis/bin_smooth.csv'
    datafile='../bin_analysis/bin_data.csv'
    pfile='../bin_analysis/bin_params.csv'
    params=pd.read_csv(pfile, index_col=[0,1,2,3]).sort_index()
    data=pd.read_csv(datafile, index_col=[0,1,2]).sort_index()
    data=data.loc[(visit, binsize, bin)]
    params=params.loc[(visit, binsize, bin, 'Values')]
    smooth=pd.read_csv(modelfile, index_col=[0,1,2]).sort_index()
    smooth=smooth.loc[(visit, binsize, bin)]
   
    
    flux=data['Norm Flux'].values
    error=data['Norm Flux Error'].values
    nflux=data['Flux'].values
    nerror=data['Flux Error'].values  
    resids=data['Residuals'].values*1e6
    binphase=data['Model Phase'].values
    cor=data['Corrected Flux'].values
    corerr=data['Corrected Flux Error'].values
    
    model=smooth['Model'].values
    modelx=smooth['Phase'].values
    try:
        norm=params['Zero-flux'].values[0]
    except AttributeError:
        norm=params['Zero-flux']
    try:
        photon_error=params['Photon Error'].values[0]
    except AttributeError:
        photon_error=params['Photon Error']
    sys_model=flux/cor/norm
    full_model=flux-resids*norm/1e6
    xmin=np.min(binphase)-0.02
    xmax=np.max(binphase)+.02


  elif method=='ramp':
    dfile='../bin_analysis/binramp_data.csv'
    pfile='../bin_analysis/binramp_params.csv'
    sfile='../bin_analysis/binramp_smooth.csv'
    binsmooth=pd.read_csv(sfile, index_col=[0,1,2]).sort_index()
    smooth=binsmooth.loc[(visit, binsize, bin)]
    data=pd.read_csv(dfile, index_col=[0,1,2]).sort_index()
    data=data.loc[(visit, binsize, bin)]
    params=pd.read_csv(pfile, index_col=[0,1,2,3]).sort_index()
    params=params.loc[(visit,binsize, bin, 'Values')]
    
    
    flux=data['Norm Flux'].values
    error=data['Norm Flux Error'].values
    nflux=data['Flux'].values
    nerror=data['Flux Error'].values
    resids=data['Residuals'].values*1e6
    binphase=data['Model Phase'].values
    cor=data['Corrected Flux'].values
    corerr=data['Corrected Flux Error'].values
    try:
        photon_error=params['Photon Error'].values[0]
    except AttributeError:
        photon_error=params['Photon Error']
    try:
        norm=params['Zero-flux'].values[0]
    except AttributeError:
        norm=params['Zero-flux']
   
    model=smooth['Model'].values
    modelx=smooth['Phase'].values
    sys_model=flux/cor/norm
    full_model=flux-resids*norm/1e6
    xmin=np.min(binphase)-0.02
    xmax=np.max(binphase)+.02
  elif method=='mcmc':
    dfile='../bin_analysis/binmcmc_data.csv'
    pfile='../bin_analysis/binmcmc_params.csv'
    sfile='../bin_analysis/binmcmc_smooth.csv'

    
    binsmooth=pd.read_csv(sfile, index_col=[0,1,2]).sort_index()
    smooth=binsmooth.loc[(visit, binsize, bin)]
    data=pd.read_csv(dfile, index_col=[0,1,2]).sort_index()
    data=data.loc[(visit, binsize, bin)]
    params=pd.read_csv(pfile, index_col=[0,1,2,3]).sort_index()
    params=params.loc[(visit,binsize, bin, 'Values')]
    
    
    flux=data['Norm Flux'].values
    error=data['Norm Flux Error'].values
    nflux=data['Flux'].values
    nerror=data['Flux Error'].values
    resids=data['Residuals'].values*1e6
    binphase=data['Model Phase'].values
    cor=data['Corrected Flux'].values
    corerr=data['Corrected Flux Error'].values
    try:
        photon_error=params['Photon Error'].values[0]
    except AttributeError:
        photon_error=params['Photon Error']
    try:
        norm=params['Zero-flux'].values[0]
    except AttributeError:
        norm=params['Zero-flux']
   
    model=smooth['Model'].values
    modelx=smooth['Phase'].values
    sys_model=flux/cor/norm
    full_model=flux-resids*norm/1e6
    xmin=np.min(binphase)-0.02
    xmax=np.max(binphase)+.02

  f=plt.figure(figsize=(8,12))
  plt.subplot(311)
  plt.errorbar(binphase, flux, error, color='b', ls='', marker='o',
               ecolor='g', label='Error ')
  plt.plot(binphase, full_model)
  plt.xlim([xmin, xmax])
  plt.ylabel('Normalized Flux')
  plt.text(-0.02, 1.0, 'Raw Light Curve')
  #plt.text(0.0,.99, 'Error ')

  plt.subplot(312)
  plt.errorbar(binphase, cor, corerr, color='b', ls='', marker='o', ecolor='purple')
  plt.plot(modelx, model)
  plt.xlim([xmin, xmax])
  plt.ylabel('Normalized Flux')
  plt.text(-.02, 1.0, 'Systematics removed')
  #plt.text(-.2,.998, 'Error ')  

  
  flat=np.zeros_like(resids)
  corerr*=1e6
  plt.subplot(313)
  p3=plt.errorbar(binphase, resids, corerr, color='r', ls='', marker='o'
                  , ecolor='red', label='Residuals')
  plt.xlim([xmin, xmax])
  plt.xlabel('Phase')
  plt.ylabel('Obs - Model [ppm]')
  plt.plot(binphase, flat)
  plt.text(-.23, np.max(resids), 'Residuals')

  std_res=np.std(resids)
  std_err=std_res/np.sqrt(2*len(resids))
  ratio=std_res/photon_error
  # ''photon error'' is really theory limit for bins, and should be same as flux_error/flux*1e6

  #plt.text(-.2,np.min(resids)+100, 'RMS: %03d +- %03d' % (std_res, std_err))
  plt.text(.01,np.min(resids)+100, 'RMS/photon: %.3f +- %.3f' % (ratio, std_err/photon_error))

  plt.title('%.03f $\mu$m' % wave, size=12)
  plt.savefig('bin_lightcurves'+method+'.png')
  savename='binpaper_'+visit.replace('/','_')+'_'+method+'.pdf'
  savename='bin%03d_outlier.pdf' % bin
  #f.savefig(savename)
  #f.clf()
  #plt.close(f)
  #plt.savefig('bin_resids_cdf'+method+'.png')
  plt.show()
  """res, cdf1, gauss_resids_0, gauss_resids_3, gauss_cdf = adtest(resids, photon_error)
  plt.subplot(414)
  plt.plot(res, cdf1, 'ro', label='Residuals')
  plt.plot(gauss_resids_0, gauss_cdf, label='Theoretical noise limit Gaussian')
  plt.plot(gauss_resids_3, gauss_cdf, 'purple', label='Gaussian')
  plt.legend(numpoints=1)
  #plt.show()
  figure=plt.gcf()
  figure.set_size_inches(12, 10)
  plt.savefig('wlpaper_'+visit.replace('/','_')+'_'+method+'.pdf', )

  #plt.clf()
  plt.plot(res, cdf1, 'ro', label='Residuals')
  plt.plot(gauss_resids_0, gauss_cdf, label='Theoretical noise limit Gaussian')
  plt.plot(gauss_resids_3, gauss_cdf, 'purple', label='Gaussian')
  plt.legend(numpoints=1)
  plt.xlabel('Residuals [PPM]')
  plt.ylabel('CDF')
  plt.savefig('bin_resids_cdf.png')"""

def binvis(visit, binsize, wave, method='marg'):
    
    if method=='marg':
        modelfile='../bin_analysis/bin_smooth2.csv'
        datafile='../bin_analysis/bin_data2.csv'
        data=pd.read_csv(datafile, index_col=[0,1,2]).sort_index()
        data=data.loc[(visit, binsize)]
        smooth=pd.read_csv(modelfile, index_col=[0,1,2]).sort_index()
        smooth=smooth.loc[(visit, binsize)]


        # chi squared stuff
        pfile='../bin_analysis/bin_params2.csv'
        params=pd.read_csv(pfile, index_col=[0,1,2,3]).sort_index()
        params=params.loc[(visit, binsize)]

        #flux=data['Norm Flux'].values
        #error=data['Norm Flux Error'].values
        #nflux=data['Flux'].values
        #nerror=data['Flux Error'].values  
        #resids=data['Residuals'].values*1e6
        fig = plt.figure(figsize=(7, 12))
        colors = iter(cm.inferno(np.linspace(0.1, .8, len(spec))))
        start_bin = 0
        #start_bin = 14
        #end_bin=len(spec)
        end_bin=14
        for i in range(start_bin):
            c=next(colors)
        for i in range(end_bin-start_bin):
            c = next(colors)
            dat = data.loc[start_bin]
            smoo = smooth.loc[start_bin]
        
            binphase=dat['Model Phase'].values
            cor=dat['Corrected Flux'].values
            corerr=dat['Corrected Flux Error'].values
            #print np.median(corerr)*1e6
            model=smoo['Model'].values
            modelx=smoo['Phase'].values

            # Get reduced chi-squared

            mresids=data.loc[start_bin,'Residuals'].values
            errors = data.loc[start_bin, 'Corrected Flux Error'].values
            #errors = errors/1.1
            nfree = (params.loc[(start_bin, 'Errors'),'rprs':'WL Coeff'].values != 0).sum()
            dof = len(mresids) - nfree
            chi2 = np.sum(mresids*mresids/errors/errors)
            rchi2 = chi2/dof
            print(rchi2)
            # Plot
            xmin=np.min(binphase)-0.005
            xmax=np.max(binphase)+.02
            con = 0.005
            plt.errorbar(binphase, cor-i*con, corerr, color=c,
                         ls='', marker='o', ecolor=c, markersize = 3)
            plt.plot(modelx, model-i*con, color=c)
            plt.xlim([xmin, xmax])
            plt.ylabel('Normalized Flux - Constant')
            plt.xlabel('Orbital Phase')
            plt.text(.045, 1.002-i*con, r'%.2f$\mu$m' % wave[start_bin], color=c)
            plt.text(-.06, 1.002-i*con, r'$\chi^2_{red}$=%.2f' % rchi2, color=c)
            start_bin += 1
    
        plt.savefig('../../hat41_bincurves1.pdf')
        #plt.show()
    elif method=='mcmc':
        dfile='../bin_analysis/binmcmc_data.csv'
        pfile='../bin_analysis/binmcmc_params.csv'
        sfile='../bin_analysis/binmcmc_smooth.csv'
        binsmooth=pd.read_csv(sfile, index_col=[0,1,2]).sort_index()
        smooth=binsmooth.loc[(visit, binsize)]
        data=pd.read_csv(dfile, index_col=[0,1,2]).sort_index()
        data=data.loc[(visit, binsize)]
        params=pd.read_csv(pfile, index_col=[0,1,2,3]).sort_index()
        params=params.loc[(visit,binsize, 'Values')]
        
    
        flux=data['Norm Flux'].values
        error=data['Norm Flux Error'].values
        nflux=data['Flux'].values
        nerror=data['Flux Error'].values
        resids=data['Residuals'].values*1e6
        binphase=data['Model Phase'].values
        cor=data['Corrected Flux'].values
        corerr=data['Corrected Flux Error'].values
        try:
            photon_error=params['Photon Error'].values[0]
        except AttributeError:
            photon_error=params['Photon Error']
        try:
            norm=params['Zero-flux'].values[0]
        except AttributeError:
            norm=params['Zero-flux']
   
        model=smooth['Model'].values
        modelx=smooth['Phase'].values
        sys_model=flux/cor/norm
        full_model=flux-resids*norm/1e6
        xmin=np.min(binphase)-0.02
        xmax=np.max(binphase)+.02

        fig = plt.figure(figsize=(7, 12))
        colors = iter(cm.inferno(np.linspace(0.1, .8, len(spec))))
        bin = 0
        for i in range(len(spec)):
            c = next(colors)
            dat = data.loc[bin]
            smoo = smooth.loc[bin]
        
            binphase=dat['Model Phase'].values
            cor=dat['Corrected Flux'].values
            corerr=dat['Corrected Flux Error'].values
            #print np.median(corerr)*1e6
            model=smoo['Model'].values
            modelx=smoo['Phase'].values

            # Get reduced chi-squared

            mresids=data.loc[i,'Residuals'].values
            errors = data.loc[i, 'Corrected Flux Error'].values
            #errors = errors/1.1
            nfree = (params.loc[(i, 'Errors'),'rprs':'WL Coeff'].values != 0).sum()
            dof = len(mresids) - nfree
            chi2 = np.sum(mresids*mresids/errors/errors)
            rchi2 = chi2/dof
            print(rchi2)
            # Plot
            xmin=np.min(binphase)-0.005
            xmax=np.max(binphase)+.01
            con = 0.003
            plt.errorbar(binphase, cor-i*con, corerr, color=c,
                         ls='', marker='o', ecolor=c, markersize = 3)
            plt.plot(modelx, model-i*con, color=c)
            plt.xlim([xmin, xmax])
            plt.ylabel('Normalized Flux - Constant')
            plt.xlabel('Orbital Phase')
            plt.text(.03, 1.00012-i*con, r'%.2f$\mu$m' % wave[bin], color=c)
            plt.text(-.05, 1.0005-i*con, r'$\chi^2_{red}$=%.2f' % rchi2, color=c)
            bin += 1
    
        #plt.savefig('../../l9859c_bincurves.pdf')
        plt.show()


    ### HERE: need to make binvis and bincorrelated work for both new bin_data2 and for mcmc stuff.
    ## Hopefully, this will make difference in depths more obvious. Probably the slope though.
    ## I would like the prove that ramp misses slope with just a linear fit, since first two orbits
    ## are flat. Can also look more at acor_resids. Can also test ramp with quad or log slope or something.

    ## Finally, calculate evidences and chi squareds for both too.
    elif method=='ramp':
        dfile='../bin_analysis/binramp_data.csv'
        pfile='../bin_analysis/binramp_params.csv'
        sfile='../bin_analysis/binramp_smooth.csv'
        binsmooth=pd.read_csv(sfile, index_col=[0,1,2]).sort_index()
        smooth=binsmooth.loc[(visit, binsize, bin)]
        data=pd.read_csv(dfile, index_col=[0,1,2]).sort_index()
        data=data.loc[(visit, binsize, bin)]
        params=pd.read_csv(pfile, index_col=[0,1,2,3]).sort_index()
        params=params.loc[(visit,binsize, bin, 'Values')]
        
    
        flux=data['Norm Flux'].values
        error=data['Norm Flux Error'].values
        nflux=data['Flux'].values
        nerror=data['Flux Error'].values
        resids=data['Residuals'].values*1e6
        binphase=data['Model Phase'].values
        cor=data['Corrected Flux'].values
        corerr=data['Corrected Flux Error'].values
        try:
            photon_error=params['Photon Error'].values[0]
        except AttributeError:
            photon_error=params['Photon Error']
        try:
            norm=params['Zero-flux'].values[0]
        except AttributeError:
            norm=params['Zero-flux']
   
        model=smooth['Model'].values
        modelx=smooth['Phase'].values
        sys_model=flux/cor/norm
        full_model=flux-resids*norm/1e6
        xmin=np.min(binphase)-0.02
        xmax=np.max(binphase)+.02
        


    #plt.subplot(312)

    #plt.text(-.2,.998, 'Error ')  
    
  
    """ flat=np.zeros_like(resids)
    corerr*=1e6
    plt.subplot(313)
    p3=plt.errorbar(binphase, resids, corerr, color='r', ls='', marker='o'
    , ecolor='red', label='Residuals')
    plt.xlim([xmin, xmax])
    plt.xlabel('Phase')
    plt.ylabel('Obs - Model [ppm]')
    plt.plot(binphase, flat)
    plt.text(-.23, np.max(resids), 'Residuals')
    
    std_res=np.std(resids)
    std_err=std_res/np.sqrt(2*len(resids))
    ratio=std_res/photon_error
    # ''photon error'' is really theory limit for bins, and should be same as flux_error/flux*1e6
    
    #plt.text(-.2,np.min(resids)+100, 'RMS: %03d +- %03d' % (std_res, std_err))
    plt.text(.01,np.min(resids)+100, 'RMS/photon: %.3f +- %.3f' % (ratio, std_err/photon_error))"""
    
    #plt.title('%.03f $\mu$m' % wave, size=12)
    #plt.savefig('bin_lightcurves'+method+'.png')
    savename='binpaper_'+visit.replace('/','_')+'_'+method+'.pdf'
    savename='bin%03d_outlier.pdf' % start_bin
    #f.savefig(savename)
    #f.clf()
    #plt.close(f)
    #plt.savefig('bin_resids_cdf'+method+'.png')  
  
if __name__=='__main__':
    visit=sys.argv[1]+'/'+sys.argv[2]+'/'+sys.argv[3]
    binsize=int(sys.argv[4])
    bin=int(sys.argv[5])
    if len(sys.argv) == 7:
        save=bool(int(sys.argv[6]))
    else:
        save=False
    print(save)
    datafile='../bin_analysis/bin_data2.csv'
    rampfile='../bin_analysis/binramp_data.csv'
    spectra='../bin_analysis/spectra.csv'
    pfile='../bin_analysis/bin_params2.csv'
    phot_error=pd.read_csv(pfile, index_col=[0,1,2,3]).sort_index()
    try:
        phot_error=phot_error.loc[(visit, binsize, bin, 'Values')
                                  , 'Photon Error'].values[0]
    except AttributeError:
        phot_error=phot_error.loc[(visit, binsize, bin, 'Values')
                                  , 'Photon Error']

    sp=pd.read_csv(spectra, index_col=[0,1,2]).sort_index()
    spec=sp.loc[(visit, 'marg', binsize), 'Central Wavelength'].values
    wave=spec[bin]

    #nbin=29
    #ratio_marg=np.zeros(nbin)
    #ratio_ramp=np.zeros(nbin)
    #for i in range(nbin):
    #    print i

    
    #binvis(visit, binsize, spec, method='marg')
    #comp_methods(visit, binsize, bin, wave)
    #sys.exit()
    #binpaper(visit, binsize, bin, wave, method='mcmc')
    #binpaper(visit, binsize, bin, wave, method='ramp')

    #datafile='../bin_analysis/binmcmc_data.csv'
    #pfile='../bin_analysis/binmcmc_params.csv'
    
    marg=pd.read_csv(datafile, index_col=[0,1,2]).sort_index()
    marg=marg.loc[(visit, binsize)]

    nrow=int(np.ceil(len(spec)/3.))
    f, axes=plt.subplots(nrow, 3, sharex='col', sharey='row', figsize=(12,12))
    ax=f.add_subplot(111, frame_on=False)
    ax.set_xlabel('Exposures Per Bin', labelpad=30, fontsize=15)
    ax.set_ylabel('Normalized RMS', labelpad=40, fontsize=15)
    s=visit.split('/')
    #ax.set_title('WASP-19b Correlated Noise Analysis\n', fontsize=18)
    #ax.set_title('%s %s %s Correlated Noise Analysis\n' % (s[0],s[1],s[2]) , fontsize=18)
    #ax.set_title('Marginalization Correlated Noise Analysis\n', fontsize=18)
    ax.tick_params(bottom=False, labelbottom=False, top=False, labeltop=False
                   ,left=False, labelleft=False,  right=False, labelright=False)
    plt.subplots_adjust(hspace=0)
    plt.subplots_adjust(wspace=0)
    colors = iter(cm.rainbow(np.linspace(0.2, .9, len(np.ravel(axes)))))
    beta=np.empty(len(spec))

    # Get free params
    params=pd.read_csv(pfile, index_col=[0,1,2,3]).sort_index()
    params=params.loc[(visit, binsize)]
   
    for i, ax in enumerate(np.ravel(axes)):
        if i == len(spec):
            #ax.set_xscale('log')
            #ax.set_yscale('log')
   
            #ax.xaxis.set_major_formatter(ScalarFormatter())
            #ax.get_xaxis().get_major_formatter().labelOnlyBase = False
            #ax.yaxis.set_major_formatter(ScalarFormatter())
            #ax.minorticks_off()
            #ax.set_yticks([.1,1])
            #ax.set_xticks([1,2,3,4,5,6,7,8,9])
            modelfile='../wl_preprocess/wl_models_info.csv'
            model_info=pd.read_csv(modelfile, index_col=[0,1]).loc[visit]
            best_model=model_info.loc['Weight'].iloc[:-1].astype(float).idxmax()
            best= model_info.loc[:,best_model]
            resids=best.loc['Residuals'].values*1e6
            correlated(resids, 0, ax, 'grey')
            break
        mresids=marg.loc[i,'Residuals'].values
        errors = marg.loc[i, 'Corrected Flux Error'].values
        nfree = (params.loc[(i, 'Errors'),'rprs':'WL Coeff'].values != 0).sum()
        dof = len(mresids) - nfree
        chi2 = np.sum(mresids*mresids/errors/errors)
        rchi2 = chi2/dof
        print('Bin %2d' % i)
        print('Chi squared:  %.2f' % chi2)
        print('Reduced Chi squared:  %.2f' % rchi2)
        print('DOF:  %d' % dof)
        
        wave=spec[i]
        adtest(mresids, phot_error/1e6)
        beta[i]=correlated(mresids, wave, ax, next(colors))
    
  
    name='rednoise_'+visit.replace('/','_')+'_marg.pdf'
    name = '../../rednoise_'+visit.replace('/','_')+'_marg.pdf'
    axes[0,0].legend()
    save=True
    if save==True:
        f.savefig(name)
        f.clf()
        plt.close(f)
    else:
        plt.show()
        #f.close()
    sys.exit()

    #sp.loc[(visit, 'marg', binsize), 'Beta Max']=beta
    #sp.to_csv('../bin_analysis/spectra.csv', index_label=['Obs', 'Method', 'Bin Size'])
    print(beta)
    print(np.mean(beta))
    print(np.median(beta))
    
    rspec=sp.loc[(visit, 'ramp', binsize), 'Central Wavelength'].values
    rwave=rspec[bin]
    ramp=pd.read_csv(rampfile, index_col=[0,1,2]).sort_index()
    ramp=ramp.loc[(visit, binsize)]

    f, axes=plt.subplots(nrow, 3, sharex='col', sharey='row', figsize=(12,12))
    ax=f.add_subplot(111, frame_on=False)
    ax.set_xlabel('Points Per Bin', labelpad=30, fontsize=15)
    ax.set_ylabel('Normalized RMS', labelpad=40, fontsize=15)
    s=visit.split('/')
    #ax.set_title('%s %s %s Correlated Noise Analysis\n' % (s[0],s[1],s[2]) , fontsize=18)
    ax.set_title('Ramp Correlated Noise Analysis\n' , fontsize=18)
    ax.tick_params(bottom=False, labelbottom=False, top=False, labeltop=False
                   ,left=False, labelleft=False,  right=False, labelright=False)
    plt.subplots_adjust(hspace=0)
    plt.subplots_adjust(wspace=0)
    colors = iter(cm.rainbow(np.linspace(0.2, .9, len(np.ravel(axes)))))
    beta=np.empty(len(spec))

    for i, ax in enumerate(np.ravel(axes)):
        if i == len(rspec):
            break
        rresids=ramp.loc[i,'Residuals'].values
        wave=rspec[i]
        adtest(rresids, phot_error/1e6)
        beta[i]=correlated(rresids, wave, ax, next(colors))
        name='rednoise_'+visit.replace('/','_')+'_ramp.pdf'
        print(beta)
        print(np.mean(beta))
        print(np.median(beta))
        axes[0,0].legend(loc=3)
    if save==True:
        f.savefig(name)
        f.clf()
        plt.close(f)
    else:
        f.show()
    sys.exit()
    sp.loc[(visit, 'ramp', binsize), 'Beta Max']=beta
    sp.to_csv('../bin_analysis/spectra.csv', index_label=['Obs', 'Method', 'Bin Size'])





