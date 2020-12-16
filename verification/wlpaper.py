import sys


import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

sys.path.append('./MCcubed')
import MCcubed as mc3

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

def correlated(resids, name):
    #n=resids.shape[0]/
    n=9
    print len(resids)
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
    print significant
    if len(significant) == 0:
        max_beta=1.0
    else:
        max_beta = np.max(rms[significant]/expected[significant])
        ind = np.argmax(rms[significant]/expected[significant])
        print (rms[significant[ind]]/expected[significant[ind]]- 
               rmslo[significant[ind]]/expected[significant[ind]])

    plt.plot(binsize, expected/rms[0],color='black', label='Expected')
    plt.errorbar(binsize, rms/rms[0], yerr=[rmslo/rms[0], rmshi/rms[0]]
                , color='b', label='Data RMS')
    #plt.xscale('log')
   # plt.yscale('log')
    #ax.minorticks_off()
    #ax.set_yticks([.1,1])
    #a.xscale('log')
    #plt.yscale('log')
    #ax.text(1, .2, '%.03f $\mu$m' % wave)
    #ax.text(1, .2, r'$\beta_{max}$ = %.03f' % max_beta)
    #max_beta=rms[0]/expected[0]
    plt.show()
    return max_beta


def wlcorrelated(resids, phase, error, name=None):
    #n=len(resids)
    n=9
    #phot_err=np.zeros(n)
    rms=np.zeros(n)
    binsize=np.arange(n)+1
    for i in range(n):
        #f=bin_op(flux, i+1, op='sum')
        #e=bin_op(error, i+1, op='sqsum')
        r=bin_op(resids, i+1)
        p = bin_op(phase, i+1)
        #phot_err[i]=1e6*np.median(e/f)
        rms[i]=np.std(r)
        plt.plot(p, r, 'bo', ls='')
        plt.show()
    sys.exit()
    expected=rms[0]/np.sqrt(binsize)
    f=plt.figure()
    plt.plot(binsize, expected ,color='black', label='Expected')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Exposures per bin')
    plt.ylabel('RMS [ppm]')
    plt.title('Whitelight Correlated Noise Test')
    plt.plot(binsize, rms,'b', 'Actual')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(numpoints=1)
    f.show()

    if name:
        pass
      #f.savefig('wlcorrelated_'+name+'.png')
    else:
      f.show()

def adtest(resids, photon_error):

  """ Eventually save AD to params? Or something. Maybe it's own CSV with all of this info."""

  #st.probplot(resids, plot=plt)
  #plt.show()
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
  print 'Compared to theory-limited: %f' % ad_0
  print 'Compared to Gaussian: %f' % ad_3
  print 'Shapiro p-value: %f' % shapiro[1]
  print 'Pearson p-value: %f' % pearson[1]
  
  return res, cdf1, gauss_resids_0, gauss_resids_3, gauss_cdf


def wlpaper(visit, method='marg'):

  """ This puts all information about quality of fit for 
  one visit into a few nice figures"""
  if method=='marg':
    modelfile='../wl_preprocess/wl_models_info.csv'
    datafile='../wl_preprocess/wl_data.csv'
    model_info=pd.read_csv(modelfile, index_col=[0,1]).loc[visit]
    best_model=model_info.loc['Weight'].iloc[:-1].astype(float).idxmax()
    best= model_info.loc[:,best_model]
    resids=best.loc['Residuals'].values*1e6
    data=pd.read_csv(datafile, index_col=[0,1]).loc[visit]
    
    flux=data.loc['data','Normalized Flux'].values
    error=data.loc['data','Normalized Error'].values
    nflux=data.loc['data','Flux'].values
    nerror=data.loc['data','Error'].values
    photon_error=data.loc['photon err','Values']
    
    best_model=model_info.loc['Weight'].iloc[:-1].astype(float).idxmax()
    best= model_info.loc[:,best_model]
    nfree=(model_info.loc['Params Errors', best_model] != 0.0).sum()
    dof = len(flux) - nfree
    resids=best.loc['Residuals'].values*1e6
    wlphase=best.loc['Corrected Phase'].values
    cor=best.loc['Corrected Flux'].values
    corerr=best.loc['Corrected Error'].values
    model=best.loc['Smooth Model'].values
    modelx=best.loc['Smooth Model Phase'].values
    norm=best.loc['Params'].values[1]
    sys_model=flux/cor/norm
    full_model=flux-resids*norm/1e6
    xmin=np.min(wlphase)-0.02
    xmax=np.max(wlphase)+.02

    #resids=resids[1:]
    #res, cdf1, gauss_resids_0, gauss_resids_3, gauss_cdf = adtest(resids, photon_error)
  elif method=='ramp':
    dfile='../wl_preprocess/wl_ramp_data.csv'
    pfile='../wl_preprocess/wl_ramp_params.csv'
    sfile='../wl_preprocess/wl_ramp_smooth.csv'
    wlsmooth=pd.read_csv(sfile, index_col=0)
    smooth=wlsmooth.loc[visit]
    data=pd.read_csv(dfile, index_col=0).loc[visit]
    
    
    flux=data['Norm Flux'].values
    error=data['Norm Flux Error'].values
    nflux=data['Flux'].values
    nerror=data['Flux Error'].values
    params=pd.read_csv(pfile, index_col=[0,1]).loc[(visit,'Values')]
    
    photon_error=params['Photon Error']
    norm=params['Zero-flux']
    resids=data['Residuals'].values*1e6
    
    wlphase=data['Model Phase'].values+.5
    cor=data['Corrected Flux'].values
    corerr=data['Corrected Flux Error'].values
    model=smooth['Model'].values
    modelx=smooth['Phase'].values+.5
    sys_model=flux/cor/norm
    full_model=flux-resids*norm/1e6
    xmin=np.min(wlphase)-0.02
    xmax=np.max(wlphase)+.02

  f=plt.figure(figsize=(8,12))
  plt.subplot(311)
  plt.errorbar(wlphase, flux, error, color='w', ls='', marker='o',
               ecolor='w', label='Error ', markersize=6, markeredgecolor='b')
  #plt.plot(wlphase, full_model)
  plt.xlim([xmin, xmax])
  plt.ylabel('Normalized Flux')
  plt.text(-.055, 0.9975, '(a)')
  #plt.text(.48,.999, 'Error ')  

  plt.subplot(312)
  plt.errorbar(wlphase, cor, corerr, color='w', ls='', marker='o', ecolor='w'
               , markersize=6, alpha=.8, markeredgecolor='b')
  plt.plot(modelx, model, 'k')
  plt.xlim([xmin, xmax])
  plt.ylabel('Normalized Flux')
  plt.text(-.055, .9975, '(b)')
  #plt.text(.42,1.0, 'Error ')  
  
  
  flat=np.zeros_like(resids)
  corerr*=1e6
  plt.subplot(313)
  p3=plt.errorbar(wlphase, resids, corerr, color='w', ls='', marker='o'
                  , ecolor='blue', label='Residuals', markersize=6, elinewidth=.7,markeredgecolor='b')
  plt.xlim([xmin, xmax])
  plt.xlabel('Phase')
  plt.ylabel('Obs - Model [ppm]')
  plt.plot([wlphase[0]-.06, wlphase[-1]+.06], [0,0], 'k')
  plt.text(-.055, 350, '(c)')

  std_res=np.std(resids)
  ratio=std_res/photon_error

  rchi2=np.sum(resids*resids/corerr/corerr)/dof
  print rchi2
  #plt.text(.52,-500, 'RMS: %03d ppm' % std_res)
  #rchi2=1.1
  plt.text(.045,-600, r'$\chi^2_{red} = %.2f$' % rchi2)
  #plt.text(.52,-700, 'RMS/photon: %03f' % ratio)
  #plt.show()
  savename='../../wlpaper_'+visit.replace('/','_')+'_'+method+'.pdf'
  #plt.show()
  plt.savefig(savename)
  #f.clf()
  res, cdf1, gauss_resids_0, gauss_resids_3, gauss_cdf = adtest(resids, photon_error)
  #sys.exit()
  #plt.subplot(414)
  #plt.plot(res, cdf1, 'ro', label='Residuals')
  #plt.plot(gauss_resids_0, gauss_cdf, label='Theoretical noise limit Gaussian')
  #plt.plot(gauss_resids_3, gauss_cdf, 'purple', label='Gaussian')
  #plt.legend(numpoints=1)
  #plt.show()
  #figure=plt.gcf()
  #figure.set_size_inches(12, 10)
  #plt.savefig('wlpaper_'+visit.replace('/','_')+'_'+method+'.pdf', )

  plt.clf()
  plt.plot(res, cdf1, 'ro', label='Residuals')
  plt.plot(gauss_resids_0, gauss_cdf, label='Theoretical noise limit Gaussian')
  plt.plot(gauss_resids_3, gauss_cdf, 'purple', label='Gaussian')
  plt.legend(numpoints=1)
  plt.xlabel('Residuals [ppm]')
  plt.ylabel('CDF')
  plt.show()
  #plt.savefig('cdf'+method+'.png')

  #correlated(resids,name=visit.replace('/','_')+'_'+method)
  
if __name__=='__main__':
  visit=sys.argv[1]+'/'+sys.argv[2]+'/'+sys.argv[3]
  #visit='no_inflation_hatp41'
  wlpaper(visit, method='marg')
  #wlpaper(visit, method='ramp')
