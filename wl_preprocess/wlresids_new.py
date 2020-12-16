import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob as glob
from astropy.io import fits

import marg_mcmc as wl

# Run once with hatp41 old way to recreate spectrum

def wlresids(visit, transit=False):

    # read in xc==cc, xb, first, and last
    orig = visit
    pre=pd.read_csv('preprocess_info.csv', index_col=0).loc[visit]
    #visit = 'no_inflation_hatp41'
    white=pd.read_csv('wl_data.csv', index_col=[0,1]).loc[visit]
   
    first = pre['User Inputs'].values[-2].astype(int)
    last = pre['User Inputs'].values[-1].astype(int)
    norm1 = white.loc['Norm index1', 'Values'].astype(int) + first
    norm2 = white.loc['Norm index2', 'Values'].astype(int) + first
    
    # READ IN ALL PROCESSED DATA
    proc='processed_data.csv'
    df=pd.read_csv(proc, index_col=[0,1]).loc[orig] # back to visit
    transit=df['Transit'].values[0]
    spec=df.loc['Value'].iloc[:,1:-2].dropna(axis=1).values
    specerr=df.loc['Error'].iloc[:,1:-2].dropna(axis=1).values
    date=df.loc['Value','Date'].values
    dir_array=df.loc['Value','Scan Direction'].values
    nexposure=len(date)
    
    # folder = '../data_reduction/reduced/%s/final/*.fits' % visit
    # data=np.sort(np.asarray(glob.glob(folder)))
    # nexposure = len(data)
    # folder = '../reduced/' + visit + '/final/*.fits'
    # date=np.zeros(nexposure)
    # test=fits.open(data[0])
    
    # xlen, ylen = test[0].data.shape
    # test.close()
    # x,y=0,0
    # xlen-=2*x
    # ylen-=2*y
    # allspec=np.ma.zeros((nexposure, xlen, ylen))
    # allerr=np.zeros((nexposure, xlen, ylen))
    # xmin=x
    # xmax=xlen-x
    # ymin=y
    # ymax=ylen-y

    # # RETRIEVE EXPOSURES AND DATES

    # for i, img in enumerate(data):
    #     expfile=fits.open(img)
    #     hdr=expfile[0].header
    #     exp=expfile[0].data
    #     mask=expfile[1].data
    #     errs=expfile[2].data
    #     expfile.close() 
    #     date[i]=(hdr['EXPSTART']+hdr['EXPEND'])/2.
    #     expo=exp[xmin:xmax, ymin:ymax]
    #     mask=mask[xmin:xmax, ymin:ymax]
    #     errs=errs[xmin:xmax, ymin:ymax]
    #     allspec[i,:,:]=np.ma.array(expo, mask=mask)
    #     allerr[i,:,:]=np.ma.array(errs, mask=mask)

    # date_order=np.argsort(date)
    # date=date[date_order]
    # expos=allspec[date_order,:,:]
    # allerr=allerr[date_order,:,:]
  
    #date=date[first:last]          
    #expos=allspec[first:last,:,:]
    #allerr=allerr[first:last,:,:]
   # spec=np.ma.sum(expos,axis=1)
   # specerr=np.sqrt(np.ma.sum(allerr*allerr, axis=1))
    flux = np.sum(spec,axis=1)
    err=np.sqrt(np.sum(specerr*specerr, axis=1))
    
    testbin=spec[:,50:60]
    testbinerr=specerr[:,50:60]
    binflux=np.ma.sum(testbin, axis=1)
    binerr=np.sqrt(np.ma.sum(testbinerr*testbinerr, axis=1))

    norm=np.median(flux[norm1:norm2])
    binnorm=np.median(binflux[norm1:norm2])

    fluxnorm=flux/norm
    errnorm=err/norm
    binflux=binflux/norm
    binerr=binerr/norm

    #CALCULATE THE SHIFT IN DELTA_lambda
    sh = wl.get_shift(spec)
    
    # READ IN SYSTEMATIC MODEL PARAMETERS for all models FROM WHITELIGHT FIT

    models_df = pd.read_csv('wl_models_info.csv', index_col=[0,1]).loc[visit].dropna(axis=1)
    params=models_df.loc['Params'].values[:,:-1].T
    nModels=params.shape[0]

    
    # CALCULATE HST PHASE AT EACH TIME
    HSTper = 96.36 / (24.*60.)
    HSTphase = (date-date[first])/HSTper
    HSTphase = HSTphase - np.floor(HSTphase)
    HSTphase[HSTphase > 0.5] = HSTphase[HSTphase > 0.5] - 1.0
    #plt.clf()
    #plt.close()
    #plt.plot(HSTphase, date, 'ro', ls='')
    #plt.show()
    # DEFINE ARRAY FOR WHICH TO SAVE WHITELIGHT RESIDUALS
    sys_residuals=np.zeros((nModels, nexposure))


    #LOOP OVER MODELS
    #lab = np.array(['Depth', 'Epoch', 'HST1', 'HST2'
    #                , 'HST3', 'HST4', 'sh1','sh2'
    #                , 'sh3', 'sh4', 'i', 'ars', 'c1'
    #                , 'c2', 'c3', 'c4', 'Per', 'Eclipse Depth'
    #                , 'fnorm', 'flinear', 'fquad', 'fexpb'
    #                , 'fexpc', 'flogb', 'flogc', 'rnorm'
    #                , 'rlinear', 'rquad', 'rexpb'
    #                , 'rexpc', 'rlogb', 'rlogc' ])

    for s, par in enumerate(params):
        #pars = par[3:15]
        #print names[np.where(pars != 0.0)]
        model=wl.lightcurve(par, date, sh, HSTphase, dir_array, transit=transit)
        resids=(fluxnorm-model)
        sys_residuals[s,:]=resids
        per=par[16]
        t=par[1]
        # SAVE RESIDUALS
        
        #plt.clf()
        #plt.plot(date[norm1:], 1-resids[norm1:], 'ro',label='Model - Data')
        #plt.plot(date[norm1:], fluxnorm[norm1:], 'bs', label='Normalized Flux')
        #plt.plot(date[norm1:], model[norm1:], marker='x', ls='', label='Model')
        #plt.legend(numpoints=1)
    #plt.show()
    phase = (date-t)/per 
    phase = phase - np.floor(phase)
    phase[phase > 0.5] = phase[phase > 0.5] -1.0

    cols = ['Model ' + str(i) for i in range(nModels)]
    wlresids=pd.DataFrame(sys_residuals.T, columns=cols)
    wlresids['Visit']=visit
    wlresids['Transit']=transit
    wlresids['Scan Direction']=dir_array
    wlresids=wlresids.set_index(['Visit', 'Transit'])
    
    try:
        cur=pd.read_csv('./wlresiduals.csv', index_col=[0,1])
        cur=cur.drop((visit, transit), errors='ignore')
        cur=pd.concat((cur,wlresids), sort=False)
        cur.to_csv('./wlresiduals.csv', index_label=['Obs', 'Transit'])
    except IOError:
        wlresids.to_csv('./wlresiduals.csv', index_label=['Obs', 'Transit'])
        # Do I need to save expos, sh, hstphase, date, flux, fluxnorm, err, errnorm, phase?
        #   SAVE, filename=savename, sys_residuals, expos, sh, hstphase, date
        # saved stuff below for wlcurve too
        #  SAVE, filename='./paper/wlerror/'+planet+'/'+file+'.sav', flux, fluxnorm
        #, err, errnorm, date, phase, xc, cc, start, finish, xb


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit()
    planet=sys.argv[1]
    obs = sys.argv[2]
    direction=sys.argv[3]
    visit=planet+'/'+obs+'/'+direction
    resids=wlresids(visit)
