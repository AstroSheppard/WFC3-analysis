from __future__ import print_function
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from tqdm import tqdm
import pandas as pd

from wave_solution import orbits
from kapteyn import kmpfit
from RECTE import RECTE
import batman

def get_sys_model(p, date, phase, exptime, orbit_start, orbit_end):
    start=date-exptime/2./60/60/24
    count=np.zeros_like(date)+p[16]
    ramp=RECTE(count,start*24*3600., exptime, p[4], p[5], p[6], p[7])
    ramp=ramp/np.median(ramp[orbit_start:orbit_end])
    systematic_model = (phase*p[3] + 1.0) * ramp
    return systematic_model

def get_lightcurve_model(p, date, transit=False):
    #  p0 = [rprs,flux0,epoch,m,traps, trapf, dtraps, dtrapf
        # , inclin,a_r,c1,c2,c3,c4,Per,fp, intrinsic_count]
    params=batman.TransitParams()
    params.w=90.
    params.ecc=0
    params.rp=p[0]
    tc=p[2]
    params.inc=p[8]
    params.a=p[9]
    params.per=p[14]
    depth=p[15]

    if transit==True:
        params.t0=tc
        params.u=p[10:14]
        params.limb_dark="nonlinear"
        m=batman.TransitModel(params, date, fac=0.01875)
        model=m.light_curve(params)
    else:
        params.fp=depth
        params.t_secondary=tc
        params.u=[]
        params.limb_dark="uniform"
        m=batman.TransitModel(params, date, transittype="secondary")
        model=m.light_curve(params)

    return model

def lightcurve(p, x, exptime, orbit_start, orbit_end, transit=False):
    
    """ Function used by MPFIT to fit data to lightcurve model. 

    Inputs: p: input parameters that we are fitting for
    x: Date of each observation, to be converted to phase
    means: Mean pixel count rate time series
    exptime: exposure time 
    transit: True for transit, false for eclipse

    Output: Returns weighted deviations between model and data to be minimized
    by MPFIT. """

    Per = p[14]              

    phase = (x-p[2])/Per 
    phase = phase - np.floor(phase)
    phase[phase > 0.5] = phase[phase > 0.5] -1.0

    systematic_model=get_sys_model(p, x, phase, exptime, orbit_start, orbit_end)
    lcmodel=get_lightcurve_model(p, x, transit=transit)
    model=lcmodel * p[1] * systematic_model

    return model


def residuals(p,data):
    x, y, err, exptime, transit, orbit_start, orbit_end = data
    ym=lightcurve(p, x, exptime, orbit_start, orbit_end, transit=transit)
    return (y-ym)/err
  
def ramp2018(p_start, img_date, allspec, allerr, intrinsic_count, exptime, plotting=False
                   , fixtime=False, norandomt=False, openinc=False, openar=False
                   , savewl=False, transit=False):
    """ Inputs
    p_start: rp/rs
    event time
    inclination
    semimajor axis/stellar radius
    period
    planetary flux for secondary eclipses
    limb darkening params

    img_date: time of each observation
    allspec: all 1D spectra: flux at each pixel column for each observation
    allerr: all 1D errors: error for each pixel column
    intrinsic_count: raw count of leveling off of ramp in orbit before event (per pixel per second)
    exptime = exposure time
    """
    
    # TOTAL NUMBER OF EXPOSURES IN THE OBSERVATION
    nexposure = len(img_date)

    # SET THE CONSTANTS USING THE PRIORS
    rprs = p_start[0]
    epoch = p_start[1]
    inclin = p_start[2]
    a_r = p_start[3]
    Per = p_start[4]
    fp=p_start[5] #eclipse depth (planetary flux)
    c1=p_start[6]
    c2=p_start[7]
    try:
        c3, c4 = p_start[8:]
    except ValueError:
        c3, c4 = 0.0, 0.0
    flux0 = 1.
    m = 0.0         # Linear Slope
    traps=2
    trapf=10
    dtraps=0.0
    dtrapf=0.
    
    #PLACE ALL THE PRIORS IN AN ARRAY
    p0 = [rprs,flux0,epoch,m,traps, trapf, dtraps, dtrapf
          ,inclin,a_r,c1,c2,c3,c4,Per,fp, intrinsic_count]
    system=[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0]
    
    nParam=len(p0)
    # SELECT THE SYSTEMATIC GRID OF MODELS TO USE 

    #  SET UP THE ARRAYS  ;

    phase = np.zeros(nexposure)
    x = img_date
    y=allspec.sum(axis=1)
    err = np.sqrt(np.sum(allerr*allerr, axis=1))
    #phot_err=1e6/np.median(np.sqrt(y))
    phot_err=1e6*np.median(err/y) #theoretical limit
    
    # Normalised Data
    # get in eclipse orbit, or first transit orbit
    ### Check if this works
    orbit_start, orbit_end=orbits('holder', x=x, y=y, transit=transit)[1]
    norm=np.median(y[orbit_start:orbit_end])
  
    rawerr=err
    rawflux=y
    err = err/norm
    y = y/norm
    
    if fixtime==True: system[2] = 1
    if openinc==True: system[8] = 0
    if openar==True: system[9] = 0
    if transit==False:
        system[0]=1
        system[15]=0
    parinfo=[]
    for i in range(len(p0)):
        parinfo.append({'fixed':system[i]})
    fa=(x,y,err,exptime,transit, orbit_start, orbit_end)
    m2=kmpfit.Fitter(residuals=residuals, data=fa, parinfo=parinfo, params0=p0)
    m2.fit()
    params_w=m2.params
    AIC=(2*len(x)*np.log(np.median(err))+len(x)*np.log(2*np.pi)
         + m2.chi2_min + 2*m2.nfree)
    if transit==True:
        print('Depth = ', np.square(params_w[0]), ' at ', params_w[2])
    else:
        print('Depth = ', params_w[15], ' at ', params_w[2])

    # Re-Calculate each of the arrays dependent on the output parameters
    phase = (x-params_w[2])/params_w[14] 
    phase -= np.floor(phase)
    phase[phase > 0.5] = phase[phase > 0.5] -1.0
    
    # LIGHT CURVE MODEL: calculate the eclipse model for the resolution of the data points
    # this routine is from MANDEL & AGOL (2002)

    systematic_model=get_sys_model(params_w, x, phase, exptime, orbit_start, orbit_end)
    lc_model=get_lightcurve_model(params_w, x, transit=transit)
    w_model=params_w[1]*lc_model*systematic_model  
    w_residuals = (y - w_model)/params_w[1]
    std = np.std(w_residuals)

    #######################################
    # Scale error by resid_stddev[top]
    if np.median(err) < std:
        error=err*std/np.median(err)
    else:
        error=err

    # Define the new priors as the parameters from the best fitting
    # systematic model
    p0=params_w
    fa=(x,y,error,exptime,transit, orbit_start, orbit_end)
    m2=kmpfit.Fitter(residuals=residuals, data=fa, parinfo=parinfo, params0=p0)
    m2.fit()
    params=m2.params
    perror=m2.xerror
    nfree=m2.nfree
    stderror=m2.stderr

    if transit==True:
        print('Depth = ',np.square(params[0]), ' at ', params[2])
    else:
        print('Depth = ',params[15], ' at ', params[2])

    # Re-Calculate each of the arrays dependent on the output parameters
    phase = (x-params[2])/params[14] 
    phase -= np.floor(phase)
    phase[phase > 0.5] = phase[phase > 0.5] -1.0

    systematic_model=get_sys_model(params, x, phase, exptime, orbit_start, orbit_end)
    lc_model=get_lightcurve_model(params, x, transit=transit)
    model=params[1]*lc_model*systematic_model
    corrected = y / (params[1]*systematic_model)   
    fit_residuals = (y - model)/params[1]
    fit_err = error/params[1]

    # Smooth Transit Model: change this from phase to time
    time_smooth = (np.arange(500)*0.002-.5)*params[14]+params[2]
    phase_smooth=np.arange(500)*.002-.5
    smooth_model=get_lightcurve_model(params, time_smooth, transit=transit)
    
    # PLOTTING
    if plotting == True:
        #plt.errorbar(img_date, y, error,ecolor='red', color='red', marker='o', ls='')
        #plt.ylim([0.982, 1.005])
        #plt.plot(img_date, systematic_model, color='blue', marker='o', ls='')
        plt.errorbar(phase, corrected, fit_err, marker='o', color='blue', ecolor='blue', ls='')
        plt.plot(phase_smooth, smooth_model)
        plt.xlim([phase[0]-(phase[1]-phase[0]), phase[-1]+(phase[1]-phase[0])])
        plt.title('HAT-P-41b WFC3 whitelight curve: Zhou Ramp')
        plt.xlabel('Phase')
        plt.ylabel('Normalized Flux')
       # plt.ylim([.999,1.001])
        plt.show(block=False)
       # plt.errorbar(phase, fit_residuals, fit_err, marker='o', color='b', ls='',ecolor='blue')
       # plt.plot(phase, np.zeros_like(phase))
       # plt.show()
        
    # SAVE out the arrays for each systematic model ;
        
    if transit==True:
        depth = np.square(params[0])
        depth_err = stderror[0]*2.0*params[0]
    else:
        depth = params[15]
        depth_err = stderror[15]

    
    epoch=params[2]
    epoch_err=stderror[2]
    inc=params[8]
    inc_err=stderror[8]
    ar=params[9]
    ar_err=stderror[9]
    c=params[10:14]
    c_err=stderror[10:14]

    rms=np.std(fit_residuals)*1e6
    ratio=rms/phot_err
    print('a/rs: %.3f +/- %.3f' % (ar, ar_err))
    print('a/rs: %.3f +/- %.3f' % (inc, inc_err))
    print('Rms: %f' % rms)
    print('Photon error: %f' % phot_err)
    print('Ratio: %f' % ratio)

    if savewl:
        ################# make sure this works
        ### Two dataframes, both multi-indexed
        # First: Anything dependent on systematic models. Adjusted data, weights,
        # residuals, etc.
        # Second: All data things for plots. Flux/errors normalized and un. Dates,
        # And then all relevant fitting results (RMS/photon)

        # To retrieveas numpy array: df.loc[visit,column].values
        # Example: wl_models_info.loc['hatp41/visit01/reverse','Params'].values[0]

        # Save all plotting stuff
        cols = ['Date', 'Flux', 'Flux Error', 'Norm Flux', 'Norm Flux Error', 'Model Phase'
                , 'Model', 'Corrected Flux', 'Corrected Flux Error', 'Residuals']

        wl=pd.DataFrame(np.vstack((x, rawflux, rawerr, y, error, phase, model
                                   , corrected, fit_err, fit_residuals)).T,
                        columns=cols)
        wl['Visit']=savewl
        wl=wl.set_index('Visit')
        wl['Transit']=transit

        # Save smooth models
        cols=['Time', 'Phase', 'Model']
        data=np.vstack((time_smooth, phase_smooth, smooth_model)).T
        wl_smooth=pd.DataFrame(data, columns=cols)
        wl_smooth['Visit']=savewl
        wl_smooth=wl_smooth.set_index('Visit')
        wl_smooth['Transit']=transit

        # Save results
        cols=['Depth', 'RMS', 'Photon Error', 'Ratio', 'Norm index1', 'Norm index2', 'rprs'
              , 'Zero-flux' , 'Event time', 'Slope', 'ramp1', 'ramp2','ramp3', 'ramp4'
              , 'inc','ar', 'c1', 'c2', 'c3', 'c4', 'Period', 'eclipse depth', 'Intrinsic Count']
        data=[depth, rms,phot_err, ratio, orbit_start, orbit_end] + params
        errors= np.append([depth_err, 0, 0, 0, 0, 0], stderror)
        ind2=pd.MultiIndex.from_product([[savewl],['Values', 'Errors']])
        wl_params = pd.DataFrame(np.vstack((data,errors)), columns=cols, index=ind2)
        wl_params['Transit']=transit
    
        try:
            cur=pd.read_csv('./wl_ramp_params.csv', index_col=[0,1])
            #cur=cur.drop(savewl, level=0)
            cur=pd.concat((cur,wl_params))
            cur=cur[~cur.index.duplicated(keep='first')]
            cur.to_csv('./wl_ramp_params.csv', index_label=['Obs', 'Type'])
        except IOError:
            wl_params.to_csv('./wl_ramp_params.csv', index_label=['Obs', 'Type'])
            
        try:
            curr=pd.read_csv('./wl_ramp_data.csv', index_col=0)
            curr=curr.drop(savewl)
            curr=pd.concat((curr,wl))
            curr.to_csv('./wl_ramp_data.csv', index_label='Obs')
        except IOError:
            wl.to_csv('./wl_ramp_data.csv', index_label='Obs')

        try:
            currr=pd.read_csv('./wl_ramp_smooth.csv', index_col=0)
            currr=currr.drop((savewl))
            currr=pd.concat((currr,wl_smooth))
            currr.to_csv('./wl_ramp_smooth.csv', index_label='Obs')
        except IOError:
            wl_smooth.to_csv('./wl_ramp_smooth.csv', index_label='Obs')
            

    return [depth, depth_err, epoch
            , epoch_err, inc, inc_err
            , ar, ar_err, c, c_err]




