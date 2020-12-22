from __future__ import print_function
import sys
import time
sys.path.insert(0, '../')

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from tqdm import tqdm
import pandas as pd
import scipy.optimize as op
from scipy.stats import norm

from wave_solution import orbits
from kapteyn import kmpfit
from RECTE import RECTE
import batman

import emcee
import corner


def get_sys_model(p, date, phase, exptime, orbit_start, orbit_end):
    start=date-exptime/2./60/60/24
    count=np.zeros_like(date)+p[8]
    ramp=RECTE(count,start*24*3600., exptime, p[4], p[5], p[6], p[7])
    ramp=ramp/np.median(ramp[orbit_start:orbit_end])
    systematic_model = (phase*p[3] + 1.0) * ramp
    return systematic_model


def get_lightcurve_model(p, date, c1, c2, c3
                         , c4, Per, transit=True):

    #  p0 = [rprs,flux0,m,traps, trapf, dtraps, dtrapf, intrinsic_count]
    params=batman.TransitParams()
    params.w=90.
    params.ecc=0
    params.rp=p[0]
    tc=p[2]
    params.inc=p[10]
    params.a=p[9]
    params.per=Per
    if params.inc>90.: return np.zeros_like(date)
    
    if transit==True:
        params.t0=tc
        params.u=c1, c2, c3, c4
        params.limb_dark="nonlinear"
        m=batman.TransitModel(params, date, fac=0.03)
        model=m.light_curve(params)
    else:
        params.fp=depth
        params.t_secondary=tc
        params.u=[]
        params.limb_dark="uniform"
        m=batman.TransitModel(params, date, transittype="secondary")
        model=m.light_curve(params)

    return model

def lightcurve(p, x, c1, c2, c3, c4,  Per, exptime
               , orbit_start, orbit_end, transit=True):
    """ Function used by MPFIT to fit data to lightcurve model. 

    Inputs: p: input parameters that we are fitting for
    x: Date of each observation, to be converted to phase
    means: Mean pixel count rate time series
    exptime: exposure time 
    transit: True for transit, false for eclipse

    Output: Returns weighted deviations between model and data to be minimized
    by MPFIT. """
             
    phase = (x-p[2])/Per 
    phase = phase - np.floor(phase)
    phase[phase > 0.5] = phase[phase > 0.5] -1.0

    systematic_model=get_sys_model(p, x, phase, exptime, orbit_start, orbit_end)
    lcmodel=get_lightcurve_model(p, x, c1, c2, c3
                                 , c4, Per)
    model=lcmodel * p[1] * systematic_model
    return model

def lnlike(p,x,y, yerr, *args):
    """ p i paramters of model, model is the name of the function of the model
    args contains any extraneous arguments used in model calculation, like sh in
    marginalization. """
    lnf=p[-1]
    #lnf=0
    theory=lightcurve(p,x,*args)
    inv_sigma=1.0/(yerr**2* (1 + np.exp(2*lnf)))
    return -.5*np.sum((y-theory)**2*inv_sigma - np.log(inv_sigma))

def max_like(p_start, x, y, yerr, *extras):
    """ Function to maximize log likelihood. Gives parameter values at max
    log likelihood so we can initialize our walkers to those values."""
    nll = lambda *args: -lnlike(*args)
    #exptime, orbit_start, orbit_end, transit = extras
    result = op.minimize(nll, p_start, args=(x, y, yerr, extras[0]
                                             , extras[1], extras[2], extras[3]
                                             , extras[4], extras[5], extras[6]
                                             , extras[7]))

    #result = op.minimize(nll, p_start, args=(x, y, yerr, model
    #                                         , exptime, orbit_start
    #                                         , orbit_end, transit))
    p_max= result["x"]
    #p_max[0]=np.abs(p_max[0])
    return p_max

def lnprob(p, x, y, yerr, p_start, p_error=0, *args):
    lp=lnprior(p, p_start, p_error)
    if not np.isfinite(lp):
        return -np.inf
    #print 'lp: ', lp
    #print lnlike(p, x, y, yerr, *args)
    return lp + lnlike(p, x, y, yerr, *args)

def lnprior(theta, theta_initial, theta_error, transit=True):
 
    """ Priors on parameters. For system, try both fixing and gaussian priors.
    For depth and others, do "uninformative" uniform priors over a large enough
    range to cover likely results

    Right now I'm extremely conservative. Prior is any possible value for
    open parameters (uniform), and fixed for all others. In future, I will 
    update fixed with gaussian priors and uniform with more appropriate uninformative
    priors. """
    
    # Params: rprs, flux0, m, traps, trapf, dtraps, dtrapf
    # intrinsic_count
    # uninformative: rprs, flux0, m, traps, trapf, dtraps, dtrapf, fp?, intrinsic count
    # gaussian or fixed: inclin, a_r, c1-c4, per
    # if transit==True:
    #     index=np.array([1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0 ,0 ,0, 0, 0, 1, 1])
    #     theta=theta[index==1]
    #     closed=theta[index==0]
    #     closed_i=theta_initial[index==0]
    # else:
    #     index=np.array([0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0 ,0 ,0, 0, 1, 1, 1])
    #     theta=theta[index==1]
    #     closed=theta[index==0]
    #     closed_i=theta_initial[index==0]

    test=np.ones(len(theta))
    if transit==True:
        if not 0 < theta[0] < 0.5: test[0]=np.inf
        if not .5 < theta[1] < 1.5: test[1]=np.inf
        if not theta_initial[2]-1.0 < theta[2] < theta_initial[2]+1.0: test[2]=np.inf
        #test[2]=norm(theta_initial[2], theta_error[0]).pdf(theta[2])
        #sss
        if not -1000 < theta[3] < 1000: test[3]=np.inf
        if not -1000 < theta[4] < 1500: test[4]=np.inf
        if not -1000 < theta[5] < 1500: test[5]=np.inf
        if not -1000 < theta[6] < 1500: test[6]=np.inf
        if not -1000 < theta[7] < 1500: test[7]=np.inf
        if not 0 < theta[8] < 1e5: test[8]=np.inf
        if not theta_initial[9]-2. < theta[9] < theta_initial[9]+2.: test[9]=np.inf
        #test[9]=norm(theta_initial[9], theta_error[2]).pdf(theta[9])
        #test[10]=norm(theta_initial[10], theta_error[1]).pdf(theta[10])
        #if not theta[10] < 90.0: test[10]=np.inf
        if theta_initial[10]-10 < theta[10] < 90.0: test[10]=1
        if not -1000.0 < theta[11] < 100.0: test[11]=np.inf
        test[test==0]=1e-300
        if np.isfinite(np.sum(test)):
            return -np.sum(np.log(test))
        else:
            return -np.inf
    else:
        sys.exit("Didn't do eclipses yet")

def plot_chain(chain, n, nbin, save=False):
    for i in range(chain.shape[0]):
        plt.plot(chain[i,:,n])
    if save:
        #plt.savefig('chains_%02d.png' % n)
        plt.savefig('chains_bin%02d.png' % int(nbin))
        plt.clf()
    else:
        plt.show()
    return None
        
def binramp(p_start # ,perr
            , img_date
            , allspec
            , allerr
            , intrinsic_count
            , exptime
            , visit
            , binsize
            , plotting=False
            , save=False
            , transit=False
            , nbin='test'):
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
    #if fit_method not in ['mcmc', 'mpfit']: sys.exit('Please use either mcmc or mpfit as fit method')

    nexposure = len(img_date)

    # SET THE CONSTANTS USING THE PRIORS
    perr=0
    #perr=perr*5
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
    phot_err=1e6*np.median(err/y)
    
    # Normalised Data
    # get in eclipse orbit, or first transit orbit
    ### Check if this works
    orbit_start, orbit_end=orbits('holder', x=x, y=y, transit=transit)[1]
    norm=np.median(y[orbit_start:orbit_end])
  
    rawerr=err
    rawflux=y
    err = err/norm
    y = y/norm

    """if fit_method=='mpfit':
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
            print 'Depth = ',np.square(params[0]), ' at ', params[2]
        else:
            print 'Depth = ',params[15], ' at ', params[2]"""

    start_time=time.time()
    p0=[rprs, flux0, epoch, m, traps, trapf, dtraps, dtrapf, intrinsic_count, a_r, inclin]
    p0=np.append(p0, 0.0)
    #do stuff, have output of 50th percentile called params.
    #Have point errors called error (error/f/params[1])
 
        
    p_max=max_like(p0, x, y, err, c1, c2, c3
                   , c4, Per, exptime, orbit_start, orbit_end)
    print(p_max)
    
    # phase = (x-epoch)/Per
    # phase -= np.floor(phase)
    # phase[phase > 0.5] = phase[phase > 0.5] -1.0
    # systematic_model=get_sys_model(p_max, x, phase, exptime, orbit_start, orbit_end)
    # lc_model=get_lightcurve_model(p_max, x, epoch, inclin, a_r, c1, c2, c3
    #                               , c4, Per, transit=transit)
    # model=p_max[1]*lc_model*systematic_model
    
    # corrected = y / (p_max[1]*systematic_model)   
    # fit_residuals = (y - model)/p_max[1]
    # fit_err = err/p_max[1]
    
    # Smooth Transit Model: change this from phase to time
    # time_smooth = (np.arange(4000)*0.00025-.5)*Per+epoch
    # phase_smooth=np.arange(4000)*.00025-.5
    # smooth_model=get_lightcurve_model(p_max, time_smooth, epoch, inclin, a_r, c1, c2, c3
    #                                   , c4, Per)
    # plt.clf()
    # plt.errorbar(phase, corrected, fit_err, marker='o', color='blue', ecolor='blue', ls='')
    # plt.plot(phase_smooth, smooth_model)
    # plt.xlim([phase[0]-(phase[1]-phase[0]), phase[-1]+(phase[1]-phase[0])])
    # plt.title('HAT-P-41b WFC3 whitelight curve: Zhou Ramp')
    # plt.xlabel('Phase')
    # plt.ylabel('Normalized Flux')
    # plt.show()
    #p_max=p0

        
        
    ndim, nwalkers = len(p0), 50
    print('done with maximizing likelihood')
    #scale=np.array([1e-3, 1e-2, 1e-4, 1e-2, .1, .1, .1, .1, .1, 1e-3, 1e-3])
    pos=[p_max + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob
                                    , args=(x, y, err, p0, perr, c1, c2, c3 
                                            , c4, Per, exptime
                                            , orbit_start, orbit_end))
    nsteps = 5000
    for i, result in enumerate(sampler.sample(pos, iterations=nsteps)):
        if (i+1) % 100 == 0:
            print("{0:5.1%}".format(float(i) / nsteps))
    #sampler.run_mcmc(pos, nsteps)
    savechain=True
    plot_chain(sampler.chain, 0,nbin, save=savechain)
    #plot_chain(sampler.chain, 2, save=save)
    #plot_chain(sampler.chain, 10, save=save)
    #plot_chain(sampler.chain, 8, save=save)
    burn = 3500
    samples = sampler.chain[:,burn:,:].reshape((-1, ndim))
    samples[:,0]=samples[:,0]**2*1e6
    samples[:,-1]=np.sqrt(1.0+ np.exp(2.0*samples[:,-1]))
    s2=samples[:,:3]
    fig = corner.corner(samples, labels=['depth', 'Norm', 't0'
                                         ,'slope', 'ramp_ts', 'ramp_tf'
                                         ,'ramp_dts', 'ramp_dtf', 'ramp_Count'
                                         , 'ar*', 'i', 'f']) #list of params
    #plt.show()
    plt.savefig("corner_f"+nbin+'.png')
    plt.clf()
        
    p_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                 zip(*np.percentile(samples, [16, 50, 84],
                                    axis=0)))
    print(p_mcmc)
    params=np.zeros_like(p0)
    for i, tup in enumerate(p_mcmc):
        params[i]=tup[0]

    params[0]=(params[0]/1e6)**.5
    phase = (x-params[2])/Per
    phase -= np.floor(phase)
    phase[phase > 0.5] = phase[phase > 0.5] -1.0
    systematic_model=get_sys_model(params, x, phase, exptime, orbit_start, orbit_end)
    lc_model=get_lightcurve_model(params, x, c1, c2, c3
                                  , c4, Per, transit=transit)
    model=params[1]*lc_model*systematic_model
    
    corrected = y / (params[1]*systematic_model)   
    fit_residuals = (y - model)/params[1]
    fit_err = err*params[-1]/params[1]
    rms = np.std(fit_residuals)
    depth=p_mcmc[0][0]/1e6
    depth_err=np.mean(p_mcmc[0][1:])/1e6

    # Smooth Transit Model: change this from phase to time
    time_smooth = (np.arange(500)*0.00025-.5)*Per+params[2]
    phase_smooth=np.arange(500)*.00025-.5
    smooth_model=get_lightcurve_model(params, time_smooth, c1, c2, c3
                                      , c4, Per)
    plt.clf()
    plt.errorbar(phase, corrected, fit_err, marker='o', color='blue', ecolor='blue', ls='')
    plt.plot(phase_smooth, smooth_model)
    plt.xlim([phase[0]-(phase[1]-phase[0]), phase[-1]+(phase[1]-phase[0])])
    plt.title('HAT-P-41b WFC3 whitelight curve: Zhou Ramp')
    plt.xlabel('Phase')
    plt.ylabel('Normalized Flux')
    #plt.savefig('mcmcfit_f.png')
    #plt.show()
    plt.clf()

    plt.errorbar(phase, fit_residuals, fit_err, marker='o', color='blue', ecolor='blue', ls='')
    plt.plot(phase, np.zeros_like(phase), 'r')
    #plt.show()
    #plt.savefig('mcmc_residuals_f.png')
    plt.clf()
    print(np.std(fit_residuals)*1e6/np.median(phot_err))
    plt.hist((fit_residuals/fit_err)/np.sum(fit_residuals/fit_err), 20)
    plt.clf()
    #plt.savefig('residual_f.png')
    print(time.time()-start_time)

    #####################################################3
    """if save == True:
        ################# make sure this works
        ### Two dataframes, both multi-indexed

        # To retrieveas numpy array: df.loc[visit,column].values
        # Example: wl_models_info.loc['hatp41/visit01/reverse','Params'].values[0]

        # Save all plotting stuff
        cols = ['Date', 'Flux', 'Flux Error', 'Norm Flux', 'Norm Flux Error', 'Model Phase'
                , 'Model', 'Corrected Flux', 'Corrected Flux Error', 'Residuals']

        bins=pd.DataFrame(np.vstack((x, rawflux, rawerr, y, error, phase, model
                                   , corrected, fit_err, fit_residuals)).T,
                        columns=cols)
        bins['Visit']=visit
        bins['binsize']=binsize
        bins['bin']=nbin
        bins=bins.set_index(['Visit','binsize', 'bin'])
        bins['Transit']=transit

        # Save smooth models
        cols=['Time', 'Phase', 'Model']
        data=np.vstack((time_smooth, phase_smooth, smooth_model)).T
        bin_smooth=pd.DataFrame(data, columns=cols)
        bin_smooth['Visit']=visit
        bin_smooth['binsize']=binsize
        bin_smooth['bin']=nbin
        bin_smooth=bin_smooth.set_index(['Visit','binsize', 'bin'])
        bin_smooth['Transit']=transit

        # Save results
        cols=['Depth', 'RMS', 'Photon Error', 'Ratio', 'Norm index1', 'Norm index2', 'rprs'
              , 'Zero-flux' , 'Event time', 'Slope', 'ramp1', 'ramp2','ramp3', 'ramp4'
              , 'inc','ar', 'c1', 'c2', 'c3', 'c4', 'Period', 'eclipse depth', 'Intrinsic Count']
        data=[depth, rms,phot_err, ratio, orbit_start, orbit_end] + params
        errors= [depth_err, 0, 0, 0, 0, 0] + stderror.tolist()
        ind2=pd.MultiIndex.from_product([[visit],[binsize],[nbin],['Values', 'Errors']])
        bin_params = pd.DataFrame(np.vstack((data,errors)), columns=cols, index=ind2)
        bin_params['Transit']=transit
    
        try:
            cur=pd.read_csv('./binmcmc_params.csv', index_col=[0,1, 2, 3])
            #cur=cur.drop((visit, binsize,bin))
            cur=pd.concat((cur,bin_params))
            cur=cur[~cur.index.duplicated(keep='first')]
            cur.to_csv('./binmcmc_params.csv', index_label=['Obs','Bin Size','Bin', 'Type'])
        except IOError:
            bin_params.to_csv('./binmcmc_params.csv', index_label=['Obs','Bin Size', 'Bin', 'Type'])
            
        try:
            curr=pd.read_csv('./binmcmc_data.csv', index_col=[0,1, 2])
            curr=curr.drop((visit, binsize,int(nbin)), errors='ignore')
            curr=pd.concat((curr,bins))
            #curr=curr[~curr.index.duplicated(keep='first')]
            curr.to_csv('./binmcmc_data.csv', index_label=['Obs', 'Bin Size', 'Bin'])
        except IOError:
            bins.to_csv('./binmcmc_data.csv', index_label=['Obs', 'Bin Size', 'Bin'])

        try:
            currr=pd.read_csv('./binmcmc_smooth.csv', index_col=[0,1,2])
            currr=currr.drop((visit, binsize,int(nbin)), errors='ignore')
            currr=pd.concat((currr,bin_smooth))
           # currr=currr[~currr.index.duplicated(keep='first')]
            currr.to_csv('./binmcmc_smooth.csv', index_label=['Obs', 'Bin Size', 'Bin'])
        except IOError:
            bin_smooth.to_csv('./binmcmc_smooth.csv', index_label=['Obs', 'Bin Size', 'Bin'])
    """

    return [depth, depth_err, rms]
   
