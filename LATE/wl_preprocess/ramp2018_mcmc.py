
import sys
import time
sys.path.insert(0, '../')

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from tqdm import tqdm
import pandas as pd
import scipy.optimize as op
import scipy.stats
import pickle

from wave_solution import orbits
from kapteyn import kmpfit
from RECTE import RECTE
import batman
import emcee
import corner


def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4.0 * n

    # Optionally normalize
    if norm:
        acf /= float(acf[0])


    return acf

def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= float(len(y))
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


def plot_chain(chain, n, lab='param', save=False, mc_dir='.'):
    for i in range(chain.shape[0]):
        plt.plot(chain[i,:,n])
        plt.title(lab)
    if save:
        plt.savefig(mc_dir+'/'+lab+'_chains.pdf')
        plt.clf()
    else:
        plt.show()
    return None

def get_sys_model(p, date, phase, exptime, orbit_start, orbit_end, lc=1.0):
    start=date-exptime/2./60/60/24
    count=(np.zeros_like(date)+p[16])*lc # replace this with transit

    ramp=RECTE(count,start*24*3600., exptime=exptime, trap_pop_s=p[4],
               trap_pop_f=p[5], dTrap_s=p[6], dTrap_f=p[7])
    ramp=ramp/np.median(ramp[orbit_start:orbit_end])
    #ramp = ramp/np.median(ramp)
    # can easily add other slopes here
    systematic_model = p[1] * (phase*p[3] + 1.0) * ramp
    return systematic_model


def get_lightcurve_model(p, date, transit=True):

                    #p0 = [rprs,flux0,epoch,m,traps, trapf, dtraps, dtrapf
    #  ,inclin,a_r,c1,c2,c3,c4,Per,fp, intrinsic_count]

    #  p0 = [rprs,flux0,m,traps, trapf, dtraps, dtrapf, intrinsic_count]
    params=batman.TransitParams()
    params.w=90.
    params.ecc=0
    params.rp=p[0]
    tc=p[2]
    params.inc=p[8]
    params.a=p[9]
    params.per=p[14]
    if params.inc>90.: return np.zeros_like(date)
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

def lightcurve(p, x, exptime, orbit_start, orbit_end, transit=True):
    """ Function used by MPFIT to fit data to lightcurve model.

    Inputs: p: input parameters that we are fitting for
    x: Date of each observation, to be converted to phase
    means: Mean pixel count rate time series
    exptime: exposure time
    transit: True for transit, false for eclipse

    Output: Returns weighted deviations between model and data to be minimized
    by MPFIT. """

    phase = (x-p[2])/p[14]
    phase = phase - np.floor(phase)
    phase[phase > 0.5] = phase[phase > 0.5] -1.0

    lcmodel=get_lightcurve_model(p, x, transit=transit)
    systematic_model=get_sys_model(p, x, phase, exptime, orbit_start
                                   , orbit_end, lc=lcmodel)
    lcmodel=1.0
    model=lcmodel * systematic_model
    return model

def residuals(p,data):
    x, y, err, exptime, transit, orbit_start, orbit_end = data
    ym=lightcurve(p, x, exptime, orbit_start, orbit_end, transit=transit)
    return (y-ym)/err

def lnlike(p,x,y, yerr, *args):
    """ p i paramters of model, model is the name of the function of the model
    args contains any extraneous arguments used in model calculation, like sh in
    marginalization. """

    theory=lightcurve(p,x,*args)
    inv_sigma=1.0/(yerr**2)
    return -.5*np.sum((y-theory)**2*inv_sigma - np.log(inv_sigma))

def max_like(p_start, x, y, yerr, perr, *extras):
    """ Function to maximize log likelihood. Gives parameter values at max
    log likelihood so we can initialize our walkers to those values."""


    # This is out of date since changing emcee so that extras can just be params
    nll = lambda *args: -lnlike(*args)
    bounds = ((0,.5), (.5, 1.5), (p_start[2]-1.0, p_start[2]+1.0), (-5, 5), (0, 200), (0, 200)
              ,(0, 200), (0, 200), (0, 1e4), (p_start[9],p_start[9])
              , (p_start[10], p_start[10]), (0,0.0))
    #exptime, orbit_start, orbit_end, transit = extras
    result = op.minimize(nll, p_start, bounds=bounds, method='TNC'
                         , args=(x, y, yerr, extras[0]
                                 , extras[1], extras[2], extras[3]
                                 , extras[4], extras[5], extras[6]
                                 , extras[7]))

    #result = op.minimize(nll, p_start, args=(x, y, yerr, model
    #                                         , exptime, orbit_start
    #                                         , orbit_end, transit))
    p_max= result["x"]
    #p_max[0]=np.abs(p_max[0])

    return p_max

def lnprob(p, x, y, yerr, p_start, p_error, syst, *args):

    params = p_start.copy()
    params[syst==0] = p
    lp=lnprior(params, p_start, p_error, syst)
    if not np.isfinite(lp):
        return -np.inf, -np.inf
    #print lp
    #print lnlike
    return lp + lnlike(params, x, y, yerr, *args), lp

def lnprior(theta, theta_initial, theta_error, syst, transit=True):

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

    #p0 = [rprs,flux0,epoch,m,traps, trapf, dtraps, dtrapf
    #  ,inclin,a_r,c1,c2,c3,c4,Per,fp, intrinsic_count]
    if not np.all(theta[syst==1] == theta_initial[syst==1]): return -np.inf
    ind = np.where(syst==0)
    test=np.ones(len(theta))
    if transit==True:
        if syst[0]==0 and not 0 < theta[0] < 0.2: return -np.inf
        if syst[1]==0 and not 0.5 < theta[1] < 1.5: return -np.inf
        if syst[2]==0 and not theta_initial[2]-1.0 < theta[2] < theta_initial[2]+1.0: return -np.inf
        if syst[3]==0 and not -5 < theta[3] < 5: return -np.inf
        if syst[4]==0 and not -100 < theta[4] < 500: return -np.inf
        if syst[5]==0 and not -100 < theta[5] < 500: return -np.inf
        if syst[6]==0 and not -100 < theta[6] < 500: return -np.inf
        if syst[7]==0 and not -100 < theta[7] < 500: return -np.inf
        if not theta[8] < 90.0: return -np.inf
        if syst[8] == 0: test[8]=scipy.stats.norm.pdf(theta[8], theta_initial[8], theta_error[8])
        #if syst[9]==0 and not 0 < theta[9] < 10.0: return -np.inf
        if syst[9] == 0: test[9]=scipy.stats.norm.pdf(theta[9], theta_initial[9], theta_error[9])
        if syst[10]==0 and not theta_initial[10] == theta[10]: return -np.inf
        if syst[11]==0 and not theta_initial[11] == theta[11]: return -np.inf
        if syst[12]==0 and not theta_initial[12] == theta[12]: return -np.inf
        if syst[13]==0 and not theta_initial[13] == theta[13]: return -np.inf
        if syst[14]==0 and not theta_initial[14] == theta[14]: return -np.inf
        if syst[15]==0 and not 0 < theta[15] < 0.2: return -np.inf
        if syst[16]==0 and not 0 < theta[16] < 10000: return -np.inf

        test[test==0]=1e-300

        if np.isfinite(np.sum(np.log(test))):
            return np.sum(np.log(test))
        else:
            return -np.inf
    else:
        sys.exit("Didn't do eclipses yet")


def ramp2018(p_start,
             perr,
             img_date,
             allspec,
             allerr,
             intrinsic_count,
             exptime,
             plotting=False,
             fixtime=False,
             norandomt=False,
             openinc=False,
             openar=False,
             savewl=False,
             transit=False,
             fit_method='mcmc'):


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
    if fit_method not in ['mcmc', 'mpfit']: sys.exit('Please use either mcmc or mpfit as fit method')

    nexposure = len(img_date)

    # SET THE CONSTANTS USING THE PRIORS
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
    lab= np.array(['Depth', 'Norm', 'Epoch', 'Slope', 'R_ts'
                   , 'R_tf', 'R_dts', 'R_dtf', 'Inc', 'ars'
                   ,'c1','c2','c3','c4','Period', 'Eclipse Depth', 'Count'])

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

    if fixtime==True: system[2] = 1
    if openinc==True: system[8] = 0
    if openar==True: system[9] = 0
    if transit==False:
        system[0]=1
        system[15]=0
    parinfo=[]
    for i in range(len(p0)):
        dic = {'fixed':system[i]}
        if lab[i]=='R_ts' or lab[i]=='R_tf' or lab[i]=='R_dts' or lab[i]=='R_dtf':
            dic = {'fixed':system[i], 'limits': [-100,500]}
        if lab[i] == 'count':
            dic = {'fixed':system[i], 'limits': [0,None]}
        if lab[i]=='Inc':
            dic = {'fixed':system[i], 'limits': [0.0,90.0]}
        parinfo.append(dic)
    fa=(x,y,err,exptime,transit, orbit_start, orbit_end)
    m2=kmpfit.Fitter(residuals=residuals, data=fa, parinfo=parinfo, params0=p0)
    m2.fit()
    params = m2.params
    perror = m2.stderr
    print(m2.rchi2_min)

    # Re-Calculate each of the arrays dependent on the output parameters
    phase = (x-params[2])/params[14]
    phase -= np.floor(phase)
    phase[phase > 0.5] = phase[phase > 0.5] -1.0

    # LIGHT CURVE MODEL: calculate the eclipse model for the resolution of the data points
    # this routine is from MANDEL & AGOL (2002)

    lc_model=get_lightcurve_model(params, x, transit=transit)
    plt.plot(x, lc_model)
    systematic_model=get_sys_model(params, x, phase, exptime, orbit_start, orbit_end, lc=lc_model)
    lc_model=1.0
    w_model=lc_model*systematic_model
    w_residuals = (y - w_model)
    std = np.std(w_residuals)
    print(std/np.median(err))
    plt.plot(x, w_model, 'ro')
    #plt.plot(x, systematic_model, 'go')
    plt.errorbar(x, y,err, color='b', marker='x')
    plt.show()
    plt.close()
    plt.clf()
    plt.plot(x, w_residuals, 'bo')
    plt.show()

    ac_resids = autocorr_func_1d(w_residuals, norm=True)
    mins = np.zeros_like(ac_resids)
    mins[ac_resids<0] = ac_resids[ac_resids<0]
    maxs = np.zeros_like(ac_resids)
    maxs[ac_resids>0]=ac_resids[ac_resids>0]
    plt.close()
    plt.clf()
    lags = np.arange(len(ac_resids))
    plt.plot(ac_resids, 'bo')
    plt.vlines(lags, mins, maxs, 'b')
    sig = 0.05 # 95% confidence interval
    conf = scipy.stats.norm.ppf(1-sig/2.)/np.sqrt(len(w_residuals))
    plt.plot(lags, np.zeros_like(ac_resids)+conf, color='r', label='2 sigma range')
    plt.plot(lags, np.zeros_like(ac_resids)-conf, color = 'r')
    plt.title('Autocorrelation function of residuals')
    plt.legend()
    plt.show()

    scale=std/np.median(err)
    #err=err*scale

    #######################################
    # Scale error by resid_stddev[top]
    #if np.median(err) < std:
    #    scale = std/np.median(err)
    #else:
    #    scale = 1.0
    #print scale
    #scale = 1.0
    #error = scale * err
    # Define the new priors as the parameters from the best fitting
    # systematic model
    #p0=params_w
    #fa=(x,y,error,exptime,transit, orbit_start, orbit_end)
    #m2=kmpfit.Fitter(residuals=residuals, data=fa, parinfo=parinfo, params0=p0)
    #m2.fit()
    #params=m2.params
    #perror=m2.xerror
    #nfree=m2.nfree
    #stderror=m2.stderr

    if transit==True:
        print('Depth = %.1f +/- %.1f at %.5f' %(np.square(params[0])*1e6,
                                                perror[0]*2.0*params[0]*1e6, params[2]))
    else:
        print('Depth = ',params[15], ' at ', params[2])

    if fit_method=='mcmc':
        start_time=time.time()

        literr = perr.copy()
        #p_max=max_like(p0, x, y, err, perr, c1, c2, c3
        #               , c4, Per, exptime, orbit_start, orbit_end)

        p0 = np.array(params).copy()
        perr = np.array(perror).copy()
        #print 'Max likelihood', p_max
        print('NLLS', p0)

        #p0=np.append(p0, 0.0)
        #perr=np.append(perr, 0.0)

        #do stuff, have output of 50th percentile called params.
        #Have point errors called error (error/f/params[1])




        #p_max[-1]=1.0
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

           #labels= ['Depth', 'Norm', 'Epoch', 'Slope', 'R_ts'
           #      , 'R_tf', 'R_dts', 'R_dtf', 'Inc', 'ars'
           #      ,'c1','c2','c3','c4','Period', 'Eclipse Depth', 'Count']
        syst=np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0])
        if fixtime==True: syst[2] = 1
        if openinc==True: syst[8] = 0
        if openar==True:
            syst[9] = 0
            p0[9]=p_start[3]
            perr[9]=literr[3]
        if transit==False:
            syst[0]=1
            syst[15]=0
        ndim, nwalkers = len(p0[syst==0]), int(len(p0[syst==0])*2.5/2)*2
        print('done with maximizing likelihood')
        #scale=np.array([1e-3, 1e-2, 1e-4, 1e-2, .1, .1, .1, .1, .1, 1e-3, 1e-3])

        # when using gaussian prior, set ars prior to literature values

        #print p0[9], perr[9]
        if np.any(perr[syst==0]==0.0):
            ix = np.where((syst==0) & (perr==0.0))[0]
            for i in ix:
                if p0[i] != 0.0:
                    perr[i] = np.abs(p0[i])*.05
                else:
                    perr[i]=1.0

        #sss
        pos=np.array([p0[syst==0] + 5*perr[syst==0]*np.random.randn(ndim) for i in range(nwalkers)])
        # hack to deal with forcing ramp params to be positive for hatp41
        pos[:,-1]=np.abs(pos[:,-1])
        pos[:,4:8][pos[:,4:8]>500]=499
        pos[:,4:8][pos[:,4:8]<-100]=-99

        #theta=p0.copy()
        #lp=lnprior(theta, p0, perr, syst)
        #theta[9]=p_start[3]
        #lpp=lnprior(theta,p0, perr, syst)
        #sss
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob
                                        , args=(x, y, err, p0, perr, syst, exptime
                                                , orbit_start, orbit_end))
        nsteps = 20000
        for i, result in enumerate(sampler.sample(pos, iterations=nsteps)):
            if (i+1) % 100 == 0:
                print("{0:5.1%}".format(float(i) / nsteps))
        #sampler.run_mcmc(pos, nsteps)
        print("Time elapsed in minutes %.2f" % ((time.time()-start_time)/60))


        burn = 5000

        for pp in range(len(p0[syst==0])):
            print(lab[syst==0][pp])
            chain = sampler.chain[:,burn:,pp]
            N = np.exp(np.linspace(np.log(100), np.log(chain.shape[1]), 10)).astype(int)
            new = np.empty(len(N))
            for i, n in enumerate(N):
                new[i] = autocorr_new(chain[:, :n])
            plt.loglog(N, new, "o-", label=lab[pp])
        plt.plot(N, N/50., 'go', label='N/50')
        plt.xlabel('Chain Length')
        plt.ylabel('Autocorrelation time estimate')
        plt.legend(prop={'size': 6})
        plt.show()
        taus = np.zeros_like(p0[syst==0])
        for pp in range(len(p0[syst==0])):
            print(lab[syst==0][pp])
            chain = sampler.chain[:,burn:,pp]
            taus[pp] = autocorr_new(chain)

        print(taus)
        print(' Mean integrated auto time: %.2f' % np.mean(taus))

        pickle_dict = {'sampler': sampler, 'ndim': ndim,
                       'nwalkers':nwalkers, 'syst':syst,'lab':lab,
                       'taus':taus, 'burn':burn}
        savemc=False
        if savemc:
            mc_dir='emcee_runs/ramp/hatp41/visit01/reverse'
            pickle.dump(pickle_dict
                        , open( mc_dir +"/sampler.p", "wb" ) )

        samples = sampler.chain[:,burn:,:].reshape((-1, ndim))
        samples_orig = samples.copy()

        # Cut-off here: I did tau stuff already, now plot all resids, all models, corner
        inds = np.random.randint(len(samples), size=100)
        pp = p0.copy()
        for ind in inds:
            samp = samples[ind]
            pp[syst==0] = samp
            phase = (x-pp[1])/Per
            phase -= np.floor(phase)
            phase[phase > 0.5] = phase[phase > 0.5] -1.0
            mod = lightcurve(pp, x, exptime, orbit_start, orbit_end, transit=transit)
            plt.plot(x, mod, '.k', ls='', alpha=.1)
        plt.errorbar(x, y, err, marker='o', color='b', ecolor='b', ls='')
        plt.show()

        samples[:,0]=samples[:,0]**2*1e6

        save=False
        #for i in range(len(index)):
        #    if index[i] == 1:
        #        plot_chain(sampler.chain, i, save=save)
        for i in range(ndim):
            print('skip')
            plot_chain(sampler.chain, i, lab[syst==0][i],
                       save=save)
        #plot_chain(sampler.chain, 1, save=save)
        #plot_chain(sampler.chain, 2, save=save)
        #plot_chain(sampler.chain, 10, save=save)
        #plot_chain(sampler.chain, 8, save=save)
        plt.close()
        plt.clf()


        lnprior=np.array(sampler.blobs).flatten()
        fig = corner.corner(samples, labels=lab[syst==0]) #list of params
        plt.show()
        accept=sampler.acceptance_fraction
        print('accept rate: ', accept)

        #plt.savefig("mcmc_corner.png")
        plt.clf()
        plt.close()

        p_mcmc = [(v[1], v[2]-v[1], v[1]-v[0]) for v in zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0))]
        print(p_mcmc)


        params=np.zeros(len(p_mcmc))
        param_errs = np.zeros_like(params)
        for i, tup in enumerate(p_mcmc):
            params[i]=tup[0]
            param_errs[i]=np.mean(tup[1:])

        p0[syst==0]=params
        perror[syst==0] = param_errs
        depth=p0[0]/1e6
        depth_err=perror[0]/1e6
        epoch=p0[2]
        epoch_err=perror[2]
        inc=p0[8]
        inc_err=perror[8]
        ar=p0[9]
        ar_err=perror[9]
        c=p0[10:14]
        c_err=perror[10:14]

        if transit==True:
            p0[0]=(p0[0]/1e6)**.5
        else:
            p0[0]=p0[0]/1e6

        phase = (x-params[2])/Per
        phase -= np.floor(phase)
        phase[phase > 0.5] = phase[phase > 0.5] -1.0
        lc_model=get_lightcurve_model(p0, x, transit=transit)
        model=get_sys_model(p0, x, phase, exptime,
                            orbit_start, orbit_end, lc=lc_model)
        systematic_model = model/lc_model


        corrected = y / (systematic_model)
        fit_residuals = (y - model)
        fit_err = err*params[1]
        rms = np.std(fit_residuals)*1e6
        ratio = rms/phot_err


        """
        # Smooth Transit Model: change this from phase to time
        time_smooth = (np.arange(4000)*0.00025-.5)*Per+params[2]
        phase_smooth=np.arange(4000)*.00025-.5
        smooth_model=get_lightcurve_model(params, time_smooth, c1, c2, c3
                                          , c4, Per)
        plt.clf()
        plt.errorbar(phase, corrected, fit_err, marker='o', color='blue', ecolor='blue', ls='')
        plt.plot(phase_smooth, smooth_model)
        plt.xlim([phase[0]-(phase[1]-phase[0]), phase[-1]+(phase[1]-phase[0])])
        plt.title('HAT-P-41b WFC3 whitelight curve: Zhou Ramp')
        plt.xlabel('Phase')
        plt.ylabel('Normalized Flux')
        #plt.show()
        plt.savefig('median_model_corrected.png')
        plt.clf()

        plt.errorbar(phase, y, fit_err, marker='o', color='blue', ecolor='blue', ls='')
        plt.plot(phase, model, color='orange',ls='-')
        plt.xlim([phase[0]-(phase[1]-phase[0]), phase[-1]+(phase[1]-phase[0])])
        plt.title('HAT-P-41b WFC3 whitelight curve: Zhou Ramp')
        plt.xlabel('Phase')
        plt.ylabel('Normalized Flux')
        #plt.show()
        plt.savefig('median_sys_model.png')
        plt.clf()

        plt.errorbar(phase, fit_residuals, fit_err, marker='o', color='blue', ecolor='blue', ls='')
        plt.plot(phase, np.zeros_like(phase), 'r')
        #plt.show()
        plt.savefig('mcmc_residuals.png')
        plt.clf()
        print np.std(fit_residuals)*1e6/np.median(phot_err)
        plt.hist((fit_residuals/fit_err)/np.sum(fit_residuals/fit_err), 20)
        plt.savefig('residual_hist.png')
        print time.time()-start_time"""



        if savewl:
            # Save results
            # convert to
            cols=['Depth', 'RMS', 'Photon Error', 'Ratio', 'Norm index1', 'Norm index2', 'rprs'
                  , 'Zero-flux' , 'Event time', 'Slope', 'ramp1', 'ramp2','ramp3', 'ramp4'
                  , 'inc','ar', 'c1', 'c2', 'c3', 'c4', 'Period', 'eclipse depth', 'Intrinsic Count']
            data=[depth, rms, phot_err, ratio, orbit_start, orbit_end] + p0.tolist()
            errors= np.append([depth_err, 0, 0, 0, 0, 0], perror)
            ind2=pd.MultiIndex.from_product([[savewl],['Values', 'Errors']])
            wl_params = pd.DataFrame(np.vstack((data,errors)), columns=cols, index=ind2)
            wl_params['Transit']=transit

            try:
                cur=pd.read_csv('./wl_ramp_params.csv', index_col=[0,1])
                #cur=cur.drop(savewl, level=0)
                cur=cur.drop(savewl, level=0, errors='ignore')
                cur=pd.concat((cur,wl_params), sort=False)
                #cur=cur[~cur.index.duplicated(keep='first')]
                cur.to_csv('./wl_ramp_params.csv', index_label=['Obs', 'Type'])
            except IOError:
                wl_params.to_csv('./wl_ramp_params.csv', index_label=['Obs', 'Type'])

        return [depth, depth_err, epoch
                , epoch_err, inc, inc_err,
                ar, ar_err, c, c_err]
