
import sys
import time
import time
import shutil
import os
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

def max_like(p_start, x, y, yerr, *extras):
    """ Function to maximize log likelihood. Gives parameter values at max
    log likelihood so we can initialize our walkers to those values.
    Outdated"""
    nll = lambda *args: -lnlike(*args)
    #exptime, orbit_start, orbit_end, transit = extras
    result = op.minimize(nll, p_start, args=(x, y, yerr, extras[0]
                                             , extras[1], extras[2], extras[3]
                                             , extras[4], extras[5], extras[6]
                                             , extras[7], extras[8], extras[9]
                                             , extras[10], extras[11], extras[12]))

    #result = op.minimize(nll, p_start, args=(x, y, yerr, model
    #                                         , exptime, orbit_start
    #                                         , orbit_end, transit))
    p_max= result["x"]
    #p_max[0]=np.abs(p_max[0])
    return p_max

def lnprob(p, x, y, yerr, p_start, syst, *args):

    params = p_start.copy()
    params[syst==0] = p
    lp=lnprior(params, p_start, syst)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(params, x, y, yerr, *args)



def lnprior(theta, theta_initial, syst, transit=True):

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
        if syst[8] == 0: test[8]=scipy.stats.norm.pdf(theta[8], theta_initial[8], 10)
        if syst[9] == 0: test[9]=scipy.stats.norm.pdf(theta[9], theta_initial[9], 10)
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


def binramp(p_start
            , img_date
            , allspec
            , allerr
            , intrinsic_count
            , exptime
            , visit
            , binsize
            , beta=1.0
            , plotting = False
            , savemc = False
            , save= False
            , transit=False
            , nbin='test'):


    nexposure = len(img_date)

    # SET THE CONSTANTS USING THE PRIORS
    #perr=perr*5
    rprs = p_start[0]
    epoch = p_start[1]
    inclin = p_start[2]
    a_r = p_start[3]
    Per = p_start[4]
    fp=p_start[5] #eclipse depth (planetary flux)

    if transit==True:
        depth=rprs
    else:
        depth=fp
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
    dtraps=1.0
    dtrapf=1.0

    #PLACE ALL THE PRIORS IN AN ARRAY
    p0 = [depth,flux0,epoch,m,traps, trapf, dtraps, dtrapf
          ,inclin,a_r,c1,c2,c3,c4,Per,fp, intrinsic_count]
    system=[0,0,1,0,0,0,0,0,1,1,1,1,1,1,1,1,0]

    nParam=len(p0)
    lab= np.array(['Depth', 'Norm', 'Epoch', 'Slope', 'R_ts'
                   , 'R_tf', 'R_dts', 'R_dtf', 'Inc', 'ars'
                   ,'c1','c2','c3','c4','Period', 'Eclipse Depth', 'Count'])
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
    err*=beta


    parinfo=[]
    for i in range(len(p0)):
        dic = {'fixed':system[i]}
        if lab[i]=='R_ts' or lab[i]=='R_tf' or lab[i]=='R_dts' or lab[i]=='R_dtf' or lab[i]=='count':
            dic = {'fixed':system[i], 'limits': [-100,500]}
        if lab[i]=='Inc':
            dic = {'fixed':system[i], 'limits': [0.0,90.0]}
        parinfo.append(dic)
    fa=(x,y,err,exptime,transit, orbit_start, orbit_end)
    m2=kmpfit.Fitter(residuals=residuals, data=fa, parinfo=parinfo, params0=p0)
    m2.fit()
    params = m2.params
    perror = m2.stderr
    depth_err_nlls=perror[0]*params[0]*2.0
    print('Reduced chi^2 NLLS = %.3f' % (m2.rchi2_min))


    # Re-Calculate each of the arrays dependent on the output parameters
    phase = (x-params[2])/params[14]
    phase -= np.floor(phase)
    phase[phase > 0.5] = phase[phase > 0.5] -1.0

    lc_model=get_lightcurve_model(params, x, transit=transit)
    #plt.plot(x, lc_model)
    systematic_model=get_sys_model(params, x, phase, exptime,
                                   orbit_start, orbit_end, lc=lc_model)
    lc_model=1.0
    w_model=lc_model*systematic_model
    w_residuals = (y - w_model)
    std = np.std(w_residuals)
    print('Ratio: %.2f' % (std/np.median(err)))
    #plt.plot(x, w_model, 'ro')
    #plt.plot(x, systematic_model, 'go')
    #plt.errorbar(x, y,err, color='b', marker='x')
    #plt.show()
    #plt.close()
    #plt.clf()
    #plt.plot(x, w_residuals, 'bo')
    #plt.show()


    #plt.show()




    if savemc:
        mc_dir = './emcee_runs/ramp/' + visit + '/' + str(binsize) + '/' + nbin
        try:
            os.makedirs(mc_dir)
        except OSError:
            if os.path.isdir(mc_dir):
                shutil.rmtree(mc_dir)
                os.makedirs(mc_dir)
            else:
                raise
    else:
        mc_dir = False
    start_time=time.time()


    p0 = np.array(params).copy()
    perr = np.array(perror).copy()
    print('NLLS', p0)
    #p_max=max_like(p0, x, y, err, c1, c2, c3
    #               , c4, Per, exptime, orbit_start, orbit_end
    #               , epoch, inclin, a_r, rprs, transit)
    #print p_max
    syst=np.array([0,0,1,0,0,0,0,0,1,1,1,1,1,1,1,1,0])
    if transit==False:
        syst[0]=1
        syst[15]=0

    ndim, nwalkers = len(p0[syst==0]), int(len(p0[syst==0])*2.5/2)*2

    # The following block is for weird mpfit behavior cropping up
    # in the bins. It sometimes finds the error on the ramp parameters
    # to be exactly 0, in which case the sampling blob all starts in
    # one spot and never moves. To avoid this:
    # If the error and value are 0, set the error to 1 to move around.
    # If the value isn't 0, set the error to be 5% of the value.
    if np.any(perr[syst==0]==0.0):
        ix = np.where((syst==0) & (perr==0.0))[0]
        for i in ix:
            if p0[i] != 0.0:
                perr[i] = np.abs(p0[i])*.05
            else:
                perr[i]=1.0


    pos=np.array([p0[syst==0] + 5*perr[syst==0]*np.random.randn(ndim)
                  for i in range(nwalkers)])
    # Make sure ramp params start off positive to avoid issues with prior.
    pos[:,3:7][pos[:,3:7]>500]=499
    pos[:,3:7][pos[:,3:7]<-100]=-99
    #pos[:,-1]=np.abs(pos[:,-1])

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob
                                    , args=(x, y, err, p0, syst, exptime
                                            , orbit_start, orbit_end))

    nsteps = 25000
    for i, result in enumerate(sampler.sample(pos, iterations=nsteps)):
        if (i+1) % 100 == 0:
            print("{0:5.1%}".format(float(i) / nsteps))
    #sampler.run_mcmc(pos, nsteps)

    burn = 5000
    for pp in range(len(p0[syst==0])):
        #print lab[syst==0][pp]
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
    if savemc:
        plt.savefig(mc_dir+'/acor_est.pdf')
        #plt.close()
        plt.clf()
    else:
        plt.show()
    #plt.show()
    taus = np.zeros_like(p0[syst==0])
    for pp in range(len(p0[syst==0])):
        #print lab[syst==0][pp]
        chain = sampler.chain[:,burn:,pp]
        taus[pp] = autocorr_new(chain)

    print(taus)
    print(' Mean integrated auto time: %.2f' % np.mean(taus))

    samples = sampler.chain[:,burn:,:].reshape((-1, ndim))
    samples_orig = samples.copy()

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
    if savemc:
        plt.savefig(mc_dir+'/model_fits.pdf')
        #plt.close()
        plt.clf()
    else:
        plt.show()


    if transit==True:
        samples[:,0]=samples[:,0]**2*1e6
    else:
        samples[:,0]*=1e6

    for i in range(ndim):
        plot_chain(sampler.chain, i, lab[syst==0][i],
                   save=savemc, mc_dir=mc_dir)
    #plt.close()
    plt.clf()
    fig = corner.corner(samples, labels=lab[syst==0],
                        show_titles=True, quantiles=[.16,.5,.84])
    #plt.show()
    if savemc:
        plt.savefig(mc_dir+'/marg_corner.pdf')
        #plt.close()
        plt.clf()
    else:
        plt.show()
    accept=sampler.acceptance_fraction
    print('accept rate: ', np.median(accept))
    #plt.clf()
    #plt.close()
    p_mcmc = [(v[1], v[2]-v[1], v[1]-v[0]) for v in zip(*np.percentile(samples, [16, 50, 84],
                                    axis=0))]



    params=np.zeros(len(p_mcmc))
    param_errs=np.zeros(len(p_mcmc))
    for i, tup in enumerate(p_mcmc):
        params[i]=tup[0]
        param_errs[i]=np.mean(tup[1:])
    p0[syst==0]=params
    perror[syst==0]=param_errs
    depth=p0[0]/1e6
    depth_err=param_errs[0]/1e6


    if transit==True:
        p0[0]=(p0[0]/1e6)**.5
        perror[0]=perror[0]/1e6/2.0/p0[0]
    else:
        p0[0]=p0[0]/1e6
        perror[0]=perror[0]/1e6

    phase = (x-epoch)/Per
    phase -= np.floor(phase)
    phase[phase > 0.5] = phase[phase > 0.5] -1.0
    lc_model=get_lightcurve_model(p0, x, transit=transit)
    model=get_sys_model(p0, x, phase, exptime,
                                   orbit_start, orbit_end, lc=lc_model)
    systematic_model = model/lc_model
    corrected = y / (systematic_model)
    fit_residuals = (y - model)
    fit_err = err*params[1]
    rms = np.std(fit_residuals)
    ratio = rms/phot_err*1e6

    time_smooth = (np.arange(500)*0.002-.5)*Per+epoch
    phase_smooth=np.arange(500)*.002-.5
    smooth_model=get_lightcurve_model(p0, time_smooth, transit=transit)

    # chi2
    dof = len(img_date) - np.sum(syst==0)
    rchi2 = np.sum(fit_residuals**2/fit_err**2)/dof

    print('Reduced chi^2 = %.3f' % rchi2)

    #mc_depth_err = (p_mcmc[0][1]+p_mcmc[0][2])/2.0
    mc_model_ratio = depth_err/depth_err_nlls
    cols = ['Median', '16th percentile', '84th percentile']
    mc_results = pd.DataFrame(p_mcmc, columns=cols)
    mc_results['Parameter'] = lab[syst==0]
    mc_results = mc_results.set_index('Parameter')
    mc_results['MCMC_NLLS Ratio'] = mc_model_ratio
    mc_results['Residuals Ratio'] = rms/np.median(fit_err)
    mc_results['Rchi2'] = rchi2

    print(mc_results)
    if savemc:
        mc_results.to_csv(mc_dir+'/best_params.csv')

    # Autocorrelation function of the light curve residuals
    ac_resids = autocorr_func_1d(fit_residuals, norm=True)
    mins = np.zeros_like(ac_resids)
    mins[ac_resids<0] = ac_resids[ac_resids<0]
    maxs = np.zeros_like(ac_resids)
    maxs[ac_resids>0]=ac_resids[ac_resids>0]
    #plt.close()
    lags = np.arange(len(ac_resids))
    plt.plot(ac_resids, 'bo')
    plt.vlines(lags, mins, maxs, 'b')
    sig = 0.05 # 95% confidence interval
    conf = scipy.stats.norm.ppf(1-sig/2.)/np.sqrt(len(fit_residuals))
    plt.plot(lags, np.zeros_like(ac_resids)+conf, color='r', label='2 sigma range')
    plt.plot(lags, np.zeros_like(ac_resids)-conf, color = 'r')
    plt.title('Autocorrelation function of residuals')
    plt.legend()

    if savemc:
        plt.savefig(mc_dir+'/acor_resids.pdf')
        #plt.close()
        plt.clf()
    else:
        plt.show()

    pickle_dict = {'sampler': sampler, 'ndim': ndim,
                   'nwalkers':nwalkers, 'syst':syst,'lab':lab,
                   'taus':taus, 'burn':burn}
    if savemc:
        pickle.dump(pickle_dict
                    , open( mc_dir +"/sampler.p", "wb" ) )




    #plt.errorbar(phase, corrected, fit_err, marker='o', color='blue', ecolor='blue', ls='')

    #for params in samples[np.random.randint(len(samples), size=100)]:
    #    params[0]=np.sqrt(params[0]/1e6)
    #    params[-1]=np.sqrt(1.0+ np.exp(2.0*params[-1]))
    #    fit_err = err*params[-1]/params[1]
    #    lm=get_lightcurve_model(params, x, c1, c2, c3
    #                            , c4, Per, epoch, inclin, a_r, transit=transit)
    #    sm=get_sys_model(params, x, phase, exptime, orbit_start, orbit_end)
    #    model=params[1]*lm*sm
    #    resids=(y-model)/params[1]
    #    mid=len(phase)/2
    #    plt.errorbar(phase[mid], resids[mid], fit_err[mid], color='k'
    #                 , ecolor='k', alpha=0.1, marker='o', ls='')
    #    plt.plot(phase, resids, 'ok', alpha=0.1, ls='')

    #plt.errorbar(phase, np.zeros_like(phase), color='blue')
    #plt.xlim([phase[0]-(phase[1]-phase[0]), phase[-1]+(phase[1]-phase[0])])
    #plt.plot(phase_smooth, smooth_model)
    #plt.title('HAT-P-41b WFC3 whitelight curve: Zhou Ramp')
    #plt.xlabel('Phase')
    #plt.ylabel('Normalized Flux')
    #plt.savefig('./mcmc_figs/eresids%02d.png' % int(nbin))
    #plt.show()
    #plt.clf()
    #plt.errorbar(phase, fit_residuals, fit_err, marker='o', color='blue', ecolor='blue', ls='')
    #plt.plot(phase, np.zeros_like(phase), 'r')
    #plt.show()
    #plt.savefig('mcmc_residuals_f.png')
    #plt.clf()
    #print np.std(fit_residuals)*1e6/np.median(phot_err)
    #plt.hist((fit_residuals/fit_err)/np.sum(fit_residuals/fit_err), 20)
    #plt.clf()
    #plt.savefig('residual_f.png')
    #print time.time()-start_time

    #####################################################3
    if save == True:
        ################# make sure this works
        ### Two dataframes, both multi-indexed

        # To retrieveas numpy array: df.loc[visit,column].values
        # Example: wl_models_info.loc['hatp41/visit01/reverse','Params'].values[0]

        # Save all plotting stuff
        cols = ['Date', 'Flux', 'Flux Error', 'Norm Flux', 'Norm Flux Error', 'Model Phase'
                , 'Model', 'Corrected Flux', 'Corrected Flux Error', 'Residuals']

        bins=pd.DataFrame(np.vstack((x, rawflux, rawerr, y, err, phase, model
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
        data=[depth, rms,phot_err, ratio, orbit_start, orbit_end] + p0.tolist()
        errors= [depth_err, 0, 0, 0, 0, 0] + perror.tolist()
        ind2=pd.MultiIndex.from_product([[visit],[binsize],[nbin],['Values', 'Errors']])
        bin_params = pd.DataFrame(np.vstack((data,errors)), columns=cols, index=ind2)
        bin_params['Transit']=transit

        try:
            cur=pd.read_csv('./binmcmc_params.csv', index_col=[0,1, 2, 3])
            cur=cur.drop((visit, binsize,int(nbin)), errors='ignore')
            cur=pd.concat((cur,bin_params), sort=False)
            cur=cur[~cur.index.duplicated(keep='first')]
            cur.to_csv('./binmcmc_params.csv', index_label=['Obs','Bin Size','Bin', 'Type'])
        except IOError:
            bin_params.to_csv('./binmcmc_params.csv', index_label=['Obs','Bin Size', 'Bin', 'Type'])

        try:
            curr=pd.read_csv('./binmcmc_data.csv', index_col=[0,1, 2])
            curr=curr.drop((visit, binsize,int(nbin)), errors='ignore')
            curr=pd.concat((curr,bins), sort=False)
            #curr=curr[~curr.index.duplicated(keep='first')]
            curr.to_csv('./binmcmc_data.csv', index_label=['Obs', 'Bin Size', 'Bin'])
        except IOError:
            bins.to_csv('./binmcmc_data.csv', index_label=['Obs', 'Bin Size', 'Bin'])

        try:
            currr=pd.read_csv('./binmcmc_smooth.csv', index_col=[0,1,2])
            currr=currr.drop((visit, binsize,int(nbin)), errors='ignore')
            currr=pd.concat((currr,bin_smooth), sort=False)
            # currr=currr[~currr.index.duplicated(keep='first')]
            currr.to_csv('./binmcmc_smooth.csv', index_label=['Obs', 'Bin Size', 'Bin'])
        except IOError:
            bin_smooth.to_csv('./binmcmc_smooth.csv', index_label=['Obs', 'Bin Size', 'Bin'])


    return [depth, depth_err, rms*1e6]

