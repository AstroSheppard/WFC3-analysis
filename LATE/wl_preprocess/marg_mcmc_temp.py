
import sys
import time
import shutil
import os

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from tqdm import tqdm
import pandas as pd
import scipy.optimize as op
import scipy.stats
import batman
import emcee
import corner
import pickle


from wave_solution import orbits
from kapteyn import kmpfit


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
    if save == True:
        plt.savefig(mc_dir+'/'+lab+'_chains.pdf')
        plt.clf()
    else:
        plt.show()
    return None

def get_sys_model(p, phase, HSTphase, sh, dir_array):
    # The first line makes sure the forward slope only applies to
    # the forward scans and the reverse slope (p[26]) only applies to reverse
    # scans.
    fslope = phase*p[19] + phase*phase*p[20] - p[21]*np.exp(-1.0*phase/p[22]) - p[23]*np.log(phase+p[24])
    rslope = phase*p[26] + phase*phase*p[27] - p[28]*np.exp(-1.0*phase/p[29]) - p[30]*np.log(phase+p[31])
    # Hack to fit for one slope for bidrectional scan
    temp = np.zeros_like(dir_array)
    # replace temp with dir_array to go back to separate f/r slopes
    #temp = dir_array


    systematic_model = ((fslope*(1-temp)+rslope*temp + 1.0)
                        * (HSTphase*p[2] + HSTphase**2.*p[3] + HSTphase**3.*p[4]
                           + HSTphase**4.*p[5] + 1.0)
                        * (sh*p[6] + sh**2.*p[7] + sh**3.*p[8] + sh**4.*p[9]
                           + 1.0))
    systematic_model = p[18]*systematic_model*(1-dir_array) + p[25]*systematic_model*dir_array
    return systematic_model

def get_lightcurve_model(p, date, limb="nonlinear", transit=False):

    params=batman.TransitParams()
    params.w=90.
    params.ecc=0
    params.rp=p[0]
    tc=p[1]
    params.inc=p[10]
    params.a=p[11]
    params.per=p[16]
    depth=p[17]

    if transit==True:
        params.t0=tc
        if limb=="nonlinear":
            params.u=p[12:16]
            params.limb_dark="nonlinear"
        if limb=="quadratic":
            params.u=p[12:14]
        params.limb_dark=limb
        m=batman.TransitModel(params, date, fac=0.0315)
        model=m.light_curve(params)
    else:
        params.fp=depth
        params.t_secondary=tc
        params.u=[]
        params.limb_dark="uniform"
        m=batman.TransitModel(params, date, transittype="secondary")
        model=m.light_curve(params)

    return model

def lightcurve(p, x, sh, HSTphase, dir_array, transit=False):

    """ Function used by MPFIT to fit data to lightcurve model.

    Inputs: p: input parameters that we are fitting for
    x: Date of each observation, to be converted to phase
    y: Flux of each time series observation
    err: Error on flux of each time series point
    sh: Parameter for wavelength shift on the detector (for each exposure)
    rprs: Ratio of planet to star radii
    transit: True for transit, false for eclipse

    Output: Returns weighted deviations between model and data to be minimized
    by MPFIT. """
    # Add in dir_array here
    # Old param order= [rprs,flux0,tc,m,HSTP1,HSTP2,HSTP3,HSTP4,HSTP5,HSTP6,xshift1
    # ,xshift2,xshift3,xshift4,xshift5,xshift6,inclin,A_R,c1,c2,c3,c4,Per,fp]

    # New param order
    #p0 = [rprs,epoch,
    #      HSTP1,HSTP2,HSTP3,HSTP4,
    #      xshift1 ,xshift2 ,xshift3,xshift4,
    #      inclin,a_r,c1,c2,c3,c4,
    #      Per,fp,fnorm, flinear, fquad,
    #      fexpb, fexpc, flogb, flogc,
    #      rnorm, rlinear, rquad,
    #      rexpb, rexpc, rlogb, rlogc]
    Per = p[16]

    phase = (x-p[1])/Per
    phase = phase - np.floor(phase)
    phase[phase > 0.5] = phase[phase > 0.5] - 1.0
    #sss

    systematic_model=get_sys_model(p,phase,HSTphase,sh, dir_array)
    lcmodel=get_lightcurve_model(p, x, transit=transit)
    model=lcmodel*systematic_model
    #model=lcmodel * p[1] * systematic_model

    return model

def lnlike(p,x,y, yerr, *args):
    """ p i paramters of model, model is the name of the function of the model
    args contains any extraneous arguments used in model calculation, like sh in
    marginalization. """
    theory=lightcurve(p,x,*args)
    inv_sigma=1.0/yerr/yerr
    return -.5*np.sum((y-theory)**2*inv_sigma - np.log(inv_sigma))

def max_like(p_start, x, y, yerr, perr, *extras):
    """ Function to maximize log likelihood. Gives parameter values at max
    log likelihood so we can initialize our walkers to those values."""
    nll = lambda *args: -lnlike(*args)
    bounds = ((0,.5), (.5, 1.5), (p_start[2]-1.0, p_start[2]+1.0), (-5, 5), (0, 200), (0, 200)
              ,(0, 200), (0, 200), (0, 1e4), (p_start[9],p_start[9])
              , (p_start[10], p_start[10]))
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
        return -np.inf
    return lp + lnlike(params, x, y, yerr, *args)

def lnprior(theta, theta_initial, theta_error, syst, transit=True):

    """ Priors on parameters. For system, try both fixing and gaussian priors.
    For depth and others, do "uninformative" uniform priors over a large enough
    range to cover likely results

    Right now I'm extremely conservative. Prior is any possible value for
    open parameters (uniform), and fixed for all others. In future, I will
    update fixed with gaussian priors and uniform with more appropriate uninformative
    priors. """

    # Walkers only exist in the dim when sys == 0
    if not np.all(theta[syst==1] == theta_initial[syst==1]): return -np.inf
    # then uniform
    ind = np.where(syst==0)
    #print lab[ind]
    test=np.ones(len(theta))
    if transit==True:
        #i=0
        #print 'test %d' % i
        #i+=1
        if syst[0]==0 and not 0 < theta[0] < 0.2: return -np.inf
        if syst[1]==0 and not theta_initial[1]-0.2 < theta[1] < theta_initial[1]+0.2: return -np.inf
        if syst[2]==0 and not -4 < np.log10(np.abs(theta[2])) < 7:  return -np.inf
        if syst[3]==0 and not -4 < np.log10(np.abs(theta[3])) < 7:  return -np.inf
        if syst[4]==0 and not -4 < np.log10(np.abs(theta[4])) < 7:  return -np.inf
        if syst[5]==0 and not -4 < np.log10(np.abs(theta[5])) < 7:  return -np.inf
        if syst[6]==0 and not -4 < np.log10(np.abs(theta[6])) < 7:  return -np.inf
        if syst[7]==0 and not -4 < np.log10(np.abs(theta[7])) < 7:  return -np.inf
        if syst[8]==0 and not -4 < np.log10(np.abs(theta[8])) < 7:  return -np.inf
        if syst[9]==0 and not -4 < np.log10(np.abs(theta[9])) < 7:  return -np.inf
        if not theta[10] < 90.0: return -np.inf
        if syst[10] == 0: test[10]=scipy.stats.norm.pdf(theta[10], theta_initial[10], theta_error[10])
        if syst[11] == 0: test[11]=scipy.stats.norm.pdf(theta[11], theta_initial[11], theta_error[11])

        if syst[17] == 0 and not 0 < theta[17] < 0.2: return -np.inf
        if not .5 < theta[18] < 20:  return -np.inf
        if syst[19]==0 and not -5 < theta[19] < 3:  return -np.inf

        if syst[20]==0 and not -5 < theta[20] < 3:  return -np.inf
        print('test 20')
        if syst[21]==0 and not -4 < np.log10(np.abs(theta[21])) < 3:  return -np.inf
        print('test 21')
        if syst[22]==0 and not -10 < theta[22] < 10:  return -np.inf
        print('test 22')
        if syst[23]==0 and not  -4 < np.log10(np.abs(theta[23])) < 3:  return -np.inf
        print('test 23')
        if syst[24]==0 and not .5 < theta[24] < 200:  return -np.inf
        print('test 24')
        if not .9 < theta[25] < 1.1:  return -np.inf
        if syst[26]==0 and not -5 < theta[16] < 3:  return -np.inf
        if syst[27]==0 and not -5 < theta[27] < 3:  return -np.inf
        if syst[28]==0 and not -4 < np.log10(np.abs(theta[28])) < 3:  return -np.inf
        if syst[29]==0 and not -10 < theta[29] < 10:  return -np.inf
        if syst[30]==0 and not  -4 < np.log10(np.abs(theta[30])) < 3:  return -np.inf
        if syst[31]==0 and not .5 < theta[31] < 200:  return -np.inf
        if np.isfinite(np.sum(np.log(test))):
            return np.sum(np.log(test))
        else:
            print('how is this infinite?')
            return -np.inf
        #print 'priors are good'
        #return 0.0
    else:
        sys.exit("Didn't do eclipses yet")

def residuals(p,data):
    x, y, err, sh, HSTphase, dir_array, transit = data
    ym=lightcurve(p, x, sh, HSTphase, dir_array, transit=transit)
    return (y-ym)/err

def get_shift(allspec):
    nexposure=allspec.shape[0]
    sh = np.zeros(nexposure)
    nLag=3
    inp1=allspec[-1,:]-allspec[-1,:].mean()

    for i in trange(nexposure, desc='Performing cross correlation'):
        # Subtract mean to mimic
        inp2=allspec[i,:]-allspec[i,:].mean()
        corr_tuple = plt.xcorr(inp1, inp2,  maxlags=nLag)
        lag,corr=corr_tuple[0],corr_tuple[1]
        mx=np.argmax(corr)
        srad=3
        sublag=lag[max(mx-srad,0):max(mx+srad,2*nLag+1)]
        subcorr=corr[max(mx-srad,0):max(mx+srad,2*nLag+1)]
        p=np.polyfit(sublag, subcorr, 2)
        sh[i]=p[1]/2./p[0]
    return sh

def systematic_model_grid_selection(size=4,
                                    dir_array=np.ones(1),
                                    transit=False,
                                    linear=True,
                                    quad = False,
                                    exp = False,
                                    log = False):
    """ Returns model grid that indicates which parameters
    will be openly fit for and which will be fixed. Larger
    size will test higher powers. """
    if size not in [2,4,6]:
        sys.exit('Grid size can only be 2, 4, or 6')
    # MODEL GRID FOR JUST ECLIPSE DEPTH
    if size == 2:
        grid = [[0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,1,0,0,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,1,0,0,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,1,0,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,0,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,0,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,0,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1]]
    if size == 4:
        # MODEL GRID UPTO THE 4th ORDER for HST & DELTA_lambda
        grid = [[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]


    grid=np.asarray(grid)
    #if transit == True:
    #    grid[:,18:22]=0


    # Remove the 5/6 powers of hst phase and wavelength shift (50 x 20)
    grid = np.delete(grid, [8, 9, 14, 15], axis=1)
    grid = np.vstack((grid.T,grid[:,1])).T
    grid = np.vstack((grid.T,grid[:,3])).T
    grid = np.delete(grid, [1, 3], axis=1)
    # Add 12 spots in grid for bi-directional scan normalization and different slopes (50 x 32)
    grid = np.vstack((grid.T,np.ones((12, grid.shape[0])))).T

    #linear = True
    #quad = True
    #exp = True
    #log = True
    #p0 = [rprs,epoch,
    #      HSTP1,HSTP2,HSTP3,HSTP4,
    #      xshift1 ,xshift2 ,xshift3,xshift4,
    #      inclin,a_r,c1,c2,c3,c4,
    #      Per,fp,fnorm, flinear, fquad,
    #      fexpb, fexpc, flogb, flogc,
    #      rnorm, rlinear, rquad,
    #      rexpb, rexpc, rlogb, rlogc]
    grid[:25, 19] = 1
    grid[25:, 19] = int(not linear)
    # Add another 25 models for quad slope
    # Set values to 1 to fix (and ignore this slope in fits), or to zero to
    # leave open.
    quad_grid = grid[:25,:].copy()
    quad_grid[:, 19] = int(not quad)
    quad_grid[:, 20] = int(not quad)
    grid = np.hstack((grid.T, quad_grid.T)).T

    # Add another 25 models for exp slope
    exp_grid = grid[:25,:].copy()
    exp_grid[:, 21] = int(not exp)
    exp_grid[:, 22] = int(not exp)
    grid = np.hstack((grid.T, exp_grid.T)).T
    # Add another 25 models for log slope
    log_grid = grid[:25,:].copy()
    log_grid[:, 23] = int(not log)
    log_grid[:, 24] = int(not log)
    grid = np.hstack((grid.T, log_grid.T)).T

    # If dir_array isn't all ones or all zeros, then turn on reverse slope
    # and normalization whenever forward slope/norm is on
    if 0 < np.sum(dir_array) < len(dir_array):
        grid[:, 25:] = grid[:, 18:25]
        # Hack to fit for only 1 directional slope even with forward/reverse array
        grid[:, 26:] = 1.0

    return grid


def whitelight2020(p_start, img_date, allspec, allerr, dir_array, plotting=False
                   , mcmc = False, fixtime=False, norandomt=False, openinc=False
                   , openar=False, save_mcmc=False, save_model_info=False, transit=False
                   , save_name=None):
    """
  NAME:
       WHITELIGHT2020

  AUTHOR:
     Based on Hannah R. Wakeford, NASA/GSFC code 693, Greenbelt, MD 20771
     hannah.wakeford@nasa.gov

     Kyle Sheppard: Converted to python and added eclipse functionality

  PURPOSE:
     Perform Levenberg-Marquardt least-squares minimization with
     MPFIT on spectral data from HST WFC3 to compute the band-integrated light curve

  MAJOR TOPICS:
     Generate band-integrated light curve
     Determine the most favoured systematic model for the observation
     Measure the marginalised secondary eclipse depth

  CALLING SEQUENCE:;
     whitelight_eclipse(p_start, img_date, allspec,plotting=False
                        fixtime=False, norandomt=False, openinc=False, savefile=False
                        , transit=False)

INPUTS:

 P_START - priors for each parameter used in the fit passed in
 an array in the form
 p_start = [rprs,epoch,inclin,a/rs,Per, fp]

   rprs - Planetary radius/stellar radius
   epoch - center of eclipse time
   inclin - inclination of the planetary orbit
   a/Rs - semimajor axis normalized by stellar radius
   Per - Period of the planet in days
   fp - event depth

 IMG_DATE - array of time of each exposure as extracted from the .fits header (MJD)


 ALLSPEC - 2D array each row containing the target stellar
           spectrum extracted from the exposure images.

 ALLERR - Uncertainty for each pixel flux in allspec

 dir_array - Of length img_date indicating if each exposure is a forward scan (0) or reverse (1)

 savewl - True to save results

 plotting - set as True or False to see the plots

 FIXTIME - True to keep center of eclipse/transit time fixed

 NORANDOMT - True to not allow small random changes to center time

 OPENINC - True to fit for inclination

 TRANSIT - True for transit light curves, default false for eclipse

    """


    start_time = time.time()
    # TOTAL NUMBER OF EXPOSURES IN THE OBSERVATION
    nexposure = len(img_date)

    # CALCULATE THE SHIFT IN DELTA_lambda
    # sh = np.zeros(nexposure)
    # nLag=3
    # inp1=allspec[-1,:]-allspec[-1,:].mean()

    # for i in trange(nexposure, desc='Performing cross correlation'):
    #     Subtract mean to mimic
    #     inp2=allspec[i,:]-allspec[i,:].mean()
    #     corr_tuple = plt.xcorr(inp1, inp2,  maxlags=nLag)
    #     lag,corr=corr_tuple[0],corr_tuple[1]
    #     mx=np.argmax(corr)
    #     srad=3
    #     sublag=lag[max(mx-srad,0):max(mx+srad,2*nLag+1)]
    #     subcorr=corr[max(mx-srad,0):max(mx+srad,2*nLag+1)]
    #     p=np.polyfit(sublag, subcorr, 2)
    #     sh[i]=p[1]/2./p[0]

    sh = get_shift(allspec)
    HSTper = 96.36 / (24.*60.)
    HSTphase = (img_date-img_date[0])/HSTper
    HSTphase = HSTphase - np.floor(HSTphase)
    HSTphase[HSTphase > 0.5] = HSTphase[HSTphase > 0.5] -1.0

    # SET THE CONSTANTS USING THE PRIORS
    rprs = p_start[0]
    epoch = p_start[1]
    inclin = p_start[2]
    a_r = p_start[3]
    Per = p_start[4]
    fp=p_start[5] #depth
    c1=p_start[6]
    c2=p_start[7]
    c3=p_start[8]
    c4=p_start[9]
    fnorm = 1.0  # Forward norm
    rnorm = 1.0             # Reverse norm, if two direction scan

    flinear = 0.0         # Linear Slope
    fquad = 0.0         # Quadratic slope
    fexpb = 0.0         # Exponential slope factor
    fexpc = 1.0         # Exponential slope phase
    flogb = 0.0         # Log slope factor
    flogc = 1.0         # Log slope phase
    rlinear = 0.0        # Reverse scan slope
    rquad = 0.0         # Quadratic slope
    rexpb = 0.0         # Exponential slope factor
    rexpc = 1.0         # Exponential slope phase
    rlogb = 0.0         # Log slope factor
    rlogc = 1.0        # Log slope phase
    xshift1 = 0.0   # X-shift in wavelength
    xshift2 = 0.0   # X-shift^2 in wavelength
    xshift3 = 0.0   # X-shift^3 in wavelength
    xshift4 = 0.0   # X-shift^4 in wavelength
    HSTP1 = 0.0     # HST orbital phase
    HSTP2 = 0.0     # HST orbital phase^2
    HSTP3 = 0.0     # HST orbital phase^3
    HSTP4 = 0.0     # HST orbital phase^4

    #PLACE ALL THE PRIORS IN AN ARRAY
    p0 = [rprs,epoch,
          HSTP1,HSTP2,HSTP3,HSTP4,
          xshift1 ,xshift2 ,xshift3,xshift4,
          inclin,a_r,c1,c2,c3,c4,
          Per,fp,fnorm, flinear, fquad,
          fexpb, fexpc, flogb, flogc,
          rnorm, rlinear, rquad,
          rexpb, rexpc, rlogb, rlogc]

    lab = np.array(['Depth', 'Epoch', 'HST1', 'HST2'
                    , 'HST3', 'HST4', 'sh1','sh2'
                    , 'sh3', 'sh4', 'i', 'ars', 'c1'
                    , 'c2', 'c3', 'c4', 'Per', 'Eclipse Depth'
                    , 'fnorm', 'flinear', 'fquad', 'fexpb'
                    , 'fexpc', 'flogb', 'flogc', 'rnorm'
                    , 'rlinear', 'rquad', 'rexpb'
                    , 'rexpc', 'rlogb', 'rlogc' ])

    nParam=len(p0)
    # SELECT THE SYSTEMATIC GRID OF MODELS TO USE

    linear = True
    quad = False
    exp = False
    log = False
    grid = systematic_model_grid_selection(size=4, dir_array=dir_array,
                                           transit=transit, linear = linear,
                                           quad = quad, exp = exp, log = log)


    nsys = len(grid[:,0])
    #  SET UP THE ARRAYS  ;
    sys_depth = np.zeros((nsys,2))
    sys_model_x = np.zeros((nsys,500))
    sys_model = np.zeros((nsys,500))
    sys_lightcurve_x = np.zeros((nsys,nexposure))
    sys_lightcurve = np.zeros((nsys,nexposure))
    sys_lightcurve_err = np.zeros((nsys,nexposure))
    sys_residuals = np.zeros((nsys,nexposure))
    sys_params = np.zeros((nsys,nParam))
    sys_params_err = np.zeros((nsys,nParam))
    sys_evidence = np.zeros((nsys))
    sys_model_full=np.zeros((nsys,nexposure))

    phase = np.zeros(nexposure)

    # Scatter of the residuals for each model
    resid_stddev = np.zeros(nsys)

    #run 1 AIC and parameters from the fit
    run1_AIC = np.zeros(nsys)
    run1_params = np.zeros((nsys,nParam))

    #  ITERATION RUN
    #  First run 4 trials with slightly shifted center of eclipse times
    #  and secondary eclipse depths

    ntrials=5
    tcenter = np.zeros(ntrials+1)

    t1 = 5./60./24.
    tcenter[0] = epoch

    if fixtime==False and norandomt==False:
        tcenter[1:] = epoch + t1*np.random.normal(size=ntrials)

    # Test arrays ;
    AIC_test = np.zeros(ntrials+1)
    depth_test = np.zeros(ntrials+1)




    x = img_date
    y=allspec.sum(axis=1)
    err = np.sqrt(np.ma.sum(allerr*allerr, axis=1))
    #phot_err=1e6/np.median(np.sqrt(y))
    phot_err=1e6*np.median(err/y)

    # Normalised Data

    orbit_start, orbit_end=orbits('holder', x=x, y=y, transit=transit)[1]
    norm=np.median(y[orbit_start:orbit_end])

    rawerr=err.copy()
    rawflux=y.copy()
    err = err/norm
    y = y/norm

    print('----------      ------------     ------------')
    print('          1ST FIT         ')
    print('----------      ------------     ------------')

    for s, systematics in tqdm(enumerate(grid), desc='First MPFIT run'):
        system=systematics
        if fixtime==False and norandomt==False:
            for n in range(ntrials+1):
                epoch = tcenter[n]

                # Reset priors
                p0 = [rprs,epoch,
                      HSTP1,HSTP2,HSTP3,HSTP4,
                      xshift1 ,xshift2 ,xshift3,xshift4,
                      inclin,a_r,c1,c2,c3,c4,
                      Per,fp,fnorm, flinear, fquad,
                      fexpb, fexpc, flogb, flogc,
                      rnorm, rlinear, rquad,
                      rexpb, rexpc, rlogb, rlogc]
                if openinc==True: system[10] = 0
                if openar==True: system[11] = 0
                if transit==False:
                    system[0]=1
                    system[17]=0
                parinfo=[]
                for i in range(len(p0)):
                    dic = {'fixed':system[i]}
                    if lab[i]=='flogc' or lab[i]=='rlogc':
                        dic = {'fixed':system[i], 'limits': [0.5,None]}
                    if lab[i]=='fexpc' or lab[i]=='rexpc':
                        dic = {'fixed':system[i], 'limits': [0.0001,None]}
                    parinfo.append(dic)


                #fa = {'x':x, 'y':y, 'err':err, 'sh':sh, 'HSTphase':HSTphase, 'transit':transit}
                fa=(x,y,err,sh,HSTphase,dir_array,transit)
                m1=kmpfit.Fitter(residuals=residuals, data=fa, parinfo=parinfo, params0=p0)
                m1.fit()
                #m = mpfit.mpfit(lightcurve,functkw=fa,parinfo=parinfo, fastnorm=True)
                params_test=m1.params

                # For each tested epoch time, get the depth and the AIC
                # Make sure this is all computed correctly
                AIC_test[n]=(2*len(x)*np.log(np.median(err))+len(x)*np.log(2*np.pi)
                             + m1.chi2_min + 2*m1.nfree)
                if transit==True:
                    depth_test[n] = params_test[0]*params_test[0]
                else:
                    depth_test[n]=params_test[17]

            # Find the epoch time with the lowest AIC. Use it (or the average of
            # values if multiple) to get starting depth and epoch time for
            # next round of fits.
            best = np.argmin(AIC_test)
            print(best)
            print('Best eclipse depth prior =', depth_test[best])
            print('Best center of eclipse prior =', tcenter[best])
            depth = np.mean(depth_test[best])
            epoch = np.median(tcenter[best])

            if transit==True:
                rprs=np.sqrt(depth)
            else:
                fp=depth


        #Re-run the fitting process with the best prior as defined by the iterative fit
        p0 = [rprs,epoch,
              HSTP1,HSTP2,HSTP3,HSTP4,
              xshift1 ,xshift2 ,xshift3,xshift4,
              inclin,a_r,c1,c2,c3,c4,
              Per,fp,fnorm, flinear, fquad,
              fexpb, fexpc, flogb, flogc,
              rnorm, rlinear, rquad,
              rexpb, rexpc, rlogb, rlogc]

        # MPFIT ;;;;;;;;;;;;;;;;;;;;;;;

        if fixtime==True: system[1] = 1
        if openinc==True: system[10] = 0
        if openar==True: system[11] = 0
        if transit==False:
            system[0]=1
            system[17]=0
        parinfo=[]
        for i in range(len(p0)):
            # parinfo.append({'value':p0[i], 'fixed':system[i]
            #                 , 'limited':[0,0], 'limits':[0.,0.]})
            dic = {'fixed':system[i]}
            if lab[i]=='flogc' or lab[i]=='rlogc':
                dic = {'fixed':system[i], 'limits': [0.5,None]}
            if lab[i]=='fexpc' or lab[i]=='rexpc':
                dic = {'fixed':system[i], 'limits': [0.0001,None]}
            parinfo.append(dic)

        #fa = {'x':x, 'y':y, 'err':err, 'sh':sh, 'HSTphase':HSTphase, 'transit':transit}
        fa=(x,y,err,sh,HSTphase,dir_array,transit)
        m2=kmpfit.Fitter(residuals=residuals, data=fa, parinfo=parinfo, params0=p0)
        m2.fit()
        #m = mpfit.mpfit(lightcurve,functkw=fa,parinfo=parinfo, fastnorm=True)
        params_w=m2.params

        AIC=(2*len(x)*np.log(np.median(err))+len(x)*np.log(2*np.pi)
             + m2.chi2_min + 2*m2.nfree)
        if transit==True:
            print('Depth = ', np.square(params_w[0]), ' at ', params_w[1])
        else:
            print('Depth = ', params_w[17], ' at ', params_w[1])

        # Re-Calculate each of the arrays dependent on the output parameters
        phase = (x-params_w[1])/params_w[16]
        phase -= np.floor(phase)
        phase[phase > 0.5] = phase[phase > 0.5] -1.0

        # LIGHT CURVE MODEL: calculate the eclipse model for the resolution of the data points
        # this routine is from MANDEL & AGOL (2002)

        systematic_model = get_sys_model(params_w, phase, HSTphase, sh, dir_array)
        lc_model = get_lightcurve_model(params_w, x, transit=transit)
        w_model = lc_model * systematic_model
        w_residuals = (y - w_model)

        resid_stddev[s] = np.std(w_residuals)
        run1_AIC[s] = AIC
        run1_params[s,:] = params_w


    #######################################

    #Determine which of the systematic models initially gives the best fit
    top = np.argmin(run1_AIC)

    # Scale error by resid_stddev[top]
    std=resid_stddev[top]
    if np.median(err) < std:
        scale=std/np.median(err)
        print(scale)
        #print 'Not inflating by this'
    else:
        scale=1
    #scale=1
    error=err*scale


    print('----------      ------------     ------------')
    print('         FINAL FIT        ')
    print('----------      ------------     ------------')
    for s, systematics in tqdm(enumerate(grid), desc='Final MPFIT run'):

        # Define the new priors as the parameters from the best fitting
        # systematic model
        p0=run1_params[s,:]
        if fixtime==True: systematics[1] = 1
        if openinc==True: systematics[10] = 0
        if openar==True: systematics[11] = 0
        if transit==False:
            systematics[0]=1
            systematics[17]=0
        parinfo=[]

        for i in range(len(p0)):
            dic = {'fixed':systematics[i]}
            if lab[i]=='flogc' or lab[i]=='rlogc':
                dic = {'fixed':systematics[i], 'limits': [0.5,None]}
            if lab[i]=='fexpc' or lab[i]=='rexpc':
                dic = {'fixed':systematics[i], 'limits': [0.0001,None]}
            parinfo.append(dic)

        fa=(x,y,error,sh,HSTphase,dir_array,transit)
        m2=kmpfit.Fitter(residuals=residuals, data=fa, parinfo=parinfo, params0=p0)
        m2.fit()
        params=m2.params
        perror=m2.xerror
        nfree=m2.nfree
        #AIC=m2.rchi2_min + nfree
        AIC=(2*len(x)*np.log(np.median(error))+len(x)*np.log(2*np.pi)
             + m2.chi2_min + 2*nfree)
        stderror=m2.stderr
        print()
        print('Model ', s)
        print('reduced chi^2:', m2.rchi2_min)
        print('dof', m2.dof)
        print()

        if transit==True:
            print('Depth = ',np.square(params[0]), ' at ', params[1])
        else:
            print('Depth = ',params[17], ' at ', params[1])

        # Re-Calculate each of the arrays dependent on the output parameters
        phase = (x-params[1])/params[16]
        phase -= np.floor(phase)
        phase[phase > 0.5] = phase[phase > 0.5] -1.0

        # --------------------- #
        #        EVIDENCE       #
        #sigma_points = np.median(error)
        #Npoint = len(x)
        # EVIDENCE BASED ON AIC ;
        #sys_evidence[s] = -Npoint*np.log(sigma_points)-0.5*Npoint*np.log(2*np.pi)-0.5*(AIC+nfree)
        sys_evidence[s]=-.5*AIC
        systematic_model=get_sys_model(params, phase, HSTphase, sh, dir_array)
        lc_model=get_lightcurve_model(params, x, transit=transit)
        #model=params[1]*lc_model*systematic_model
        model=lc_model*systematic_model
        corrected = y / systematic_model
        fit_residuals = (y - model)
        ########
        #fit_err = error/params[18]
        fit_err = error*(1-dir_array)*params[18] + error*dir_array*params[25]
        #######



        # Smooth Transit Model: change this from phase to time
        time_smooth = (np.arange(500)*0.002-.5)*params[16]+params[1]
        phase_smooth=np.arange(500)*.002-.5
        smooth_model=get_lightcurve_model(params, time_smooth, transit=transit)

        # PLOTTING
        if plotting == True:
            if s > 0: plt.close()
            plt.errorbar(img_date, y, error,ecolor='red', color='red', marker='o', ls='')
            #plt.ylim([0.982, 1.005])
            plt.plot(img_date, systematic_model, color='blue', marker='o', ls='')
            plt.errorbar(img_date, corrected, fit_err, marker='x', color='green', ecolor='green', ls='')
            plt.show(block=False)


        # SAVE out the arrays for each systematic model ;
        if transit==True:
            sys_depth[s,0] = np.square(params[0])
            sys_depth[s,1] = stderror[0]*2.0*params[0]
        else:
            sys_depth[s,0] = params[17]
            sys_depth[s,1] = stderror[17]
        sys_lightcurve_x[s,:] = phase
        sys_lightcurve[s,:] = corrected
        sys_lightcurve_err[s,:] = fit_err
        sys_model_x[s,:] = phase_smooth
        sys_model[s,:] = smooth_model
        sys_residuals[s,:] = fit_residuals
        sys_params[s,:] = params
        sys_params_err[s,:] = stderror
        sys_model_full[s,:] = model

    #;;;;;;;;;;;;;;;;;;;
    #;;;;;;;;
    #;
    #;MARGINALIZATION!!!
    #;
    #;;;;;;;;
    #;;;;;;;;;;;;;;;;;;;



    # ------------------------------- ;
    #            EVIDENCE             ;
    aics = sys_evidence
    depth_array = sys_depth[:,0]
    depth_err_array = sys_depth[:,1]
    epoch_array = sys_params[:,1]
    epoch_err_array = sys_params_err[:,1]
    inc_array= sys_params[:,10]
    inc_err_array=sys_params_err[:,10]
    ar_array= sys_params[:,11]
    ar_err_array=sys_params_err[:,11]
    limb_array=sys_params[:,12:16]
    limb_err_array=sys_params_err[:,12:16]

    # Reverse sort as trying to MAXIMISE the negative log evidence
    a=np.argsort(aics)[::-1]
    best=np.argmax(aics)
    print(best)
    # print aics




    zero = np.where(aics < -300)
    if (len(zero) > 1): print('Some bad fits - evidence becomes negative')
    if (len(zero) > 24):
        sys.exit('Over half the systematic models have negative evidence, adjust and rerun')

    aics[aics < -300] = np.min(aics[aics>-300])

    beta=100.
    #beta = np.min(aics)

    w_q = (np.exp(aics-beta))/np.sum(np.exp(aics-beta))
    bestfit=np.argmax(w_q)
    n01 = np.where(w_q >= 0.1)

    stdResid = np.std(sys_residuals[bestfit,:])
    print('Evidences: ', aics)
    print('Weights: ', w_q)

    print(str(len(n01[0])) + ' models have a weight over 10%. Models: ', n01[0] , w_q[n01])
    print('Most likely model is number ' +str(bestfit) +' at weight = ' + str(np.max(w_q)))

    depth = depth_array
    depth_err = depth_err_array

    # Marganilze depth formula 15 and 16 from Wakeford 2016
    mean_depth=np.sum(w_q*depth)
    theta_qi=depth
    variance_theta_qi=depth_err*depth_err
    error_theta_i = np.sqrt(np.sum(w_q*((theta_qi - mean_depth)**2 + variance_theta_qi )))
    print('Depth = %f  +/-  %f' % (mean_depth, error_theta_i))
    marg_depth = mean_depth
    marg_depth_err = error_theta_i


    # Marganilze tcenter
    t0 = epoch_array
    t0_err = epoch_err_array
    inc=inc_array
    inc_err=inc_err_array
    ar=ar_array
    ar_err=ar_err_array

    print('Depth')
    print(depth[a])
    print(depth_err_array[a])
    print('Center Time')
    print(t0[a])
    print(t0_err[a])
    mean_t0=np.sum(w_q*t0)
    theta_t0=t0
    variance_theta_t0q = t0_err*t0_err
    error_theta_t0 = np.sqrt(np.sum(w_q*((theta_t0 - mean_t0)**2 + variance_theta_t0q )))
    print('Central time = %f +/- %f' % (mean_t0, error_theta_t0))
    marg_epoch = mean_t0
    marg_epoch_err = error_theta_t0
    # Marginalize over inclination
    if openinc==True:
        mean_inc=np.sum(w_q*inc)
        bestfit_theta_inc=inc
        variance_theta_incq = inc_err*inc_err
        error_theta_inc = np.sqrt(np.sum(w_q*((bestfit_theta_inc - mean_inc )**2
                                              + variance_theta_incq )))
        print('Inclination = %f +/- %f' % (mean_inc, error_theta_inc))
        marg_inc=mean_inc
        marg_inc_err = error_theta_inc
    else:
        marg_inc=inc[0]
        marg_inc_err=0

    if openar==True:
        mean_ar=np.sum(w_q*ar)
        bestfit_theta_ar=ar
        variance_theta_arq = ar_err*ar_err
        error_theta_ar = np.sqrt(np.sum(w_q*((bestfit_theta_ar - mean_ar)**2
                                              + variance_theta_arq )))
        print('a/R* = %f +/- %f' % (mean_ar, error_theta_ar))
        marg_ar=mean_ar
        marg_ar_err = error_theta_ar
    else:
        marg_ar=ar[0]
        marg_ar_err=0

    marg_c=limb_array[0,:]
    marg_c_err=np.zeros(4)
    # if transit==True:
    #     for i, c in enumerate(limb_array.T):
    #         mean_c=np.sum(w_q*c)
    #         var=limb_err_array[:,i]*limb_err_array[:,i]
    #         error_c=np.sqrt(np.sum(w_q*((c - mean_c)**2 + var )))
    #         marg_c[i]=mean_c
    #         marg_c_err[i]=error_c

    if plotting == True:
        plt.close()
        plt.clf()
        plt.errorbar(sys_lightcurve_x[bestfit,:], sys_lightcurve[bestfit,:], sys_lightcurve_err[bestfit,:]
                     ,marker='o', color='b', ecolor='b', ls='')
        plt.plot(sys_model_x[bestfit,:], sys_model[bestfit,:], ls='-')
        delta=sys_lightcurve_x[bestfit,1]-sys_lightcurve_x[bestfit,0]
        plt.xlim([sys_lightcurve_x[bestfit,0]-delta, sys_lightcurve_x[bestfit,-1]+delta])
        plt.title('HAT-P-41b WFC3 whitelight curve: Marginalization')
        plt.xlabel('Phase')
        plt.ylabel('Normalized Flux')
        # plt.ylim([.999,1.001])
        plt.show()
        #  plt.errorbar(sys_lightcurve_x[bestfit,:], sys_residuals[bestfit,:], sys_lightcurve_err[bestfit,:]
        #, marker='o', color='b', ls='',ecolor='blue')
        #  plt.plot(sys_lightcurve_x[bestfit,:], np.zeros_like(sys_lightcurve_x[1,:]))
        #  plt.show()

    rms=np.std(sys_residuals[bestfit,:])*1e6
    ratio=rms/phot_err
    print('Rms: %f' % rms)
    print('Photon error: %f' % phot_err)
    print('Ratio: %f' % ratio)

    ####### Auto-correlation of residuals ########

    best = a[0]
    syst = grid[best,:]
    #syst[11]=0
    p0 = sys_params[best, :]
    perr = sys_params_err[best, :]
    #p0[11] = 22.3
    #perr[11] = 1.65

    ac_resids = autocorr_func_1d(sys_residuals[best,:], norm=True)
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
    conf = scipy.stats.norm.ppf(1-sig/2.)/np.sqrt(len(sys_residuals[best,:]))
    plt.plot(lags, np.zeros_like(ac_resids)+conf, color='r', label='2 sigma range')
    plt.plot(lags, np.zeros_like(ac_resids)-conf, color = 'r')
    plt.title('Autocorrelation function of residuals')
    plt.legend()
    plt.show()


    ####### EMCEE ###########
    if mcmc == True:
        if save_mcmc == True:
            mc_dir = './emcee_runs/marg/' + save_name
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

        ###### Does this part still work?? Had trouble with error for nonlinear part of slopes
        perr[18]=.2
        perr[22]=2
        ndim, nwalkers = len(p0[syst==0]), int(len(p0[syst==0])*2.5/2)*2
        pos=np.array([p0[syst==0] + 5*perr[syst==0]*np.random.randn(ndim) for i in range(nwalkers)])
        #pos[:, sys==1] = p0[sys==1]
        print(syst)
        print(p0)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob
                                        , args=(x, y, err, p0, perr, syst
                                                , sh, HSTphase,dir_array, transit))

        nsteps = 10000
        for i, result in enumerate(sampler.sample(pos, iterations=nsteps)):
            if (i+1) % 100 == 0:
                print("{0:5.1%}".format(float(i) / nsteps))

        #ac.autocorr_func_1d(samples.chain[:,:,0], norm=True)
        print("Time elapsed in minutes %.2f" % ((time.time()-start_time)/60))
        plt.clf()
        plt.close()
        burn = 4000
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
            #plt.show()
        if save_mcmc == True:
            plt.savefig(mc_dir+'/acor_est.pdf')
            plt.close()
            plt.clf()
        else:
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
        if save_mcmc == True:
            pickle.dump(pickle_dict
                        , open( mc_dir +"/sampler.p", "wb" ) )

        samples = sampler.chain[:,burn:,:].reshape((-1, ndim))


        #plt.close()
        inds = np.random.randint(len(samples), size=100)
        pp = p0.copy()
        for ind in inds:
            samp = samples[ind]
            pp[syst==0] = samp
            phase = (x-pp[1])/Per
            phase -= np.floor(phase)
            phase[phase > 0.5] = phase[phase > 0.5] -1.0
            syste=get_sys_model(pp, x, sh, HSTphase, dir_array)
            mod = lightcurve(pp, x, sh, HSTphase, dir_array, transit=transit)
            plt.plot(x, mod, '.k', ls='', alpha=.1)
        #plt.ylim([.9,1.1])
        plt.errorbar(x, y, err, marker='o', color='b', ecolor='b', ls='')
        #plt.show()
        if save_mcmc == True:
            plt.savefig(mc_dir+'/model_fits.pdf')
            plt.close()
            plt.clf()
        else:
            plt.show()
        for i in range(ndim):
            plot_chain(sampler.chain, i, lab[syst==0][i],
                       save=save_mcmc, mc_dir=mc_dir)

        plt.close()
        plt.clf()
        samples[:,0] = samples[:, 0]*samples[:, 0]*1e6
        fig = corner.corner(samples, labels=lab[syst==0]
                            , quantiles=[.16,.5,.84], show_titles=True)
        if save_mcmc == True:
            plt.savefig(mc_dir+'/marg_corner.pdf')
            plt.close()
            plt.clf()
        else:
            plt.show()

        accept=sampler.acceptance_fraction
        print('accept rate: ', accept)
        #time1 = sampler.acor
        #times=sampler.get_autocorr_time()
        #print 'acor times: ', time1
        #print 'get_autocorr_times: ', times

        p_mcmc = [(v[1], v[2]-v[1], v[1]-v[0]) for v in zip(*np.percentile(samples, [16, 50, 84],
                                        axis=0))]

        mc_depth_err = (p_mcmc[0][1]+p_mcmc[0][2])/2.0
        mc_model_ratio = mc_depth_err/depth_err[best]/1e6*ratio
        mc_marg_ratio =  mc_depth_err/marg_depth_err/1e6*ratio
        cols = ['Median', '16th percentile', '84th percentile']
        mc_results = pd.DataFrame(p_mcmc, columns=cols)
        mc_results['Parameter'] = lab[syst==0]
        mc_results = mc_results.set_index('Parameter')
        mc_results['Model ratio'] = mc_model_ratio
        mc_results['Marg ratio'] = mc_marg_ratio
        mc_results['Marg inflation'] = ratio
        if save_mcmc == True:
            mc_results.to_csv(mc_dir+'/best_params.csv')

    if save_model_info == True:

        ### Two dataframes, both multi-indexed
        # First: Anything dependent on systematic models. Adjusted data, weights,
        # residuals, etc.
        # Second: All data things for plots. Flux/errors normalized and un. Dates,
        # And then all relevant fitting results (RMS/photon)

        # To retrieveas numpy array: df.loc[visit, type you want][column].values
        # Example: wl_models_info.loc['hatp41/visit01','Params']['Model 12'].values[0]

        cols = ['Model ' + str(i) for i in range(nsys)]
        subindex=['Weight'] + ['Corrected Flux']*nexposure + ['Corrected Phase']*nexposure \
            + ['Corrected Error']*nexposure + ['Residuals']*nexposure \
            + ['Params']*nParam + ['Params Errors']*nParam + ['AIC Evidence'] \
            + ['Smooth Model']*500 + ['Smooth Model Phase']*500
        ind=pd.MultiIndex.from_product([[save_name], subindex])
        wl=pd.DataFrame(np.vstack((w_q, sys_lightcurve.T, sys_lightcurve_x.T
                                  , sys_lightcurve_err.T, sys_residuals.T
                                  , sys_params.T, sys_params_err.T, sys_evidence.T
                                   , sys_model.T,sys_model_x.T)), columns=cols, index=ind)
        wl['Transit']=transit


        ind2a=pd.MultiIndex.from_product([[save_name],['data']*nexposure])
        colsa=['Obs Date', 'Normalized Flux', 'Flux', 'Normalized Error'
               , 'Error', 'Wavelength Shift']
        dataa=np.vstack((img_date, y, rawflux, error, rawerr, sh))
        colsb=['Values']
        datab=[marg_depth, marg_depth_err, marg_epoch, marg_epoch_err, marg_inc, marg_inc_err
               , marg_ar, marg_ar_err, rms, phot_err, ratio, orbit_start, orbit_end, scale]
        ind2b=pd.MultiIndex.from_product([[save_name],['Marg Depth', 'Depth err'
                                                    , 'Marg Epoch', 'Epoch err', 'Inc', 'Inc err'
                                                    , 'ar', 'ar err', 'RMS', 'photon err' , 'ratio'
                                                    , 'Norm index1', 'Norm index2', 'Error Scaling']])
        df1 = pd.DataFrame(dataa.T, columns=colsa, index=ind2a)
        df2 = pd.DataFrame(datab, columns=colsb, index=ind2b)
        wl_data = pd.concat((df1,df2))
        wl_data['Transit']=transit

        try:
            # NOTE: I increased model numbers and changed param amount, so I needed a new file here
            cur=pd.read_csv('./wl_models_info.csv', index_col=[0,1])
            cur=cur.drop(save_name, level=0, errors='ignore')
            cur=pd.concat((cur,wl), sort=False)

            cur.to_csv('./wl_models_info.csv', index_label=['Obs', 'Type'])
        except IOError:
            wl.to_csv('./wl_models_info.csv', index_label=['Obs', 'Type'])

        try:
            curr=pd.read_csv('./wl_data.csv', index_col=[0,1])
            curr=curr.drop(save_name, level=0, errors='ignore')
            curr=pd.concat((curr,wl_data))
            curr.to_csv('./wl_data.csv', index_label=['Obs', 'Type'])
        except IOError:
            wl_data.to_csv('./wl_data.csv',index_label=['Obs', 'Type'])

    return [marg_depth, marg_depth_err, marg_epoch
            , marg_epoch_err, marg_inc, marg_inc_err
            , marg_ar, marg_ar_err, marg_c, marg_c_err]




