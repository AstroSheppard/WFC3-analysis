import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from tqdm import tqdm
import pandas as pd

from wave_solution import orbits
from kapteyn import kmpfit
import batman

def get_sys_model(p, phase, HSTphase, sh):
    systematic_model = ((phase*p[3] + 1.0) 
                        * (HSTphase*p[4] + HSTphase**2.*p[5] + HSTphase**3.*p[6]
                           + HSTphase**4.*p[7] + HSTphase**5.*p[8] + HSTphase**6.*p[9] + 1.0)
                        * (sh*p[10] + sh**2.*p[11] + sh**3.*p[12] + sh**4.*p[13]
                           + sh**5.*p[14] + sh**6.*p[15] + 1.0))
    return systematic_model

def get_lightcurve_model(p, date, limb="nonlinear", transit=False):
    
    params=batman.TransitParams()
    params.w=90.
    params.ecc=0
    params.rp=p[0]
    tc=p[2]
    params.inc=p[16]
    params.a=p[17]
    params.per=p[22]
    depth=p[23]

    if transit==True:
        params.t0=tc
        if limb=="nonlinear":
            params.u=p[18:22]
            params.limb_dark="nonlinear"
        if limb=="quadratic":
            params.u=p[18:20]
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

def lightcurve(p, x, sh, HSTphase, transit=False):
    
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
    
    # params= [rprs,flux0,tc,m,HSTP1,HSTP2,HSTP3,HSTP4,HSTP5,HSTP6,xshift1
    # ,xshift2,xshift3,xshift4,xshift5,xshift6,inclin,A_R,c1,c2,c3,c4,Per,fp]
    Per = p[22]              

    phase = (x-p[2])/Per
    phase = phase - np.floor(phase)
    phase[phase > 0.5] = phase[phase > 0.5] - 1.0
    #sss
    
    systematic_model=get_sys_model(p,phase,HSTphase,sh)
    lcmodel=get_lightcurve_model(p, x, transit=transit)
    model=lcmodel * p[1] * systematic_model

    return model

def residuals(p,data):
    x, y, err, sh, HSTphase,  transit = data
    ym=lightcurve(p, x, sh, HSTphase, transit=transit)
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

def systematic_model_grid_selection(size=4, transit=False):
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
        grid = [[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
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
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

    if grid == 6:
        # MODEL GRID UPTO THE 6th ORDER for HST & DELTA_lambda
        grid = [[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

    grid=np.asarray(grid)
    #if transit == True:
    #    grid[:,18:22]=0
        
    return grid

  
def whitelight2018(p_start, img_date, allspec, allerr, plotting=False
                   , fixtime=False, norandomt=False, openinc=False, openar=False
                   , savewl=False, transit=False):
    """
  NAME:                        
       WHITELIGHT2018.py
 
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

 savewl - True to save results

 plotting - set as True or False to see the plots

 FIXTIME - True to keep center of eclipse/transit time fixed
     
 NORANDOMT - True to not allow small random changes to center time

 OPENINC - True to fit for inclination

 TRANSIT - True for transit light curves, default false for eclipse 
 
    """

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
    flux0 = allspec[0,:].sum()
    
    m = 0.0         # Linear Slope
    xshift1 = 0.0   # X-shift in wavelength
    xshift2 = 0.0   # X-shift^2 in wavelength
    xshift3 = 0.0   # X-shift^3 in wavelength
    xshift4 = 0.0   # X-shift^4 in wavelength
    xshift5 = 0.0   # X-shift^5 in wavelength
    xshift6 = 0.0   # X-shift^6 in wavelength
    HSTP1 = 0.0     # HST orbital phase
    HSTP2 = 0.0     # HST orbital phase^2
    HSTP3 = 0.0     # HST orbital phase^3
    HSTP4 = 0.0     # HST orbital phase^4
    HSTP5 = 0.0     # HST orbital phase^5
    HSTP6 = 0.0     # HST orbital phase^5

    #PLACE ALL THE PRIORS IN AN ARRAY
    p0 = [rprs,flux0,epoch,m,HSTP1,HSTP2,HSTP3,HSTP4,HSTP5,HSTP6,xshift1
          ,xshift2 ,xshift3,xshift4,xshift5,xshift6,inclin,a_r,c1,c2,c3,c4,Per,fp]
      
    nParam=len(p0)
    # SELECT THE SYSTEMATIC GRID OF MODELS TO USE 

    grid = systematic_model_grid_selection(4, transit)
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


    print '----------      ------------     ------------'
    print '          1ST FIT         '
    print '----------      ------------     ------------'

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
    flux0=1.0 
  
    for s, systematics in tqdm(enumerate(grid), desc='First MPFIT run'):
        system=systematics
        if fixtime==False and norandomt==False:
            for n in range(ntrials+1):
                epoch = tcenter[n]

                # Reset priors
                p0 = [rprs,flux0,epoch,m,HSTP1,HSTP2,HSTP3,HSTP4,HSTP5,HSTP6,xshift1
                      ,xshift2,xshift3,xshift4,xshift5,xshift6,inclin,a_r,c1,c2,c3,c4,Per,fp]
                if openinc==True: system[16] = 0
                if openar==True: system[17] = 0
                if transit==False:
                    system[0]=1
                    system[23]=0
                parinfo=[]
                for i in range(len(p0)):
                    parinfo.append({'fixed':system[i]})
                #fa = {'x':x, 'y':y, 'err':err, 'sh':sh, 'HSTphase':HSTphase, 'transit':transit}
                fa=(x,y,err,sh,HSTphase,transit)
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
                    depth_test[n]=params_test[23]

            # Find the epoch time with the lowest AIC. Use it (or the average of 
            # values if multiple) to get starting depth and epoch time for
            # next round of fits.
            best = np.argmin(AIC_test)
            print best
            print 'Best eclipse depth prior =', depth_test[best]
            print 'Best center of eclipse prior =', tcenter[best]
            depth = np.mean(depth_test[best])
            epoch = np.median(tcenter[best])

            if transit==True:
                rprs=np.sqrt(depth)
            else:
                fp=depth


        #Re-run the fitting process with the best prior as defined by the iterative fit
        p0 = [rprs,flux0,epoch,m,HSTP1,HSTP2,HSTP3,HSTP4,HSTP5,HSTP6,xshift1
              ,xshift2,xshift3,xshift4,xshift5,xshift6,inclin,a_r,c1,c2,c3,c4,Per,fp]

        # MPFIT ;;;;;;;;;;;;;;;;;;;;;;;
       
        if fixtime==True: system[2] = 1 
        if openinc==True: system[16] = 0
        if openar==True: system[17] = 0
        if transit==False:
            system[0]=1
            system[23]=0
        parinfo=[]
        for i in range(len(p0)):
            # parinfo.append({'value':p0[i], 'fixed':system[i]
            #                 , 'limited':[0,0], 'limits':[0.,0.]})
            parinfo.append({'fixed':system[i]})
        #fa = {'x':x, 'y':y, 'err':err, 'sh':sh, 'HSTphase':HSTphase, 'transit':transit}
        fa=(x,y,err,sh,HSTphase,transit)
        m2=kmpfit.Fitter(residuals=residuals, data=fa, parinfo=parinfo, params0=p0)
        m2.fit()
        #m = mpfit.mpfit(lightcurve,functkw=fa,parinfo=parinfo, fastnorm=True)
        params_w=m2.params

        AIC=(2*len(x)*np.log(np.median(err))+len(x)*np.log(2*np.pi)
             + m2.chi2_min + 2*m2.nfree)
        if transit==True:
            print 'Depth = ', np.square(params_w[0]), ' at ', params_w[2]
        else:
            print 'Depth = ', params_w[23], ' at ', params_w[2]

        # Re-Calculate each of the arrays dependent on the output parameters
        phase = (x-params_w[2])/params_w[22] 
        phase -= np.floor(phase)
        phase[phase > 0.5] = phase[phase > 0.5] -1.0

        # LIGHT CURVE MODEL: calculate the eclipse model for the resolution of the data points
        # this routine is from MANDEL & AGOL (2002)
            
        systematic_model=get_sys_model(params_w, phase, HSTphase, sh)
        lc_model=get_lightcurve_model(params_w, x, transit=transit)
        w_model=params_w[1]*lc_model*systematic_model  
        w_residuals = (y - w_model)/params_w[1]

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
        print scale
        #print 'Not inflating by this'
    else:
        scale=1
    #scale=1
    error=err*scale
    
    
    print '----------      ------------     ------------'
    print '         FINAL FIT        '
    print '----------      ------------     ------------'
    for s, systematics in tqdm(enumerate(grid), desc='Final MPFIT run'):
  
        # Define the new priors as the parameters from the best fitting
        # systematic model
        p0=run1_params[s,:]
        if fixtime==True: systematics[2] = 1 
        if openinc==True: systematics[16] = 0
        if openar==True: systematics[17] = 0
        if transit==False:
            systematics[0]=1
            systematics[23]=0  
        parinfo=[]
        
        for i in range(len(p0)):
            parinfo.append({'fixed':systematics[i]})

        fa=(x,y,error,sh,HSTphase,transit)
        m2=kmpfit.Fitter(residuals=residuals, data=fa, parinfo=parinfo, params0=p0)
        m2.fit()
        params=m2.params
        perror=m2.xerror
        nfree=m2.nfree
        #AIC=m2.rchi2_min + nfree
        AIC=(2*len(x)*np.log(np.median(error))+len(x)*np.log(2*np.pi)
             + m2.chi2_min + 2*nfree)
        stderror=m2.stderr
        # Stderror = xerror * red chi squared, basically, where x error is an asymptotic error
        #stderror=m2.xerror
        print
        print 'Model ', s
        print 'reduced chi^2:', m2.rchi2_min
        print 'dof', m2.dof
        print

        if transit==True:
            print 'Depth = ',np.square(params[0]), ' at ', params[2]
        else:
            print 'Depth = ',params[23], ' at ', params[2]

        # Re-Calculate each of the arrays dependent on the output parameters
        phase = (x-params[2])/params[22]
        phase -= np.floor(phase)
        phase[phase > 0.5] = phase[phase > 0.5] -1.0

        # --------------------- #
        #        EVIDENCE       #
        #sigma_points = np.median(error)
        #Npoint = len(x) 
        # EVIDENCE BASED ON AIC ;
        #sys_evidence[s] = -Npoint*np.log(sigma_points)-0.5*Npoint*np.log(2*np.pi)-0.5*(AIC+nfree)
        sys_evidence[s]=-.5*AIC
        systematic_model=get_sys_model(params, phase, HSTphase, sh)
        lc_model=get_lightcurve_model(params, x, transit=transit)
        model=params[1]*lc_model*systematic_model
        corrected = y / (params[1] * systematic_model)   
        fit_residuals = (y - model)/params[1]
        fit_err = error/params[1]


        # Smooth Transit Model: change this from phase to time
        time_smooth = (np.arange(500)*0.002-.5)*params[22]+params[2]
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
            sys_depth[s,0] = params[23]
            sys_depth[s,1] = stderror[23]
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
    epoch_array = sys_params[:,2]       
    epoch_err_array = sys_params_err[:,2] 
    inc_array= sys_params[:,16]
    inc_err_array=sys_params_err[:,16]
    ar_array= sys_params[:,17]
    ar_err_array=sys_params_err[:,17]
    limb_array=sys_params[:,18:22]
    limb_err_array=sys_params_err[:,18:22]

    # Reverse sort as trying to MAXIMISE the negative log evidence
    a=np.argsort(aics)[::-1] 
    best=np.argmax(aics)
    print best
    # print aics

    zero = np.where(aics < -300)
    if (len(zero) > 1): print 'Some bad fits - evidence becomes negative'
    if (len(zero) > 24):
        sys.exit('Over half the systematic models have negative evidence, adjust and rerun')

    aics[aics < -300] = np.min(aics[aics>-300])

    beta=100.
    #beta = np.min(aics)
  
    w_q = (np.exp(aics-beta))/np.sum(np.exp(aics-beta))
    bestfit=np.argmax(w_q)
    n01 = np.where(w_q >= 0.1)
    
    stdResid = np.std(sys_residuals[bestfit,:]) 
    print 'Evidences: ', aics
    print 'Weights: ', w_q

    print str(len(n01[0])) + ' models have a weight over 10%. Models: ', n01[0] , w_q[n01]
    print 'Most likely model is number ' +str(bestfit) +' at weight = ' + str(np.max(w_q))

    depth = depth_array
    depth_err = depth_err_array

    # Marganilze depth formula 15 and 16 from Wakeford 2016
    mean_depth=np.sum(w_q*depth)
    theta_qi=depth
    variance_theta_qi=depth_err*depth_err
    error_theta_i = np.sqrt(np.sum(w_q*((theta_qi - mean_depth)**2 + variance_theta_qi )))
    print 'Depth = %f  +/-  %f' % (mean_depth, error_theta_i)
    marg_depth = mean_depth
    marg_depth_err = error_theta_i 
    
    # Marganilze tcenter
    t0 = epoch_array
    t0_err = epoch_err_array
    inc=inc_array
    inc_err=inc_err_array
    ar=ar_array
    ar_err=ar_err_array
    
    print 'Depth'
    print depth[a]
    print depth_err_array[a]
    print 'Center Time'
    print t0[a]
    print t0_err[a]
    mean_t0=np.sum(w_q*t0)
    theta_t0=t0
    variance_theta_t0q = t0_err*t0_err
    error_theta_t0 = np.sqrt(np.sum(w_q*((theta_t0 - mean_t0)**2 + variance_theta_t0q )))
    print 'Central time = %f +/- %f' % (mean_t0, error_theta_t0) 
    marg_epoch = mean_t0
    marg_epoch_err = error_theta_t0

    # Marginalize over inclination
    if openinc==True:
        mean_inc=np.sum(w_q*inc)
        bestfit_theta_inc=inc
        variance_theta_incq = inc_err*inc_err
        error_theta_inc = np.sqrt(np.sum(w_q*((bestfit_theta_inc - mean_inc )**2
                                              + variance_theta_incq )))
        print 'Inclination = %f +/- %f' % (mean_inc, error_theta_inc)
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
        print 'a/R* = %f +/- %f' % (mean_ar, error_theta_ar)
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
        plt.show(block=False)
        #  plt.errorbar(sys_lightcurve_x[bestfit,:], sys_residuals[bestfit,:], sys_lightcurve_err[bestfit,:]
        #, marker='o', color='b', ls='',ecolor='blue')
        #  plt.plot(sys_lightcurve_x[bestfit,:], np.zeros_like(sys_lightcurve_x[1,:]))
        #  plt.show()

    rms=np.std(sys_residuals[bestfit,:])*1e6
    ratio=rms/phot_err
    print 'Rms: %f' % rms
    print 'Photon error: %f' % phot_err
    print 'Ratio: %f' % ratio


    if savewl:
        ################# make sure this works
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
        ind=pd.MultiIndex.from_product([[savewl], subindex])
        wl=pd.DataFrame(np.vstack((w_q, sys_lightcurve.T, sys_lightcurve_x.T
                                  , sys_lightcurve_err.T, sys_residuals.T
                                  , sys_params.T, sys_params_err.T, sys_evidence.T
                                   , sys_model.T,sys_model_x.T)), columns=cols, index=ind)
        wl['Transit']=transit

        ind2a=pd.MultiIndex.from_product([[savewl],['data']*nexposure])
        colsa=['Obs Date', 'Normalized Flux', 'Flux', 'Normalized Error'
              , 'Error', 'Wavelength Shift']
        dataa=np.vstack((img_date, y, rawflux, error, rawerr, sh))
        colsb=['Values']
        datab=[marg_depth, marg_depth_err, marg_epoch, marg_epoch_err, marg_inc, marg_inc_err
               , marg_ar, marg_ar_err, rms, phot_err, ratio, orbit_start, orbit_end, scale]
        ind2b=pd.MultiIndex.from_product([[savewl],['Marg Depth', 'Depth err'
                                                    , 'Marg Epoch', 'Epoch err', 'Inc', 'Inc err'
                                                    , 'ar', 'ar err', 'RMS', 'photon err' , 'ratio'
                                                    , 'Norm index1', 'Norm index2', 'Error Scaling']])
        df1 = pd.DataFrame(dataa.T, columns=colsa, index=ind2a)
        df2 = pd.DataFrame(datab, columns=colsb, index=ind2b)
        wl_data = pd.concat((df1,df2))
        wl_data['Transit']=transit
    
        try:
            cur=pd.read_csv('./wl_models_info.csv', index_col=[0,1])
            cur=cur.drop(savewl, level=0, errors='ignore')
            cur=pd.concat((cur,wl), sort=False)
            cur.to_csv('./wl_models_info.csv', index_label=['Obs', 'Type'])
        except IOError:
            wl.to_csv('./wl_models_info.csv', index_label=['Obs', 'Type'])
            
        try:
            curr=pd.read_csv('./wl_data.csv', index_col=[0,1])
            curr=curr.drop(savewl, level=0, errors='ignore')
            curr=pd.concat((curr,wl_data))
            curr.to_csv('./wl_data.csv', index_label=['Obs', 'Type'])
        except IOError:
            wl_data.to_csv('./wl_data.csv',index_label=['Obs', 'Type'])

    return [marg_depth, marg_depth_err, marg_epoch
            , marg_epoch_err, marg_inc, marg_inc_err
            , marg_ar, marg_ar_err, marg_c, marg_c_err]




