import sys
import configparser

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob as glob
from astropy.io import fits

import marg_mcmc as wl
from wave_solution import orbits

def wlresids(visit
             , transit=True
             , include_error_inflation=True
             , ld_type='nonlinear'
             , ignore_first_exposures=False
             , openar = False
             , one_slope=True):

    data_index = visit
    if include_error_inflation == False:
        # Inflated or not, whitelight data and preprocess info are the same.
        visit = visit + '_no_inflation'
    if ld_type == 'linear':
        visit = visit + '_linearLD'
    if ignore_first_exposures == True:
        visit = visit + '_no_first_exps'
    if openar == True:
        visit = visit + '_openar'

    data_dir = './data_outputs/'
    pre = pd.read_csv(data_dir+'preprocess_info.csv', index_col=[0,1]).loc[(data_index, ignore_first_exposures)]
    #pre = pre.loc[pre.['Ignore first exposures'].values[0] == ignore_first_exposures]
    white = pd.read_csv(data_dir+'wl_data.csv', index_col=[0,1]).loc[visit] # previously data_index 3/22

    first = pre['User Inputs'].values[-2].astype(int)
    last = pre['User Inputs'].values[-1].astype(int)
    norm = white.loc['Flux Norm Value', 'Values']
    HSTmidpoint = white.loc['HST midpoint', 'Values'].astype(int)

    # READ IN ALL PROCESSED DATA
    proc = data_dir + 'processed_data.csv'
    df = pd.read_csv(proc, index_col=[0,1]).loc[data_index]
    transit = df['Transit'].values[0]
    spec = df.loc['Value'].drop(['Date', 'sh'
                                 , 'Mask', 'Transit'
                                 , 'Scan Direction']
                                , axis=1).dropna(axis=1).values
    specerr = df.loc['Error'].drop(['Date', 'sh'
                                    , 'Mask', 'Transit'
                                    , 'Scan Direction']
                                   , axis=1).dropna(axis=1).values
    date = df.loc['Value','Date'].values
    dir_array = df.loc['Value','Scan Direction'].values
    mask = df.loc['Value', 'Mask'].values
    if ignore_first_exposures == False:
        mask = np.ones_like(mask)


    # The two files (processed data with exposures and without) are identical except for mask (As expected, though
    # sh not changing is nice). But, when mixed (mask included with no-mask residual extraction, and possibly
    # vice versa), the model is a little off and there's a ramp. Probably HST phase -origin related. How to
    # not overwrite mask??
    
    sh = df.loc['Value', 'sh'].values

    HST_phase_ref = date[mask][first+HSTmidpoint]
    nexposure = len(date)
    flux = np.sum(spec, axis=1)
    err = np.sqrt(np.sum(specerr*specerr, axis=1))

    testbin=spec[:,50:60]
    testbinerr=specerr[:,50:60]
    binflux=np.sum(testbin, axis=1)
    binerr=np.sqrt(np.sum(testbinerr*testbinerr, axis=1))
    
    orbit1, orbit2 = orbits('holder', x=date[mask][first:], y=flux[mask][first:], transit=transit)[1]
    norm3 = np.median(flux[mask][first:][orbit1:orbit2])
    orbit_start, orbit_end = orbits('holder', x=date[mask][first:], y=binflux[mask][first:], transit=transit)[1]
    binnorm=np.median(binflux[mask][first:][orbit_start:orbit_end])

    fluxnorm = flux / norm
    errnorm = err / norm
    binflux = binflux / binnorm
    binerr = binerr / binnorm

    # READ IN SYSTEMATIC MODEL PARAMETERS for all models FROM WHITELIGHT FIT

    models_df = pd.read_csv(data_dir+'wl_models_info.csv'
                            , index_col=[0,1]).loc[visit]
    params = models_df.loc['Params', :'Model 124'].values.T
    model_tested = models_df.loc['Model Tested?', :'Model 124'].values
    nModels = params.shape[0]
    
    # CALCULATE HST PHASE AT EACH TIME
    HSTper = 96.36 / (24.*60.)
    HSTphase = (date - HST_phase_ref) / HSTper
    HSTphase = HSTphase - np.floor(HSTphase)
    # HST cutoff, sometimes 0.6 is necessary to avoid odd model predictions for first orbit.
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
        if model_tested[s] == True:
            model = wl.lightcurve(par, date, sh, HSTphase, dir_array
                                  , transit=transit, ld_type=ld_type
                                  , one_slope=one_slope)
            resids = fluxnorm - model
            sys_residuals[s,:] = resids
            per = par[16]
            t = par[1]
            # SAVE RESIDUALS
            if s == 49:
                plt.clf()
                plt.plot(date, 1-resids, 'ro',label='Model - Data')
                plt.plot(date, fluxnorm, 'bs', label='Normalized Flux')
                plt.plot(date, model, marker='x', ls='', label='Model')
                plt.legend(numpoints=1)
                plt.show()
                plt.clf()
                plt.close('all')
        
            # phase = (date-t) / per
            # phase = phase - np.floor(phase)
            # phase[phase > 0.5] = phase[phase > 0.5] -1.0

    cols = ['Model ' + str(i) for i in range(nModels)]
    wlresids = pd.DataFrame(sys_residuals.T, columns=cols)
    wlresids['Visit'] = visit
    wlresids['Transit'] = transit
    wlresids['Scan Direction'] = dir_array
    wlresids = wlresids.set_index(['Visit', 'Transit'])

    try:
        cur = pd.read_csv('./data_outputs/wlresiduals.csv', index_col=[0,1])
        cur = cur.drop((visit, transit), errors='ignore')
        cur = pd.concat((cur, wlresids), sort=False)
        cur.to_csv('./data_outputs/wlresiduals.csv', index_label=['Obs', 'Transit'])
    except IOError:
        wlresids.to_csv('./data_outputs/wlresiduals.csv', index_label=['Obs', 'Transit'])
        # Do I need to save expos, sh, hstphase, date, flux, fluxnorm, err, errnorm, phase?
        #   SAVE, filename=savename, sys_residuals, expos, sh, hstphase, date
        # saved stuff below for wlcurve too
        #  SAVE, filename='./paper/wlerror/'+planet+'/'+file+'.sav', flux, fluxnorm
        #, err, errnorm, date, phase, xc, cc, start, finish, xb


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.py')
    planet = config.get('DATA', 'planet')
    visit_number = config.get('DATA', 'visit_number')
    direction = config.get('DATA', 'scan_direction')
    transit = config.getboolean('DATA', 'transit')

    include_error_inflation = config.getboolean('MODEL', 'include_error_inflation')
    ignore_first_exposures = config.getboolean('DATA', 'ignore_first_exposures')
    ld_type = config.get('MODEL', 'limb_type')
    openar = config.getboolean('MODEL', 'openar')
    one_slope = config.getboolean('MODEL', 'one_slope')
    
    visit = planet + '/' + visit_number + '/' + direction

    resids = wlresids(visit
                      , transit=transit
                      , include_error_inflation=include_error_inflation
                      , ld_type=ld_type
                      , ignore_first_exposures=ignore_first_exposures
                      , openar=openar
                      , one_slope=one_slope )
