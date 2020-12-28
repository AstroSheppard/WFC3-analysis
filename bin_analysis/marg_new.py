import sys

import matplotlib.pyplot as plt
from tqdm import trange
from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy.stats
import batman

# Emcee stuff
import scipy.optimize as op
import emcee
import corner
import pickle

# Local stuff
#import whitelight2018 as wl
from ..wl_preprocess import marg_mcmc as wl
from ..marg_mcmc.wl_preprocess import next_pow_two, auto_window, autocorr_func_1d, autocorr_new
from ..wl_preprocess.wave_solution import orbits
from kapteyn import kmpfit



def lightcurve(p, x, sh, HSTphase, dir_array, wlmodel, err, wlerror
               , transit=False):

    """ Function used by MPFIT to fit data to lightcurve model.

    Inputs: p: input parameters that we are fitting for
    x: Date of each observation, to be converted to phase
    err: Error on flux of each time series point
    sh: Wavelength shift on the detector (for each exposure)
    HSTphase: Phase of HST for each exposure
    wlmodel: The residuals from the whitelight fit for
    the same systematic model at each exposure
    wlerror: Whitelight flux error at each exposure
    dir_array: Binary array that notes if exposure is forward or reverse scan
    transit: True for transit, false for eclipse

    Output: Returns weighted deviations between model and data to be minimized
    by MPFIT. """

    # Old params
    # params= [rprs,flux0,tc,m,HSTP1,HSTP2,HSTP3,HSTP4,HSTP5,HSTP6,xshift1
    # ,xshift2,xshift3,xshift4,xshift5,xshift6,inclin,A_R,c1,c2,c3,c4,Per,fp]

    # New param order
    #p0 = [rprs,epoch,
    #      HSTP1,HSTP2,HSTP3,HSTP4,
    #      xshift1 ,xshift2 ,xshift3,xshift4,
    #      inclin,a_r,c1,c2,c3,c4,
    #      Per,fp,fnorm, flinear, fquad,
    #      fexpb, fexpc, flogb, flogc,
    #      rnorm, rlinear, rquad,
    #      rexpb, rexpc, rlogb, rlogc, wl_coeff]
    Per = p[16]

    phase = (x-p[1])/Per
    phase = phase - np.floor(phase)
    phase[phase > 0.5] = phase[phase > 0.5] -1.0

    systematic_model=wl.get_sys_model(p,phase,HSTphase,sh,dir_array)
    lcmodel=wl.get_lightcurve_model(p, x, transit=transit)
    model=lcmodel * systematic_model  + wlmodel*p[-1]

    norm_err = p[18]*(1-dir_array)*err + p[25]*dir_array*err
    #norm_err = err
    error=np.sqrt(norm_err**2 + wlerror**2*p[-1]*p[-1])
    return model, error

def residuals(p,data):

    x, y, err, sh, HSTphase, dir_array, wlmodel, wlerror, transit = data
    ym, error=lightcurve(p, x, sh, HSTphase, dir_array, wlmodel, err
                         , wlerror, transit=transit)
    return (y-ym)/error

def marg(p_start
         , img_date
         , bin_spectra
         , binerr
         , norm1
         , norm2
         , visit
         , binsize
         , beta
         , wlerror
         , dir_array
         , sh = 0
         , first = 0
         , include_residuals = True
         , include_wl_adjustment = True
         , include_error_inflation = True
         , transit = False
         , plotting = True
         , save = False
         , nbin = 'test'):


    # First = the cutoff point for the first orbit. 0 if orbit included.

    white='../wl_preprocess/'
    # Read in residuals [nExposure x nModels]
    resids_df=pd.read_csv(white+'wlresiduals.csv', index_col=0).loc[visit].iloc[:,1:]

    # Fillna was used when the earlier wl (50 models) was combined with new wl (125 models) and it kept
    # arrays same dimension. Dropna was used when both 50/50 and 125/125 were possible. I'll keep
    # for now, but might be unnecessary in future.
    #resids_df = resids_df.fillna(0)
    resids_df = resids_df.dropna(axis=1)

    ### Get numeric columns (only needed to do once, when columns were out of order)
    #cols = [int(x[-2:]) for x in resids_df.columns]
    #resids_df.columns = cols
    # Sort columns to appropriate order
    #resids_df = resids_df.sort_index(axis=1)
    ###

    # Now, finally get values
    WLresiduals = resids_df.values
    WLresiduals = WLresiduals[first:]

    ### code to correctly order file:
    #cur = pd.read_csv('wlresiduals_orig.csv', index_col=[0,1])
    #cols = [int(x[6:]) for x in cur.columns]
    #cur.columns = cols
    #cur = cur.sort_index(axis=1)
    #new_cols = ['Model ' + str(x) for x in cur.columns]
    #cur.columns = new_cols
    #cur.to_csv('./wlresiduals.csv', index_label=['Obs', 'Transit'])

    #sss

    nexposure = len(img_date)
    if len(sh) == 1:
        sh=wl.get_shift(bin_spectra)

    # SET THE CONSTANTS USING THE PRIORS
    rprs = p_start[0]
    epoch = p_start[1]
    inclin = p_start[2]
    a_r = p_start[3]
    Per = p_start[4]
    fp=p_start[5] # eclipse depth
    c1=p_start[6]
    c2=p_start[7]
    c3=p_start[8]
    c4=p_start[9]

    fnorm = 1.0     # Forward scan light curve normalization
    rnorm = 1.0     # Reverse norm, (for two direction scan)
    flinear = 0.0   # Linear Slope
    fquad = 0.0     # Quadratic slope
    fexpb = 0.0     # Exponential slope factor
    fexpc = 1.0     # Exponential slope phase
    flogb = 0.0     # Log slope factor
    flogc = 1.0     # Log slope phase
    rlinear = 0.0   # Reverse scan slope
    rquad = 0.0     # Quadratic slope
    rexpb = 0.0     # Exponential slope factor
    rexpc = 1.0     # Exponential slope phase
    rlogb = 0.0     # Log slope factor
    rlogc = 1.0     # Log slope phase
    xshift1 = 0.0   # X-shift in wavelength
    xshift2 = 0.0   # X-shift^2 in wavelength
    xshift3 = 0.0   # X-shift^3 in wavelength
    xshift4 = 0.0   # X-shift^4 in wavelength
    HSTP1 = 0.0     # HST orbital phase
    HSTP2 = 0.0     # HST orbital phase^2
    HSTP3 = 0.0     # HST orbital phase^3
    HSTP4 = 0.0     # HST orbital phase^4
    wl_coeff=0.0    # while light residual coefficient

    # PLACE ALL THE PRIORS IN AN ARRAY
    #p0 = [rprs,flux0,epoch,m,HSTP1,HSTP2,HSTP3,HSTP4,HSTP5,HSTP6,xshift1
    #      ,xshift2,xshift3,xshift4,xshift5,xshift6,inclin,a_r,c1,c2
    #      ,c3,c4,Per,fp,wl_coeff]
    p0 = [rprs,epoch,
          HSTP1,HSTP2,HSTP3,HSTP4,
          xshift1 ,xshift2 ,xshift3,xshift4,
          inclin,a_r,c1,c2,c3,c4,
          Per,fp,fnorm, flinear, fquad,
          fexpb, fexpc, flogb, flogc,
          rnorm, rlinear, rquad,
          rexpb, rexpc, rlogb, rlogc, wl_coeff]
    nParam=len(p0)

    lab = np.array(['Depth', 'Epoch', 'HST1', 'HST2'
                    , 'HST3', 'HST4', 'sh1','sh2'
                    , 'sh3', 'sh4', 'i', 'ars', 'c1'
                    , 'c2', 'c3', 'c4', 'Per', 'Eclipse Depth'
                    , 'fnorm', 'flinear', 'fquad', 'fexpb'
                    , 'fexpc', 'flogb', 'flogc', 'rnorm'
                    , 'rlinear', 'rquad', 'rexpb'
                    , 'rexpc', 'rlogb', 'rlogc', 'WL_coeff'])

    # SELECT THE SYSTEMATIC GRID OF MODELS TO USE
    # Indicate which visit-long slope parameterizations to include
    linear = True
    quad = False
    exp = False
    log = False
    grid = wl.systematic_model_grid_selection(4, dir_array=dir_array,
                                              transit=transit, linear=linear,
                                              quad=quad, exp=exp,
                                              log=log)

    # Toggle ones and zeros to turn WL_resids off or on
    if include_residuals == True:
        wl_coeff_factor = np.zeros(grid.shape[0])
    else:
        wl_coeff_factor = np.ones(grid.shape[0])
    grid=np.vstack((grid.T, wl_coeff_factor)).T
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
    sys_rchi2 = np.zeros((nsys))
    sys_rchi2_original_err = np.zeros((nsys))
    sys_chi2 = np.zeros((nsys))
    sys_dof = np.zeros((nsys))
    sys_nfree = np.zeros((nsys))
    sys_model_full=np.zeros((nsys,nexposure))
    WLeffect=np.zeros((nsys,nexposure))

    # Scatter of the residuals for each model
    resid_stddev = np.zeros(nsys)
    run1_AIC = np.zeros(nsys)
    run1_params = np.zeros((nsys,nParam))

    x = img_date
    y=bin_spectra.sum(axis=1)
    #y[1::2] = np.median(y[::2])/np.median(y[1::2])*y[1::2]
    err = np.sqrt(np.ma.sum(binerr*binerr, axis=1))
    #phot_err=1e6/np.median(np.sqrt(y))
    phot_err=1e6*np.median(err/y)

    # For transits, normaliza by an out of transit orbit.
    # For eclipses, normalize by an in-transit orbit
    orbit_start, orbit_end=orbits('holder', x=x, y=y, transit=transit)[1]
    norm=np.median(y[orbit_start:orbit_end])

    rawerr=err.copy()
    rawflux=y.copy()
    # Normalize the data to be near 1.0. The parameters fnorm and rnorm will fine-tune this in the fit.

    err = err/norm
    y = y/norm


    # Calculate HST phase
    HSTper = 96.36 / (24.*60.)
    # Get phase using same 0 point as the whitelight curve
    HSTphase = (img_date-img_date[first])/HSTper
    HSTphase = HSTphase - np.floor(HSTphase)
    # Change back to .5. Only necessary to avoid weird behavior with first orbit for l9859c
    HSTphase[HSTphase > 0.5] = HSTphase[HSTphase > 0.5] -1.0

    phase = (x-epoch)/Per
    phase = phase - np.floor(phase)
    phase[phase > 0.5] = phase[phase > 0.5] -1.0

    for s, systematics in tqdm(enumerate(grid), desc='First MPFIT run'):
        # Ensure that the center of transit/eclipse time is fixed for these spectral fits.
        systematics[1]=1
        # Get residuals from whitelight curve for corresponding model s
        WL_model=WLresiduals[:,s]

        if transit==False:
            systematics[0]=1
            systematics[17]=0

        # Tell fitter to fix every parameter with a 1, and set
        # limits on exp/log so no NaN's are encountered
        parinfo=[]
        for i in range(len(p0)):
            dic = {'fixed':systematics[i]}
            if lab[i]=='flogc' or lab[i]=='rlogc':
                dic = {'fixed':systematics[i], 'limits': [0.5,None]}
            if lab[i]=='fexpc' or lab[i]=='rexpc':
                dic = {'fixed':systematics[i], 'limits': [0.0001,None]}
            parinfo.append(dic)
        #parinfo=[]
        #for i in range(len(p0)):
        #    parinfo.append({'fixed':systematics[i]})
        fa=(x,y,err,sh,HSTphase,dir_array,WL_model,wlerror,transit)
        m2=kmpfit.Fitter(residuals=residuals, data=fa, parinfo=parinfo, params0=p0)
        m2.fit()
        params_w=m2.params
        model_err=np.sqrt(params_w[-1]**2*wlerror**2+err**2)
        #model_err = err
        AIC=(2*len(x)*np.log(np.median(model_err))+len(x)*np.log(2*np.pi)
            + m2.chi2_min + 2*m2.nfree)

        # if transit==True:
           # print 'Depth = ', np.square(params_w[0]), ' at ', params_w[1]
        # else:
           # print 'Depth = ', params_w[17], ' at ', params_w[1]


        # Since epoch and period are fixed, no need to recalculate phase.
        systematic_model=wl.get_sys_model(params_w, phase, HSTphase, sh, dir_array)
        lc_model=wl.get_lightcurve_model(params_w, x, transit=transit)
        w_model=lc_model*systematic_model + WL_model*params_w[-1]
        w_residuals = (y - w_model)

        resid_stddev[s] = np.std(w_residuals)
        run1_AIC[s] = AIC
        run1_params[s,:] = params_w

    #Determine which of the systematic models initially gives the best fit
    top = np.argmin(run1_AIC)


    # Scale error by resid_stddev[top]

    # Note: seems wrong to enhance data errors to be scale * model error, and then
    # later incorporate model error again. I'm double counting error, so maybe
    # that affects weights? At the same time, scale is usually less than 1
    # so might not matter. Also unclear if we should judge fit by original errors,
    # or by total error with scatter introduced (instead of just using that scatter
    # to inflate parameter errors.

    std=resid_stddev[top]
    model_err=np.sqrt(run1_params[top,-1]**2*wlerror**2+err**2)
    #model_err = err
    if np.median(model_err) < std and include_error_inflation == True:
        #sys.exit('something is being scaled')
        #scale = std / np.median(model_err)
        scale = np.sqrt(np.median((std*std - wlerror*wlerror*run1_params[top,-1]*run1_params[top,-1])/err/err))
        print('Scale %.2f' % scale)
        #print 'new test scale %.2f' % scale2
    else:
        scale = 1.0
    #error = scale * model_err
    # Trying to correct for above comment, though this is simple
    #scale = (std^2 - wl^2) / err^2
    scale = 1.0
    error = scale * err


    # print 'Beta: %.2f' % beta
    # print 'Scaling: %.2f' % (std/np.median(model_err))
    #print 'Beta check %.2f' % (beta)
    # Beta, which is a hack to account for red noise, is no longer really used
    # and always set to 1. However, I'll keep it in here in case it becomes
    # relevant again.

    for s, systematics in tqdm(enumerate(grid), desc='Final MPFIT run'):
        # Update priors, re-read white light residuals for corresponding model
        WL_model=WLresiduals[:,s]
        p0=run1_params[s,:]
        # Fix event time again
        systematics[1]=1
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
        #parinfo=[]
        #for i in range(len(p0)):
        #    parinfo.append({'fixed':systematics[i]})
        fa=(x,y,error,sh,HSTphase,dir_array,WL_model,wlerror,transit)
        m2=kmpfit.Fitter(residuals=residuals, data=fa, parinfo=parinfo, params0=p0)
        m2.fit()

        params=m2.params
        perror=m2.xerror
        nfree=m2.nfree
        dof = m2.dof
        #AIC=m2.rchi2_min + nfree

        # First scale error by normalization
        norm_error = error*(1-dir_array)*params[18] + error*dir_array*params[25]
        # Then account for error introduced by scaled wl residuals
        model_error=np.sqrt(params[-1]**2*wlerror**2+norm_error**2)
        #model_error = error
        AIC=(2*len(x)*np.log(np.median(model_error))+len(x)*np.log(2*np.pi)
             + m2.chi2_min + 2*nfree)
        stderror=m2.stderr

        ### what about xerr? I think I should use that. Of course MCMC answers
        # more clearly
        print('err comp')
        print(m2.stderr/m2.xerr)
        print('-----')
        # if transit==True:
        #     print 'Depth = ',np.square(params[0]), ' at ', params[2]
        # else:
        #     print 'Depth = ',params[17], ' at ', params[2]

        # EVIDENCE BASED ON AIC #
        sys_evidence[s]= -0.5*AIC
        sys_rchi2[s]= m2.rchi2_min
        sys_chi2[s]= m2.chi2_min
        sys_dof[s]= dof
        sys_nfree[s]= nfree



        #Calculate the systematic model given the fit parameters
        systematic_model = wl.get_sys_model(params, phase, HSTphase, sh, dir_array)
        lc_model=wl.get_lightcurve_model(params, x, transit=transit)
        model=lc_model*systematic_model + WL_model*params[-1]
        corrected = (y-WL_model*params[-1]) / systematic_model
        fit_residuals = (y - model)
        fit_err=model_error
        rchi2_orig = np.sum(fit_residuals**2/norm_error/norm_error)/dof

        #rchi2_orig = np.sum(fit_residuals*fit_residuals/fit_err/fit_err)/dof
        sys_rchi2_original_err[s] = rchi2_orig

        # Smooth Transit Model #
        time_smooth = (np.arange(500)*0.002-.5)*params[16]+params[1]
        phase_smooth=np.arange(500)*.002-.5
        smooth_model=wl.get_lightcurve_model(params, time_smooth, transit=transit)
        if plotting == True:
            if s == 90:
                plt.close()
                plt.clf()
            #plt.errorbar(img_date, y, error, color='red', marker='o', ecolor='red', ls='', label='Data')
            #plt.show()
            plt.plot(img_date, np.zeros_like(img_date))
            plt.errorbar(img_date, fit_residuals, fit_err, color='blue'
                     , marker='o', ls='', label='Model')
            plt.show()
            plt.plot(img_date, model)
            plt.errorbar(img_date, corrected, fit_err, marker='x', color='green'
                         , ecolor='green', ls='', label='Corrected Data')
            plt.legend()
            plt.show()

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
        WLeffect[s,:]=WL_model*params[-1]

    ###################
    ########
    #
    #MARGINALIZATION!!!
    #
    ########
    ###################
    aics = sys_evidence
    depth_array = sys_depth[:,0]
    depth_err_array = sys_depth[:,1]
    limb_array=sys_params[:,12:16]
    limb_err_array=sys_params_err[:,12:16]

    a=np.argsort(aics)[::-1]
    best=np.argmax(aics)
    #print best
    zero = np.where(aics < -300)
    if (len(zero) > 1): print('Some bad fits - evidence becomes negative')
    if (len(zero) > 24):
        sys.exit('Over half the systematic models have negative evidence, adjust and rerun')

    aics[aics < -300] = np.min(aics[aics>-300])
    # To prevent overflow. Also look into using logs instead of exp...
    if np.any(aics >750):
        beta = np.max(aics) - 700
    else:
        beta=100.

    w_q = (np.exp(aics-beta))/np.sum(np.exp(aics-beta))
    if np.any(~np.isfinite(w_q)):
        print("weight is infinite, check beta")
        sss
    bestfit=np.argmax(w_q)
    n01 = np.where(w_q >= 0.1)
    stdResid = np.std(sys_residuals[bestfit,:])
    # print 'Evidences: ', aics
    # print 'Weights: ', w_q
    # print str(len(n01[0])) + ' models have a weight over 10%. Models: ', n01[0] , w_q[n01]
    # print 'Most likely model is number ' +str(bestfit) +' at weight = ' + str(np.max(w_q))

    depth = depth_array
    depth_err = depth_err_array
    coeffs=sys_params[:,-1]

    ### HERE ADJUST DEPTHS
    if include_wl_adjustment == False:
        pass
    else:
        # Read in marginalized white light depth
        wl_results=pd.read_csv(white+'wl_data.csv', index_col=[0,1]).loc[visit]
        margWL=wl_results.loc['Marg Depth', 'Values']
        margerrWL=wl_results.loc['Depth err', 'Values']
        # Read in white light depths for each model
        mods=pd.read_csv(white+'wl_models_info.csv',
                         index_col=[0,1]).loc[visit].dropna(axis=1)
        if transit==True:
            modeldepths=np.square(mods.loc['Params'].values[0,:-1])
        else:
            modeldepths=mods.loc['Params'].values[-1,:-1]

        #modeldeptherr=readdeptherr(inp)
        #hold = margWL-modeldepths.copy()
        #modeldepths[:] = modeldepths[-1]

        # Difference between marginalized wl depth and each models' depth

        deld = margWL-modeldepths

        # The below  was only necessary when combined old wl (50 models) with new bin fits (125)
        # deld=np.append(deld, np.zeros(75))
        print("--------------------------------------")
        # print 'Average model bias: %.1f' % (deld.mean()*1e6)
        print('Average white light residual coeff: %.2f' %  coeffs.mean())
        print("--------------------------------------")

        #delerr=np.sqrt(margerrWL**2+modeldeptherr**2)
        #coeffs=np.ones(50)
        #deld=-1.0*deld
        #coeffs=np.zeros(50)
        #coeffs=np.zeros(50)
        #print deld[bestfit]*1e6

        # Adjust each model's depth due to its scaled bias
        depth = depth_array + deld*coeffs
        # Adjust each model's error due to incorporating marg
        # white light depth. (This should also account for model
        # white light depth, but then errors are massive, if I
        # remember correctly.

        depth_err = np.sqrt(depth_err_array**2+margerrWL**2)

    print('----------------------------------------------------------------')
    print('Minimum reduced chi-squared: %.2f' % sys_rchi2[bestfit])
    print()
    print('Minimum reduced chi-squared without residual adjustment: %.2f' % sys_rchi2_original_err[bestfit])

    print("DOF: %.1f" % sys_dof[bestfit])
    print('----------------------------------------------------------------')
    mean_depth=np.sum(w_q*depth)
    theta_qi=depth
    variance_theta_qi=depth_err*depth_err
    error_theta_i = np.sqrt(np.sum(w_q*((theta_qi - mean_depth)**2 + variance_theta_qi )))
    print('Depth = %f  +/-  %f' % (mean_depth*1e6, error_theta_i*1e6))
    marg_depth = mean_depth
    marg_depth_err = error_theta_i

    # Gives marginalized depth and error if adjustment didn't exist, as a test.
    # Unless explicitly interested, can ignore
    mm=np.sum(w_q*depth_array)
    theta_qii=depth_array
    variance_theta_qii=depth_err_array*depth_err_array
    error_theta_ii = np.sqrt(np.sum(w_q*((theta_qii - mm)**2 + variance_theta_qii )))
    marg_unadj = mm
    marg_unadj_err = error_theta_ii

    """plt.scatter(deld*1e6, (depth_array)*1e6, s=w_q*1000, color='r', label='Unadjusted')
    plt.scatter(deld*1e6, (depth)*1e6, s=w_q*1000, color='b', label='Adjusted')
    plt.errorbar([0], [marg_depth*1e6], [marg_depth_err*1e6], color='k', marker='x', label='Adjusted Marginalized')
    plt.errorbar([10], [marg_unadj*1e6], [marg_unadj_err*1e6], color='brown', marker='x', label='Unadjusted Marginalized')
    plt.legend()
    plt.show()"""

    #print limb_array
    marg_c=limb_array[0,:]
    marg_c_err=np.zeros(4)
    # if transit==True:
    #     for i, c in enumerate(limb_array.T):
    #         mean_c=np.sum(w_q*c)
    #         var=limb_err_array[:,i]*limb_err_array[:,i]
    #         error_c=np.sqrt(np.sum(w_q*((c - mean_c)**2 + var )))
    #         marg_c[i]=mean_c
    #         marg_c_err[i]=error_c
    # print 'c: ', marg_c

    print('WL coeff %.2f' % coeffs[bestfit])
    #print np.median(sys_lightcurve_err[bestfit,:])
    if plotting == True:

        plt.errorbar(sys_lightcurve_x[bestfit,:], sys_lightcurve[bestfit,:]
                     , sys_lightcurve_err[bestfit,:]
                     ,marker='o', color='b', ecolor='b', ls='')
        #plt.errorbar(sys_lightcurve_x[bestfit,:], y, error
        #             ,marker='o', color='b', ecolor='b', ls='')
        #plt.plot(sys_lightcurve_x[bestfit,:],y-WLeffect[bestfit,:], 'ro')
        plt.plot(sys_model_x[bestfit,:], sys_model[bestfit,:], ls='-')
        delta=sys_lightcurve_x[bestfit,1]-sys_lightcurve_x[bestfit,0]
        plt.xlim([sys_lightcurve_x[bestfit,0]-delta, sys_lightcurve_x[bestfit,-1]+delta])
        plt.title('HAT-P-41b WFC3 whitelight curve: Marginalization')
        plt.xlabel('Phase')
        plt.ylabel('Normalized Flux')
        # plt.ylim([.999,1.001])
        plt.show()

    rms=np.std(sys_residuals[bestfit,:])*1e6
    ratio=rms/phot_err
    print('RMS/theoretical error: %.2f' % ratio)
    weight=w_q
    marg_error=marg_depth_err

    # save model number, weight, depth, and adjustment to array with bin number index
    cols = ['Model', 'Evidence', 'red_chi^2', 'dof', 'nfree', 'chi2', 'Resids sum', 'Model error', 'Weight', 'Depth']
    www = pd.DataFrame(np.vstack((np.arange(1,126), sys_evidence, sys_rchi2, sys_dof, sys_nfree, sys_chi2,
                                  np.sum(sys_residuals*sys_residuals, axis=1)*1e6,
                                  np.median(sys_lightcurve_err, axis=1)*1e6, w_q*100., depth_array*1e6)).T, columns=cols)
    www['Visit'] = visit
    www['Bin'] = nbin
    www=www.set_index(['Visit', 'Bin'])
    #print www
    """
    try:
        curr=pd.read_csv('./bin_w.csv', index_col=[0,1])
        curr=curr.drop((visit, int(nbin)), errors='ignore')
        curr=pd.concat((curr, www))
        curr.to_csv('./bin_w.csv', index_label=['Visit', 'Bin'])
    except IOError:
        www.to_csv('./bin_w.csv',index_label=['Visit', 'Bin'])"""



    best = a[0]
    syst = grid[best,:]
    #syst[11]=0

    #p0 = sys_params[best, :]
    #perr = sys_params_err[best, :]
    #p0[11] = 22.3
    #perr[11] = 1.65

    if plotting == True:
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

    if save == True:

        phase = sys_lightcurve_x[bestfit,:]
        model = sys_model_full[bestfit,:]
        corrected=sys_lightcurve[bestfit,:]
        fit_err = sys_lightcurve_err[bestfit,:]
        fit_residuals = sys_residuals[bestfit,:]

        phase_smooth = sys_model_x[bestfit,:]
        smooth_model = sys_model[bestfit,:]
        params = sys_params[bestfit,:]
        stderror = sys_params_err[bestfit,:]

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
        cols=['Phase', 'Model']
        data=np.vstack((phase_smooth, smooth_model)).T
        bin_smooth=pd.DataFrame(data, columns=cols)
        bin_smooth['Visit']=visit
        bin_smooth['binsize']=binsize
        bin_smooth['bin']=nbin
        bin_smooth=bin_smooth.set_index(['Visit','binsize', 'bin'])
        bin_smooth['Transit']=transit

        # Save results
        ## Kyle changes

        cols = ['Depth', 'RMS', 'Photon Error', 'Ratio', 'Norm index1'
                , 'Norm index2', 'Best model', 'rprs', 'Epoch'
                , 'HST1', 'HST2', 'HST3', 'HST4'
                , 'sh1','sh2', 'sh3', 'sh4'
                , 'i', 'ars', 'c1', 'c2', 'c3', 'c4'
                , 'Per', 'Eclipse Depth'
                , 'fnorm', 'flinear', 'fquad', 'fexpb'
                , 'fexpc', 'flogb', 'flogc', 'rnorm'
                , 'rlinear', 'rquad', 'rexpb'
                , 'rexpc', 'rlogb', 'rlogc', 'WL coeff']

        data=[marg_depth, rms, phot_err, ratio, orbit_start, orbit_end, bestfit] + params.tolist()
        errors= [marg_depth_err, 0, 0, 0, 0, 0, 0] + stderror.tolist()
        ind2=pd.MultiIndex.from_product([[visit],[binsize],[nbin],['Values', 'Errors']])
        bin_params = pd.DataFrame(np.vstack((data,errors)), columns=cols, index=ind2)
        bin_params['Transit']=transit

        try:
            cur=pd.read_csv('./bin_params.csv', index_col=[0,1,2,3])
            #cur=cur.drop((visit, nbin), errors='ignore')
            cur=pd.concat((cur,bin_params), sort=False)
            cur=cur[~cur.index.duplicated(keep='first')]
            cur.to_csv('./bin_params.csv', index_label=['Obs', 'Bin Size', 'Bin', 'Type'])
        except IOError:
            bin_params.to_csv('./bin_params.csv', index_label=['Obs', 'Bin Size', 'Bin', 'Type'])

        try:
            curr=pd.read_csv('./bin_data.csv', index_col=[0,1,2])
            curr=curr.drop((visit,binsize, int(nbin)), errors='ignore')
            curr=pd.concat((curr,bins), sort=False)
            curr.to_csv('./bin_data.csv', index_label=['Obs', 'Bin Size', 'Bin'])
        except IOError:
            bins.to_csv('./bin_data.csv',index_label=['Obs', 'Bin Size', 'Bin'])

        try:
            currr=pd.read_csv('./bin_smooth.csv', index_col=[0,1,2])
            currr=currr.drop((visit, binsize,int(nbin)), errors='ignore')
            currr=pd.concat((currr,bin_smooth), sort=False)
            currr.to_csv('./bin_smooth.csv', index_label=['Obs', 'Bin Size', 'Bin'])
        except IOError:
            bin_smooth.to_csv('./bin_smooth.csv', index_label=['Obs', 'Bin Size', 'Bin'])


    return marg_depth, marg_error, rms
