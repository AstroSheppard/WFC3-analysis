from __future__ import print_function
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from tqdm import tqdm

import whitelight2018 as wl
from wave_solution import orbits
from kapteyn import kmpfit
import batman

def lightcurve(p, x, sh, HSTphase, wlmodel, err, wlerror, transit=False):
    
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
    phase[phase > 0.5] = phase[phase > 0.5] -1.0

    systematic_model=wl.get_sys_model(p,phase,HSTphase,sh)
    lcmodel=wl.get_lightcurve_model(p, x, transit=transit)
    model=lcmodel * p[1] * systematic_model  + wlmodel*p[24]
    error=np.sqrt(err**2 + wlerror**2*p[24]*p[24])
    return model, error

def residuals(p,data):
    x, y, err, sh, HSTphase, wlmodel, wlerror, transit = data
    ym, error=lightcurve(p, x, sh, HSTphase, wlmodel, err
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
         , sh = 0
         , transit = False
         , plotting = True
         , save = False
         , nbin = 'test'):
    white='../wl_preprocess/'
    # Read in residuals [nExposure x nModels]
    WLresiduals=pd.read_csv(white+'wlresiduals.csv', index_col=0).loc[visit].iloc[:,1:].values
    nexposure = len(img_date)
    if len(sh) == 1:
        sh=wl.get_shift(bin_spectra)

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
    flux0 = 1.0

    m = 0.0 # Linear Slope
    xshift1 = 0.0 # X-shift in wavelength
    xshift2 = 0.0 # X-shift in wavelength
    xshift3 = 0.0 # X-shift in wavelength
    xshift4 = 0.0 # X-shift in wavelength
    xshift5 = 0.0 # X-shift in wavelength
    xshift6 = 0.0 # X-shift in wavelength
    HSTP1 = 0.0 # Correct HST orbital phase
    HSTP2 = 0.0 # Correct HST orbital phase^2
    HSTP3 = 0.0 # Correct HST orbital phase^3
    HSTP4 = 0.0 # Correct HST orbital phase^4
    HSTP5 = 0.0 # Correct HST orbital phase^5
    HSTP6 = 0.0 # Correct HST orbital phase^5
    wl_coeff=0.0   # while light residual coefficient...try making this 1 to start
    # PLACE ALL THE PRIORS IN AN ARRAY
    p0 = [rprs,flux0,epoch,m,HSTP1,HSTP2,HSTP3,HSTP4,HSTP5,HSTP6,xshift1
          ,xshift2,xshift3,xshift4,xshift5,xshift6,inclin,a_r,c1,c2
          ,c3,c4,Per,fp,wl_coeff]
    nParam=len(p0)
    
    # SELECT THE SYSTEMATIC GRID OF MODELS TO USE 
    grid = wl.systematic_model_grid_selection(4, transit)
    # Add extra 0 for open wl_coeff fitting
    grid=np.vstack((grid.T,np.zeros(grid.shape[0]))).T # toggle ones and zeros to turn WL_resids off or on
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
    y[1::2] = np.median(y[::2])/np.median(y[1::2])*y[1::2]
    err = np.sqrt(np.ma.sum(binerr*binerr, axis=1))
    #phot_err=1e6/np.median(np.sqrt(y))
    phot_err=1e6*np.median(err/y)

    orbit_start, orbit_end=orbits('holder', x=x, y=y, transit=transit)[1]
    norm=np.median(y[orbit_start:orbit_end])
    
    rawerr=err.copy()
    rawflux=y.copy()
    err = err/norm
    y = y/norm
  

    # Calculate HST phase
    HSTper = 96.36 / (24.*60.)
    HSTphase = (img_date-img_date[norm1])/HSTper
    HSTphase = HSTphase - np.floor(HSTphase)
    # Change back to .5. Only necessary to avoid weird behoavior with first orbit for l9859c
    HSTphase[HSTphase > 0.65] = HSTphase[HSTphase > 0.65] -1.0

    phase = (x-epoch)/Per 
    phase = phase - np.floor(phase)
    phase[phase > 0.5] = phase[phase > 0.5] -1.0

    for s, systematics in tqdm(enumerate(grid), desc='First MPFIT run'):
        
        system=systematics
        system[2]=1
        WL_model=WLresiduals[:,s]
       
        if transit==False:
            system[0]=1
            system[23]=0
        parinfo=[]
        for i in range(len(p0)):
            parinfo.append({'fixed':system[i]})
        fa=(x,y,err,sh,HSTphase,WL_model,wlerror,transit)
        m2=kmpfit.Fitter(residuals=residuals, data=fa, parinfo=parinfo, params0=p0)
        m2.fit()
        params_w=m2.params
        model_err=np.sqrt(params_w[24]**2*wlerror**2+err**2)
        AIC=(2*len(x)*np.log(np.median(model_err))+len(x)*np.log(2*np.pi)
            + m2.chi2_min + 2*m2.nfree)

        #if transit==True:
           # print 'Depth = ', np.square(params_w[0]), ' at ', params_w[2]
        #else:
        #    print 'Depth = ', params_w[23], ' at ', params_w[2]

        systematic_model=wl.get_sys_model(params_w, phase, HSTphase, sh)
        lc_model=wl.get_lightcurve_model(params_w, x, transit=transit)
        w_model=params_w[1]*lc_model*systematic_model + WL_model*params_w[24]
        w_residuals = (y - w_model)/params_w[1]

        resid_stddev[s] = np.std(w_residuals)
        run1_AIC[s] = AIC
        run1_params[s,:] = params_w

    #Determine which of the systematic models initially gives the best fit
    top = np.argmin(run1_AIC)

    
    # Scale error by resid_stddev[top]
    std=resid_stddev[top]
    model_err=np.sqrt(run1_params[top,24]**2*wlerror**2+err**2)
    if np.median(model_err) < std:
        scale = std / np.median(model_err)
    else:
        scale = 1.0
    error = scale * err
    #print 'Beta: %.2f' % beta
    print('Scaling: %.2f' % (std/np.median(model_err)))
    error*=beta
    for s, systematics in tqdm(enumerate(grid), desc='Final MPFIT run'):
        # Update priors
        WL_model=WLresiduals[:,s] # this didn't exist before? or did I accidentally undo for hatp41?
        p0=run1_params[s,:]
        system[2]=1
        if transit==False:
            systematics[0]=1
            systematics[23]=0
        parinfo=[]
        for i in range(len(p0)):
            parinfo.append({'fixed':systematics[i]})
        fa=(x,y,error,sh,HSTphase,WL_model,wlerror,transit)
        m2=kmpfit.Fitter(residuals=residuals, data=fa, parinfo=parinfo, params0=p0)
        m2.fit()
        
        params=m2.params
        perror=m2.xerror
        nfree=m2.nfree
        dof = m2.dof
        #AIC=m2.rchi2_min + nfree
        model_error=np.sqrt(params[24]**2*wlerror**2+error**2)
        AIC=(2*len(x)*np.log(np.median(model_error))+len(x)*np.log(2*np.pi)
             + m2.chi2_min + 2*nfree)
        stderror=m2.stderr
        
        # if transit==True:
        #     print 'Depth = ',np.square(params[0]), ' at ', params[2]
        # else:
        #     print 'Depth = ',params[23], ' at ', params[2]
       
        # EVIDENCE BASED ON AIC #
        sys_evidence[s]= -0.5*AIC
        sys_rchi2[s]= m2.rchi2_min
        sys_chi2[s]= m2.chi2_min
        sys_dof[s]= dof
        sys_nfree[s]= nfree

        #Calculate the systematic model given the fit parameters
        systematic_model = wl.get_sys_model(params, phase, HSTphase, sh)
        lc_model=wl.get_lightcurve_model(params, x, transit=transit)
        model=params[1]*lc_model*systematic_model + WL_model*params[24]
        corrected = (y-WL_model*params[24]) / (params[1] * systematic_model)
        fit_residuals = (y - model)/params[1]
        fit_err=model_error/params[1]

        # Smooth Transit Model #
        time_smooth = (np.arange(500)*0.002-.5)*params[22]+params[2]
        phase_smooth=np.arange(500)*.002-.5
        smooth_model=wl.get_lightcurve_model(params, time_smooth, transit=transit)
        if plotting == True:
            if s == 90:
                #plt.close()
                #plt.clf()
                plt.errorbar(img_date, y, error, color='red', marker='o', ecolor='red', ls='', label='Data')
                plt.show()
            #plt.plot(img_date, systematic_model+ WL_model*params[24], color='blue'
            #         , marker='o', ls='', label='Model')
            #plt.errorbar(img_date, corrected, fit_err, marker='x', color='green'
            #             , ecolor='green', ls='', label='Corrected Data')
            #plt.legend()
            #plt.show()
            
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
        WLeffect[s,:]=WL_model*params[24]

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
    limb_array=sys_params[:,18:22]
    limb_err_array=sys_params_err[:,18:22]
    
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

    ### HERE ADJUST DEPTHS
    
    wl_results=pd.read_csv(white+'wl_data.csv', index_col=[0,1]).loc[visit]
    margWL=wl_results.loc['Marg Depth', 'Values']
    margerrWL=wl_results.loc['Depth err', 'Values']
    # read in model depths
    mods=pd.read_csv(white+'wl_models_info.csv',index_col=[0,1]).loc[visit]
    if transit==True:
        modeldepths=np.square(mods.loc['Params'].values[0,:-1])
    else:
        modeldepths=mods.loc['Params'].values[-1,:-1]
    
    #modeldeptherr=readdeptherr(inp)
    deld = margWL-modeldepths
    coeffs=sys_params[:,24]
    #delerr=np.sqrt(margerrWL**2+modeldeptherr**2)
    depth = depth_array + deld*coeffs
    depth_err = depth_err_array
    #print deld[bestfit]*1e6
    
    print()
    print()
    print('Minimum red chi squared', np.min(sys_rchi2))
    print()
    mean_depth=np.sum(w_q*depth)
    theta_qi=depth
    variance_theta_qi=depth_err*depth_err
    error_theta_i = np.sqrt(np.sum(w_q*((theta_qi - mean_depth)**2 + variance_theta_qi )))
    print('Depth = %f  +/-  %f' % (mean_depth*1e6, error_theta_i*1e6))
    marg_depth = mean_depth
    marg_depth_err = error_theta_i
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
    print()
    print(coeffs[bestfit])
    print(np.median(sys_lightcurve_err[bestfit,:]))
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
    print(ratio)
    weight=w_q
    marg_error=marg_depth_err


    
    # save model number, weight, depth, and adjustment to array with bin number index
    cols = ['Model', 'Evidence', 'red_chi^2', 'dof', 'nfree', 'chi2', 'Resids sum', 'Model error', 'Weight']
    www = pd.DataFrame(np.vstack((np.arange(1,51), sys_evidence, sys_rchi2, sys_dof, sys_nfree, sys_chi2,
                                  np.sum(sys_residuals*sys_residuals, axis=1)*1e6,
                                  np.median(sys_lightcurve_err, axis=1)*1e6, w_q*100.)).T, columns=cols)
    www['Visit'] = visit
    www['Bin'] = nbin
    www=www.set_index(['Visit', 'Bin'])
    print(www)
    try:
        curr=pd.read_csv('./bin_w.csv', index_col=[0,1])
        curr=curr.drop((visit, int(nbin)), errors='ignore')
        curr=pd.concat((curr, www))
        curr.to_csv('./bin_w.csv', index_label=['Visit', 'Bin'])
    except IOError:
        www.to_csv('./bin_w.csv',index_label=['Visit', 'Bin'])
        
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
        cols=['Depth', 'RMS', 'Photon Error', 'Ratio', 'Norm index1', 'Norm index2', 'rprs'
              , 'Zero-flux' , 'Event time', 'Slope', 'hst1', 'hst2', 'hst3', 'hst4', 'hst5', 'hst6'
              , 'xs1', 'xs2', 'xs3', 'xs4', 'xs5', 'xs6', 'inc','ar', 'c1', 'c2', 'c3', 'c4'
              , 'Period', 'eclipse depth', 'WL Coeff']
        data=[marg_depth, rms,phot_err, ratio, orbit_start, orbit_end] + params.tolist()
        errors= [marg_depth_err, 0, 0, 0, 0, 0] + stderror.tolist()
        ind2=pd.MultiIndex.from_product([[visit],[binsize],[nbin],['Values', 'Errors']])
        bin_params = pd.DataFrame(np.vstack((data,errors)), columns=cols, index=ind2)
        bin_params['Transit']=transit
    
        try:
            cur=pd.read_csv('./bin_params.csv', index_col=[0,1,2, 3])
            #cur=cur.drop((visit, nbin), errors='ignore')
            cur=pd.concat((cur,bin_params))
            cur=cur[~cur.index.duplicated(keep='first')]
            cur.to_csv('./bin_params.csv', index_label=['Obs', 'Bin Size', 'Bin', 'Type'])
        except IOError:
            bin_params.to_csv('./bin_params.csv', index_label=['Obs', 'Bin Size', 'Bin', 'Type'])
        
        try:
            curr=pd.read_csv('./bin_data.csv', index_col=[0,1,2])
            curr=curr.drop((visit,binsize, int(nbin)), errors='ignore')
            curr=pd.concat((curr,bins))
            curr.to_csv('./bin_data.csv', index_label=['Obs', 'Bin Size', 'Bin'])
        except IOError:
            bins.to_csv('./bin_data.csv',index_label=['Obs', 'Bin Size', 'Bin'])

        try:
            currr=pd.read_csv('./bin_smooth.csv', index_col=[0,1,2])
            currr=currr.drop((visit, binsize,int(nbin)), errors='ignore')
            currr=pd.concat((currr,bin_smooth))
            currr.to_csv('./bin_smooth.csv', index_label=['Obs', 'Bin Size', 'Bin'])
        except IOError:
            bin_smooth.to_csv('./bin_smooth.csv', index_label=['Obs', 'Bin Size', 'Bin'])


    return marg_depth, marg_error, rms
