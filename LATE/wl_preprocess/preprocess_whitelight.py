import configparser
import os
import sys
import time

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

import marg_mcmc as wl
import batman
# Set path to read in get_limb.py from bin_analysis
sys.path.insert(0, '../bin_analysis')
import get_limb as gl

def event_time(date, properties):
    """Program to determine the expected event time
     Inputs
     date: 1D array of the date of each exposure (MJD)
     properties: 1D array containing the last observed eclipse
     and the period. (MJD, days)"""
    time = properties[1, 0]
    time_error = properties[1, 1]
    period = properties[4, 0]
    period_error = properties[4, 1]
    i = 0
    while time < date[0]:
        i += 1
        time += period
    epoch_error = np.sqrt(time_error**2 + (i*period_error)**2)
    return float(time), float(epoch_error)

def get_orbits(date):
    """Procedure to organize light curve data by HST orbit"""
    orbit=np.zeros(1).astype(int)

    for i in range(len(date)-1):
        t=date[i+1]-date[i]
        if t*86400 > 1200.:
            orbit=np.append(orbit, i+1) # 1800s is about half an HST orbit
    return np.append(orbit, len(date))

def inputs(data, transit=True):

    """ Function to read in priors for a system.
    INPUTS:
    data: Data table of priors for a particular planet
    OUTPUTS:
    Returns array of system properties with corresponding uncertainties: 
    [rprs, central event time, inclination, orbit distance/stellar radius (a/Rs),
    period, transit/eclipse depth]
    """
    inp_values = pd.read_table(data, sep=' ', index_col=None)
    data_arr = inp_values.iloc[:,2].values
    labels = inp_values.iloc[:,0].values
    param_errs = inp_values.iloc[:,3].values

    # Conversions are R_Jup to meters, R_solar to meters, AU to meters,
    # and JD to MJD.
    conversions = np.array([6.9911e7, 6.957e8, 1.49598e11, -2400000.5])
    period = data_arr[4]
    inc = data_arr[5]
    rprs = data_arr[8]
    rprs_err = param_errs[8]
    a_R = data_arr[9]

    if transit==True:
        epoch = data_arr[6] + conversions[3]
        depth = rprs * rprs
        depth_err = 2 * rprs * rprs_err
    else:
        epoch = data_arr[6] + conversions[3] + period/2.
        depth = rprs*rprs*(data_arr[2]/data_arr[3])/3

    # Save important inputs to properties (props) array.
    props = np.zeros(6)
    props[0] = rprs
    props[1] = epoch
    props[2] = inc
    props[3] = a_R
    props[4] = period
    props[5] = depth

    # Save corresponding errors, mostly for priors for MCMC fitting.
    errors = np.zeros(6)
    errors[0] = rprs_err
    errors[1] = param_errs[6]
    errors[2] = param_errs[5]
    errors[3] = param_errs[9]
    errors[4] = param_errs[4]
    errors[5] = depth_err

    return [props,errors]

# def correction(inputs, date, flux, transit=False):

#     params=batman.TransitParams()
#     params.w=90.
#     params.ecc=0
#     params.rp=np.sqrt(inputs[0])
#     t0=inputs[1]
#     params.inc=inputs[2]
#     params.a=inputs[3]
#     params.per=inputs[4]
#     depth=inputs[5]

#     phase = (date-t0)/params.per
#     phase = phase - np.floor(phase)
#     phase[phase > 0.5] = phase[phase > 0.5] - 1.0

#     if transit==True:
#         params.t0=t0
#         params.u=inputs[6:]
#         params.limb_dark="quadratic"
#         m=batman.TransitModel(params, date)
#         model=m.light_curve(params)
#     else:
#         params.fp=inputs[5]
#         params.t_secondary=t0
#         params.u=[]
#         params.limb_dark="uniform"
#         m=batman.TransitModel(params, date, transittype="secondary")
#         model=m.light_curve(params)

#     corrected=flux/model
#     return corrected


# def remove_bad_data(light_curve, spectra, light_corrected, date1, light_err, spec_err
#                     , user_inputs, check=False):
#     """Procedure to remove "bad" data from light curve"""

#     med= np.ma.median(light_corrected)
#     sigma = np.sqrt(np.sum((light_corrected-med)**2)/(2*len(light_corrected)))
#     medi=np.zeros_like(date1)+med

#     sig3=medi+3*sigma
#     sig4=medi+4*sigma
#     sig5=medi+5*sigma
#     sig3m=medi-3*sigma
#     sig4m=medi-4*sigma
#     sig5m=medi-5*sigma
#     if check==False:
#         nPasses=int(user_inputs[3])
#         sigma_cut_factor=user_inputs[2]

#     else:
#         data=plt.plot(date1, light_corrected,'bo',ls='dotted')
#         plt.xlabel('MJD')
#         plt.ylabel('Total Flux')
#         s5=plt.plot(date1, sig5,'pink',date1, sig5m, 'pink')
#         s5[0].set_label('5-sigma')
#         s4=plt.plot(date1, sig4,'g', date1, sig4m, 'g')
#         s4[0].set_label('4-sigma')
#         s3=plt.plot(date1, sig3,'r', date1, sig3m, 'r')
#         s3[0].set_label('3-sigma')
#         plt.plot(date1, medi, label='Median',ls='solid')
#         plt.legend(scatterpoints=1)
#         plt.show(block=False)
#         cut = raw_input("Enter the sigma-cut factor (3-5 recommended): ")
#         sigma_cut_factor = float(cut)
#         user_inputs[2]=sigma_cut_factor
#         passes=raw_input("Enter the number of passes for the sigma-cut: ")
#         nPasses=int(passes)
#         user_inputs[3]=nPasses
#         plt.close()

#     # Cut out the "bad" data

#     for j in range(nPasses):
#         med= np.ma.median(light_corrected)
#         sigma = np.sqrt(np.sum((light_corrected-med)**2)/(2*len(light_corrected)))
#         dif= np.abs(light_corrected-med)
#         index=np.where(dif < sigma_cut_factor*sigma)[0]
#         light_curve=light_curve[index]
#         date1=date1[index]
#         light_corrected=light_corrected[index]
#         light_err=light_err[index]
#         spectra=spectra[index,:]
#         spec_err=spec_err[index,:]

#     return [light_curve, spectra, light_corrected, date1, light_err, spec_err]


def preprocess_whitelight(visit
                          , direction
                          , x=0, y=0
                          , check=True
                          , ignore_first_exposures=False
                          , inp_file=False
                          , ld_source='claret2011.csv'
                          , save_processed_data=False
                          , transit=False
                          , data_plots=True
                          , mcmc=False
                          , include_error_inflation=True
                          , openinc=False
                          , openar=True
                          , fixtime=False
                          , ld_type="nonlinear"
                          , linear_slope=True
                          , quad_slope=False
                          , exp_slope=False
                          , log_slope=False
                          , one_slope=True
                          , norandomt=True
                          , fit_plots=True
                          , save_mcmc=False
                          , save_model_info=False):

    """
    Function to allow user to extract relevant orbital data from reduced time
    series of a visit. Also allow user to exclude first orbit or first exposure
    of each orbit. The selected data is then fed into "marg_mcmc" for model light 
    curve fitting.

    INPUTS

    See config.py file

    [DATA]
    x, y: Allow the user to reduce aperture by (x,y) pixels
    checks: set to "on" to manually reduce data
    inp_file: Allow user to load in preprocess information instead of manually
    finding. Cannot have checks and inp_file both off or both on.

    If checks is set to on, "user_inputs" will return the inputs
    that the user used: [first orbit, last orbit, sigma cut factor,
    number of passes, center eclipse time]. If checks is set to off, then
    the user_inputs array will be used as inputs (easier to automate) 

    [MODEL]
    mcmc: Use MCMC sampler, extracting corner plot and other diagnostics
    openinc: Fit for inclination (default is fixed)
    openar: Fit for a/Rstar (default is fixed)
    fixtime: Fix center of event time (default is open)
    norandomt: Do now allow center of event time starting point to be vary randomly
    fit_plots: Show model light curve fit in real time

    [SAVE]
    save_processed_data: Save the data with the systematics removed for the best fit model
    save_model_info: Save best fit parameters for every systematic model
    save_mcmc: Save MCMC products, such as corner plot, autocorrelation, etc.

    The save files will all be saved with key or name  "planet/visitXX/direction"
    """

    if direction != 'both':

        folder = '../data_reduction/reduced/%s/%s/final/*.fits' % (visit, direction)
        data=np.sort(np.asarray(glob.glob(folder)))
        nexposure = len(data)
        print('There are %d exposures in this visit' % nexposure)
        alldate=np.zeros(len(data))
        time=np.zeros_like(alldate)
        test=fits.open(data[0])
        xlen, ylen = test[0].data.shape
        test.close()
        xlen-=2*x
        ylen-=2*y
        allspec=np.ma.zeros((len(data),xlen, ylen))
        allerr=np.zeros((len(data),xlen,ylen))
        xmin=x
        xmax=xlen-x
        ymin=y
        ymax=ylen-y

        for i, img in enumerate(data):
            expfile=fits.open(img)
            hdr=expfile[0].header
            exp=expfile[0].data
            mask=expfile[1].data
            errs=expfile[2].data
            expfile.close()
            alldate[i]=(hdr['EXPSTART']+hdr['EXPEND'])/2.
            time[i]=hdr['EXPTIME']
            expo=exp[xmin:xmax, ymin:ymax]
            mask=mask[xmin:xmax, ymin:ymax]
            errs=errs[xmin:xmax, ymin:ymax]
            allspec[i,:,:]=np.ma.array(expo, mask=mask)
            allerr[i,:,:]=np.ma.array(errs, mask=mask)

        allspec1d=np.ma.sum(allspec,axis=1)
        allerr1d=np.sqrt(np.ma.sum(allerr*allerr, axis=1))
        median_flux = np.ma.median(np.ma.sum(allspec1d, axis=1))
        # Regardless of direction, if all exposures share the same one we make
        # dir_array all zeros for easy parameter use in model fitting.
        dir_array = np.zeros_like(alldate)

    else:
        direction = 'forward'
        folder = '../data_reduction/reduced/%s/%s/final/*.fits' % (visit, direction)
        data=np.sort(np.asarray(glob.glob(folder)))
        nexposure = len(data)
        print('There are %d exposures in this visit' % nexposure)

        alldate=np.zeros(len(data))
        time=np.zeros_like(alldate)
        test=fits.open(data[0])
        xlen, ylen = test[0].data.shape
        test.close()
        xlen-=2*x
        ylen-=2*y
        allspec=np.ma.zeros((len(data),xlen, ylen))
        allerr=np.zeros((len(data),xlen,ylen))

        xmin=x
        xmax=xlen-x
        ymin=y
        ymax=ylen-y
    
        for i, img in enumerate(data):
            expfile=fits.open(img)
            hdr=expfile[0].header
            exp=expfile[0].data
            mask=expfile[1].data
            errs=expfile[2].data
            expfile.close()
            alldate[i]=(hdr['EXPSTART']+hdr['EXPEND'])/2.
            time[i]=hdr['EXPTIME']
            expo=exp[xmin:xmax, ymin:ymax]
            mask=mask[xmin:xmax, ymin:ymax]
            errs=errs[xmin:xmax, ymin:ymax]
            allspec[i,:,:]=np.ma.array(expo, mask=mask)
            allerr[i,:,:]=np.ma.array(errs, mask=mask)

        allspec1d=np.ma.sum(allspec,axis=1)
        allerr1d=np.sqrt(np.ma.sum(allerr*allerr, axis=1))
        median_flux = np.ma.median(np.ma.sum(allspec1d, axis=1))
        # Now do for other direction

        direction = 'reverse'
        folder = '../data_reduction/reduced/%s/%s/final/*.fits' % (visit, direction)
        rdata=np.sort(np.asarray(glob.glob(folder)))
        nexposure = len(rdata)
        print('There are %d exposures in this visit' % nexposure)

        rdate=np.zeros(len(rdata))
        rtime=np.zeros_like(rdate)

        rtest=fits.open(rdata[0])
        rxlen,rylen = rtest[0].data.shape
        test.close()
        xlen-=2*x
        ylen-=2*y
        rallspec=np.ma.zeros((len(rdata),rxlen, rylen))
        rallerr=np.zeros((len(rdata),rxlen,rylen))

        rxmin=x
        rxmax=rxlen-x
        rymin=y
        rymax=rylen-y

        for i, img in enumerate(rdata):
            expfile=fits.open(img)
            hdr=expfile[0].header
            exp=expfile[0].data
            mask=expfile[1].data
            errs=expfile[2].data
            expfile.close()
            rdate[i]=(hdr['EXPSTART']+hdr['EXPEND'])/2.
            rtime[i]=hdr['EXPTIME']
            expo=exp[rxmin:rxmax, rymin:rymax]
            mask=mask[rxmin:rxmax, rymin:rymax]
            errs=errs[rxmin:rxmax, rymin:rymax]
            rallspec[i,:,:]=np.ma.array(expo, mask=mask)
            rallerr[i,:,:]=np.ma.array(errs, mask=mask)

        rallspec1d=np.ma.sum(rallspec,axis=1)
        rallerr1d=np.sqrt(np.ma.sum(rallerr*rallerr, axis=1))
        rmedian_flux = np.ma.median(np.ma.sum(rallspec1d, axis=1))

        dir_factor = median_flux / rmedian_flux
        #dir_factor=1

        #rallspec1d = rallspec1d * dir_factor
        #rallerr1d = rallerr1d * dir_factor

        # Define array that has 0s for forward scan and 1s for reverse
        dir_array = np.append(np.zeros_like(alldate), np.ones_like(rdate))
        alldate = np.append(alldate,rdate)
        allspec1d = np.ma.append(allspec1d, rallspec1d, axis=0)
        allerr1d = np.ma.append(allerr1d, rallerr1d, axis=0)
        direction = 'both'


    # Put in correct time order
    date_order = np.argsort(alldate)
    dir_save = dir_array[date_order].copy()
    date_sav = alldate[date_order].copy()
    spec1d_sav = allspec1d[date_order,:].copy()
    err1d_sav = allerr1d[date_order,:].copy()
    light_mask = np.ones(len(dir_array), dtype=bool)

    # Classify the data by each HST orbit. Returns array (orbit)
    # which contains the indeces for the start of each orbit


    orbit = get_orbits(date_sav)
    firsts = orbit[:-1]
    # Ignore the first 3 points of the first orbit to
    # account for slower ramp. 
    laters = orbit[1]+np.arange(3)
    # laters = firsts+2
    firsts = np.append(firsts, firsts+1)
    firsts = np.append(firsts, laters)
    # Quick test on autocorrelation if the few outliers in
    # the last orbit are removed (in addition to first points).
    # firsts = np.append(firsts, [89, 90, 91])
    light_mask[firsts] = False
    if ignore_first_exposures == True:
        dir_array = dir_save[light_mask]
        alldate = date_sav[light_mask]
        allspec1d = spec1d_sav[light_mask]
        allerr1d = err1d_sav[light_mask]
    else:
        dir_array = dir_save.copy()
        alldate = date_sav.copy()
        allspec1d = spec1d_sav.copy()
        allerr1d = err1d_sav.copy()

    mask2= np.ones(len(dir_array), dtype=bool)
    # Quick test on autocorrelation if the few outliers in
    # the last orbit are removed (and only them)
    # Note: this did improve autocorrelation, and increased depth by about 15ppm.
    # This is intuitive, since the out of transit baseline is raised. Same for above.
    # mask2[[89, 90, 91]]=False
    # dir_array = dir_save[mask2]
    # alldate = date_sav[mask2]
    # allspec1d = spec1d_sav[mask2]
    # allerr1d = err1d_sav[mask2]
    
    orbit = get_orbits(alldate)
    planet = visit[:-8]
    props, errs = inputs('../planets/%s/inputs.dat' % planet, transit)
    if ld_type=='nonlinear':
        a1 = gl.get_limb(planet,14000.,'a1', source=ld_source)
        a2 = gl.get_limb(planet,14000.,'a2', source=ld_source)
        a3 = gl.get_limb(planet,14000.,'a3', source=ld_source)
        a4 = gl.get_limb(planet,14000.,'a4', source=ld_source)
    elif ld_type=='linear':
        a1 = gl.get_limb(planet,14000., 'u', source=ld_source)
        a2 = 0
        a3 = 0
        a4 = 0
    else:
        raise ValueError('Error: Must choose nonlinear (fixed) ' \
                         'and or linear (open) limb-darkening model.')
    props = np.append(props, [a1,a2,a3,a4])
    errs = np.append(errs, np.zeros(4))
    props = np.vstack((props, errs)).T
    props_hold = props.copy()
    #orbit = np.zeros(1)

    print("Number of total orbits: %d" % (len(orbit)-1))

    # Choose which orbits to include in the eclipse fitting. 1-2 on either
    # side of the eclipse is recommended
    check2 = check
    if check==False:
        if inp_file==True:
            df = pd.read_csv('./data_outputs/preprocess_info.csv')
            df = df[df.loc[:,'Transit']==transit]
            user_inputs = df.loc[visit+direction,'User Inputs'].values
        else:
            sys.exit('Either allow checking or give csv file with pandas info.')

        #allspec1d=np.ma.sum(allspec,axis=1).data
        #allerr1d=np.sqrt(np.ma.sum(allerr*allerr, axis=1)).data

        first_orbit=user_inputs[0]
        last_orbit=user_inputs[1]
        first_data = orbit[first_orbit]
        last_data=orbit[last_orbit+1]
        date=alldate[first_data:last_data]
        dir_array=dir_array[first_data:last_data]
        #allspec2d=allspec[first_data:last_data,:,:]
        #allerr2d=allerr[first_data:last_data,:,:]
        print('Are err1d and spec1d correctly defined below?')
        breakpoint()
        spec1d=allspec1d[first_data:last_data,:]
        err1d=allerr[first_data:last_data,:]
        #allspec1d=np.ma.sum(allspec2d,axis=1) #spectra for each exposure: these axes may be backwards
        #allerr1d=np.sqrt(np.ma.sum(allerr2d*allerr2d, axis=1))
        light = np.ma.sum(spec1d, axis=1) # total light for each exposure
        lighterr=np.sqrt(np.ma.sum(err1d*err1d, axis=1))

        user_inputs[5], user_inputs[6] = first_data, last_data

    if check == True:
        user_inputs=np.zeros(7)
        while check2==True:
            if data_plots==True:
                #err=np.sqrt(np.sum(np.sum(allerr[:,:,:]*allerr[:,:,:], axis=1), axis=1))
                #fl= np.sum(allspec[:,:,:], (1,2))
                err=np.sqrt(np.sum(allerr1d*allerr1d, axis=1))
                fl= np.sum(allspec1d, axis=1)
                plt.errorbar(alldate,fl,err, fmt='o')
                plt.xlabel('MJD')
                plt.ylabel('Total Flux')
                plt.show(block=False)

            first = input("Enter the first orbit to include (starting from 0): ")
            first_orbit=int(first)
            user_inputs[0]=first_orbit
            last= input("Enter the last orbit to include (starting form 0): ")
            last_orbit=int(last)
            if data_plots==True:
                plt.clf()
                plt.close()
            user_inputs[1]=last_orbit

            #allspec1d=np.ma.sum(allspec,axis=1).data
            #allerr1d=np.sqrt(np.ma.sum(allerr*allerr, axis=1)).data

            first_data = orbit[first_orbit]
            last_data=orbit[last_orbit+1]
            date=alldate[first_data:last_data]
            dir_array=dir_array[first_data:last_data]
            #spec2d=allspec[first_data:last_data,:,:]
            #err2d=allerr[first_data:last_data,:,:]
            spec1d=allspec1d[first_data:last_data,:]
            err1d=allerr1d[first_data:last_data,:]
            #spec1d=np.ma.sum(spec2d,axis=1)
            #err1d=np.sqrt(np.ma.sum(err2d*err2d, axis=1))
            light = np.ma.sum(spec1d,axis=1)
            lighterr=np.sqrt(np.ma.sum(err1d*err1d, axis=1))
            user_inputs[5], user_inputs[6] = first_data, last_data

            if data_plots==True:
                plt.errorbar(date, light/max(light),lighterr/max(light),fmt='o')
                plt.xlabel('MJD')
                plt.ylabel('Total Flux')
                plt.show(block=False)

            ans = input("Is this correct? (Y/N): ")
            if ans.lower() in ['y','yes']: check2=False
            if data_plots==True:
                plt.clf()
                plt.close()


    props[1, :] = event_time(date, props)
    user_inputs[4] = props[1, 0]

    #  We are only interested in scatter within orbits, so correct for flux
    #  between orbits by setting the median of each orbit to the median of
    #  the first orbit

  #  light_corrected=correction(props, date1, light, transit)

    # Do a 4-pass sigma cut. 3-5 sigma is ideal. Change n to see how data
    # is affected. A sigma of 3, 4, or 5 could be used, it depends on the
    # data
    # light2=light.copy()
    # lighterr2=lighterr.copy()
    # allspec2=allspec1.copy()
    # allerr2=allerr1.copy()
    # date2=date1.copy()
    # light_corrected2=light_corrected.copy()
    # ans2=''

    # if check==False:
    #     light, allspec1, light_corrected, date1, lighterr, allerr1 = remove_bad_data(light
    #                                                                                   , allspec1
    #                                                                                   , light_corrected
    #                                                                                   , date1
    #                                                                                   , lighterr
    #                                                                                   , allerr1
    #                                                                                   , user_inputs)
    # if check==True:
    #     while check==True:
    #         light=light2.copy()
    #         lighterr=lighterr2.copy()
    #         allspec1=allspec2.copy()
    #         allerr1=allerr2.copy()
    #         date1=date2.copy()
    #         light_corrected=light_corrected2.copy()

    #         # This performs the sigma cut and returns input for the fitter: a
    #         # double array which contains a spectra for each data point

    #         light, allspec1, light_corrected, date1, lighterr, allerr1 = remove_bad_data(light
    #                                                                                      , allspec1
    #                                                                                      , light_corrected
    #                                                                                      , date1
    #                                                                                      , lighterr
    #                                                                                      , allerr1
    #                                                                                      , user_inputs
    #                                                                                      , check=check)
    #         if ploton==True:
    #             plt.errorbar(date2, light2,lighterr2, fmt='ro')
    #             plt.xlabel('MJD')
    #             plt.ylabel('Total Flux')
    #             plt.errorbar(date1, light,lighterr,  fmt='o',ls='dotted')
    #             plt.show(block=False)
    #         ans2=raw_input('This is the new data, with the red points removed. Is this okay? (Y/N): ')
    #         if ploton==True: plt.close()
    #         if ans2.lower() in ['y','yes']: check=False

    # Set inclination (2), ars (3) to desired value if you want
    #props[2]=89.17
    #props[3]=5.55
    # dir_array has only been included in marg_mcmc so far
    #results=wl.whitelight2018(props, date, spec1d.data, err1d.data,
    #                          plotting=True, norandomt=norandomt,
    #                          openinc=openinc, openar=openar, fixtime=fixtime,
    #                          transit=transit, savewl=visit)
    print(props)
    save_name = visit + '/' + direction
    data_save_name = save_name
    if include_error_inflation == False:
        save_name = save_name + '_no_inflation'
    if ld_type == 'linear':
        save_name = save_name + '_linearLD'
    if ignore_first_exposures == True:
        save_name = save_name + '_no_first_exps'
    if openar == True:
        save_name = save_name + '_openar'
        
    results=wl.whitelight2020(props
                              , date
                              , spec1d.data
                              , err1d.data
                              , dir_array
                              , plotting=fit_plots
                              , mcmc=mcmc
                              , include_error_inflation=include_error_inflation
                              , norandomt=norandomt
                              , openinc=openinc
                              , openar=openar
                              , fixtime=fixtime
                              , ld_type=ld_type
                              , linear_slope=linear_slope
                              , quad_slope=quad_slope
                              , exp_slope=exp_slope
                              , log_slope=log_slope
                              , one_slope=one_slope
                              , transit=transit
                              , save_mcmc=save_mcmc
                              , save_model_info=save_model_info
                              , save_name=save_name)

    #direction = 'forward'
    if save_processed_data == True:
        sh=wl.get_shift(spec1d_sav)
        cols=['Pixel %03d' % i for i in range(spec1d_sav.shape[1])]
        subindex = ['Value']*spec1d_sav.shape[0] + ['Error']*spec1d_sav.shape[0]
        ind = pd.MultiIndex.from_product([[data_save_name], subindex])
        processed_data = pd.DataFrame(np.vstack((spec1d_sav, err1d_sav))
                                      , columns=cols, index=ind)
        processed_data['Date'] = np.append(date_sav, date_sav)
        processed_data['sh'] = np.append(sh, sh)
        processed_data['Mask'] = np.append(light_mask, light_mask)
        processed_data['Transit'] = transit
        processed_data['Scan Direction'] = np.append(dir_save, dir_save)
        sys_p=pd.DataFrame(props_hold, columns=['Properties'
                                                , 'Errors'])
        sys_p['Label'] = ['Rp/Rs', 'T0', 'i', 'a/rs', 'period'
                          , 'Depth', 'c1', 'c2', 'c3', 'c4']
        sys_p['Visit']=data_save_name
        sys_p=sys_p.set_index('Visit')
        u1 = gl.get_limb(planet,14000., 'u', source=ld_source)
        df2 = pd.DataFrame([[u1.round(4), 0, 'u1']]
                           , columns=sys_p.columns
                           , index=[sys_p.index[0]])
        sys_p = sys_p.append(df2)
        breakpoint()

        try:
            cur=pd.read_csv('./data_outputs/processed_data.csv', index_col=[0,1])
            cur=cur.drop(data_save_name, level=0, errors='ignore')
            cur=pd.concat((cur,processed_data), sort=False)
            cur.to_csv('./data_outputs/processed_data.csv', index_label=['Obs', 'Type'])
        except IOError:
            processed_data.to_csv('./data_outputs/processed_data.csv', index_label=['Obs','Type'])
        try:
            curr=pd.read_csv('./data_outputs/system_params.csv', index_col=0)
            curr=curr.drop(data_save_name, errors='ignore')
            curr=pd.concat((curr,sys_p), sort=False)
            curr.to_csv('./data_outputs/system_params.csv')
        except IOError:
            sys_p.to_csv('./data_outputs/system_params.csv', index_label='Obs')

    return [results, user_inputs]

if __name__=='__main__':

    config = configparser.ConfigParser()
    config.read('config.py')
    planet = config.get('DATA', 'planet')
    visit_number = config.get('DATA', 'visit_number')
    visit = planet + '/' + visit_number
    direction = config.get('DATA', 'scan_direction')
    transit = config.getboolean('DATA', 'transit')
    check = config.getboolean('DATA', 'check')
    ignore_first_exposures = config.getboolean('DATA', 'ignore_first_exposures')
    inp_file = config.getboolean('DATA', 'inp_file')
    data_plots = config.getboolean('DATA', 'data_plots')

    mcmc = config.getboolean('MODEL', 'mcmc')
    include_error_inflation = config.getboolean('MODEL', 'include_error_inflation')
    openar = config.getboolean('MODEL', 'openar')
    openinc = config.getboolean('MODEL', 'openinc')
    fixtime = config.getboolean('MODEL', 'fixtime')
    ld_type = config.get('MODEL', 'limb_type')
    ld_source = config.get('MODEL', 'limb_source')
    norandomt = config.getboolean('MODEL', 'norandomt')
    linear_slope = config.getboolean('MODEL', 'linear_slope')
    quad_slope = config.getboolean('MODEL', 'quad_slope')
    exp_slope = config.getboolean('MODEL', 'exp_slope')
    log_slope = config.getboolean('MODEL', 'log_slope')
    one_slope = config.getboolean('MODEL', 'one_slope')
    fit_plots = config.getboolean('MODEL', 'fit_plots')
   

    save_mcmc = config.getboolean('SAVE', 'save_mcmc')
    save_model_info = config.getboolean('SAVE', 'save_model_info')
    save_processed_data = config.getboolean('SAVE', 'save_processed_data')
    
    
    # assert(check != inp_file)

    print(visit)
    best_results, inputs= preprocess_whitelight(visit
                                                , direction
                                                , transit=transit
                                                , check=check
                                                , ignore_first_exposures=ignore_first_exposures
                                                , include_error_inflation=include_error_inflation
                                                , inp_file=inp_file
                                                , ld_source=ld_source
                                                , data_plots=data_plots
                                                , save_processed_data=save_processed_data
                                                , save_model_info=save_model_info
                                                , fixtime=fixtime
                                                , norandomt=norandomt
                                                , openar=openar
                                                , openinc=openinc
                                                , ld_type=ld_type
                                                , linear_slope=linear_slope
                                                , quad_slope=quad_slope
                                                , exp_slope=exp_slope
                                                , log_slope=log_slope
                                                , one_slope=one_slope
                                                , fit_plots=fit_plots
                                                , mcmc=mcmc
                                                , save_mcmc=save_mcmc)

 
    print(best_results)
    print("Marg Depth: %f +/- %f" % (best_results[0]*1e6, best_results[1]*1e6))
    print("Marg Central Event Time: %f +/- %f" % (best_results[2], best_results[3]))
    print("Marg Inclination: %f +/- %f" % (best_results[4], best_results[5]))
    print("Marg a/R*: %f +/- %f" % (best_results[6], best_results[7]))
    print("Marg limb darkening params: ", best_results[8], "+/-", best_results[9])

    save_name = visit + '/' + direction
    inp = pd.DataFrame(inputs, columns=['User Inputs'])
    inp['Visit'] = save_name
    inp['Transit'] = transit
    inp['Ignore first exposures'] = ignore_first_exposures
    inp = inp.set_index(['Visit', 'Ignore first exposures'])
    try:
        cur = pd.read_csv('./data_outputs/preprocess_info.csv', index_col=[0, 1])
        cur = cur.drop((visit+'/'+direction, ignore_first_exposures), errors='ignore')
        cur = pd.concat((cur,inp), sort=False)
        cur.to_csv('./data_outputs/preprocess_info.csv')
    except IOError:
        inp.to_csv('./data_outputs/preprocess_info.csv')

