
import os
import sys


import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import time

import ramp2018_mcmc as wl_ramp2
import ramp2018 as wl_ramp
import batman
from RECTE import RECTE
sys.path.insert(0, '../bin_analysis')
import get_limb as gl

def get_data(visit, x, y, get_raw=0):
    folder = '../data_reduction/reduced/%s/final/*.fits' % (visit)
    data=np.sort(np.asarray(glob.glob(folder)))
    nexposure = len(data)
    print('There are %d exposures in this visit' % nexposure)

    date=np.zeros(len(data))
    icount=np.zeros_like(date)

    test=fits.open(data[0])
    xlen, ylen = test[0].data.shape
    test.close()
    xlen-=2*x
    ylen-=2*y
    alldata=np.ma.zeros((len(data),xlen, ylen))
    allerr=np.zeros((len(data),xlen,ylen))
    allraw=allerr.copy()

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
        raws=expfile[3].data
        expfile.close()
        date[i]=(hdr['EXPSTART']+hdr['EXPEND'])/2.
        icount[i]=hdr['COUNT']
        exptime=hdr['EXPTIME']
        expo=exp[xmin:xmax, ymin:ymax]
        mask=mask[xmin:xmax, ymin:ymax]
        errs=errs[xmin:xmax, ymin:ymax]
        raws=raws[xmin:xmax, ymin:ymax]
        alldata[i,:,:]=np.ma.array(expo, mask=mask)
        allerr[i,:,:]=errs
        allraw[i,:,:]=raws

    date_order=np.argsort(date)
    date=date[date_order]
    icount=icount[date_order]
    alldata=alldata[date_order,:,:]
    allerr=allerr[date_order,:,:]
    allraw=allraw[date_order, :,:]

    if get_raw != 0:
        return allraw, exptime
    else:
        return date, icount, alldata, allerr, allraw, exptime



def event_time(date, properties):
    """Program to determine the expected event time
     Inputs
     date: 1D array of the date of each exposure (MJD)
     properties: 1D array containing the last observed eclipse
     and the period. (MJD, days)"""
    time=properties[1]
    period=properties[4]
    while time < date[0]:
        time+=period
    return float(time)

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
    data: data table of priors for a particular planet
    OUTPUTS:
    Returns array of system properties: [rprs, central event time, inc
    ,a/r, period, depth]
    """
    inp_values=pd.read_table(data,sep=' ')
    data_arr=inp_values.iloc[:,2].values
    labels=inp_values.iloc[:,0].values
    param_errs=inp_values.iloc[:,3].values

    # Rj-m, Rsolar-m,AU-m, JD -> MJD
    conversions=np.array([6.9911e7, 6.957e8, 1.49598e11, -2400000.5])
    inc=data_arr[5]
    period=data_arr[4]
    a_R=data_arr[7]*conversions[2]/(data_arr[1]*conversions[1])
    a_R_err=np.sqrt((param_errs[7]*conversions[2]/data_arr[1]/conversions[1])**2
                    + (a_R*param_errs[1]/conversions[1])**2)
    rprs = data_arr[0]*conversions[0]/(data_arr[1]*conversions[1])

    if transit==True:
        epoch=data_arr[6]+conversions[3]
        depth=rprs*rprs
    else:
        epoch=data_arr[6]+conversions[3]+period/2.
        depth = rprs*rprs*(data_arr[2]/data_arr[3])/3

    props=np.zeros(6)
    props[0]=rprs
    props[1]=epoch
    props[2]=inc
    props[3]=a_R
    props[4]=period
    props[5]=depth

    errors=np.zeros(6)
    errors[0]=0
    errors[1]=param_errs[6]
    errors[2]=param_errs[5]
    errors[3]=a_R_err
    errors[4]=param_errs[4]
    errors[5]=0

    return [props,errors]

def intrinsic(date, light, raw, pixels=0):
    means=np.zeros(len(light))
    for i,l in enumerate(light):
        try:
            means[i]=np.mean([light[i-3], light[i-2], light[i-1], l])
        except IndexError:
            means[i]=l

    end=np.argmax(means)
    begin=end-3
    if pixels==0:
        raw_img=np.median(raw[begin:end,:,:], axis=0)
    else:
        raw_img=np.median(raw[begin:end,:,pixels[0]:pixels[1]])
    intrinsic=np.median(raw_img)

    return intrinsic


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


# def remove_bad_data(light_corrected, date1, user_inputs, check=False):
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
#         plt.close()

#     Cut out the "bad" data

#     med= np.ma.median(light_corrected)
#     sigma = np.sqrt(np.sum((light_corrected-med)**2)/(2*len(light_corrected)))
#     dif= np.abs(light_corrected-med)
#     index=np.where(dif < sigma_cut_factor*sigma)[0]

#     return index


def preprocess_ramp(visit, direction, x=0, y=0, ploton=True
                          , check=True, inp_file=False, savedata=False
                          , transit=False):

    """
    PURPOSE: Allow user to e xtract relevant orbital data from reduced time
    series of a visit. Also allow user to exclude any outlier data points.

    INPUTS

     x, y, allow the user to reduce aperture
     checks: set to "on" to manually reduce data

     If checks is set to on, "user_inputs" will return the inputs
     that the user used: [first orbit, last orbit, sigma cut factor,
     number of passes, center eclipse time]. If checks is set to off, then
     the user_inputs array will be used as inputs (easier to automate) """

    date, icount, alldata, allerr, allraw, exptime=get_data(visit+'/'+ direction, x, y)

    nexposure=len(date)
    planet = visit[:-8]
    props, errs=inputs('../planets/%s/inputs.dat' % planet, transit)
    a1=gl.get_limb(planet, 14000.,'a1')
    a2=gl.get_limb(planet, 14000.,'a2')
    a3=gl.get_limb(planet, 14000.,'a3')
    a4=gl.get_limb(planet, 14000.,'a4')
    props=np.append(props, [a1,a2,a3,a4])
    errs=np.append(errs, np.zeros(4))
    props_hold=props.copy()
    orbit = np.zeros(1)

    # Classify the data by each HST orbit. Returns array (orbit)
    # which contains the indeces for the start of each orbit

    orbit=get_orbits(date)

    print("Number of total orbits: %d" % (len(orbit)-1))

    # Choose which orbits to include in the eclipse fitting. 1-2 on either
    # side of the eclipse is recommended
    check2=check
    if check == False:
        if inp_file == True:
            df=pd.read_csv('./preprocess_info.csv')
            df=df[df.loc[:,'Transit']==transit]
            user_inputs=df.loc[visit+direction,'User Inputs'].values
        else:
            sys.exit('Either allow checking or give csv file with pd info.')

        first_orbit=user_inputs[0]
        last_orbit=user_inputs[1]
        date1=date[orbit[first_orbit]:orbit[last_orbit+1]]
        alldata=alldata[orbit[first_orbit]:orbit[last_orbit+1],:,:]
        allerr=allerr[orbit[first_orbit]:orbit[last_orbit+1],:,:]
        allraw=allraw[orbit[first_orbit]:orbit[last_orbit+1],:,:]
        allspec=np.ma.sum(alldata,axis=1) #spectra for each exposure: these axes may be backwards
        specerr=np.sqrt(np.sum(allerr*allerr, axis=1))
        light = np.ma.sum(allspec, axis=1) # total light for each exposure
        lighterr=np.sqrt(np.sum(specerr*specerr, axis=1))

    if check == True:
        user_inputs=np.zeros(5)
        while check2==True:
            if ploton==True:
                col_err=np.sqrt(np.sum(allerr*allerr, axis=1))
                err=np.sqrt(np.sum(col_err*col_err, axis=1))
                fl= np.sum(alldata, (1,2))
                plt.errorbar(date,fl,err, fmt='o')
                plt.xlabel('MJD')
                plt.ylabel('Total Flux')
                plt.show(block=False)
            first = input("Enter the first orbit to include (starting from 0): ")
            first_orbit=int(first)
            user_inputs[0]=first_orbit
            last= input("Enter the last orbit to include (starting form 0): ")
            last_orbit=int(last)
            if ploton==True: plt.close()
            user_inputs[1]=last_orbit
            date1=date[orbit[first_orbit]:orbit[last_orbit+1]]
            alldata1=alldata[orbit[first_orbit]:orbit[last_orbit+1],:,:]
            allerr1=allerr[orbit[first_orbit]:orbit[last_orbit+1],:,:]
            allraw1=allraw[orbit[first_orbit]:orbit[last_orbit+1],:,:]
            # date1=date[first_orbit:last_orbit-1]
            # allspecextract1=allspecextract[first_orbit:last_orbit-1,:,:]
            # STOP, 'change eclipse2017 back to orbit'

            allspec=np.ma.sum(alldata1,axis=1)
            specerr=np.sqrt(np.sum(allerr1*allerr1, axis=1))
            light = np.ma.sum(allspec,axis=1)
            lighterr=np.sqrt(np.sum(specerr*specerr, axis=1))

            if ploton==True:
                plt.errorbar(date1, light/max(light),lighterr/max(light),fmt='o')
                plt.xlabel('MJD')
                plt.ylabel('Total Flux')
                plt.show(block=False)

            ans = input("Is this correct? (Y/N): ")
            if ans.lower() in ['y','yes']: check2=False
            if ploton==True:  plt.close()

    props[1]=event_time(date1, props)
    user_inputs[4]=props[1]

    count=intrinsic(date1, light.data, allraw)/exptime
    #  We are only interested in scatter within orbits, so correct for flux
    #  between orbits by setting the median of each orbit to the median of
    #  the first orbit

    # light_corrected=correction(props, date1, light, transit)

    # Do a 4-pass sigma cut. 3-5 sigma is ideal. Change n to see how data
    # is affected. A sigma of 3, 4, or 5 could be used, it depends on the
    # data
    # light1=light.copy()
    # lighterr1=lighterr.copy()
    # allspec1=allspec.copy()
    # specerr1=specerr.copy()
    # alldata1=alldata.copy()
    # date2=date1.copy()
    # light_corrected1=light_corrected.copy()
    # icount1=icount.copy()
    # ans2=''

    # if check==False:
    #    index = remove_bad_data(light_corrected, date1, user_inputs, check=check)
    #    light=light[index]
    #    date1=date1[index]
    #    light_corrected=light_corrected[index]
    #    lighterr=lighterr[index]
    #    allspec=allspec[index,:]
    #    specerr=specerr[index,:]
    #    alldata=alldata[index, :, :]
    #    icount=icount[index]
    # if check==True:
    #     while check==True:
    #         light=light1.copy()
    #         lighterr=lighterr1.copy()
    #         allspec=allspec1.copy()
    #         alldata=alldata1.copy()
    #         specerr=specerr1.copy()
    #         date1=date2.copy()
    #         light_corrected=light_corrected1.copy()
    #         icount=icount1.copy()

    #         This performs the sigma cut and returns input for the fitter: a
    #         double array which contains a spectra for each data point

    #         index = remove_bad_data(light_corrected, date1, user_inputs, check=check)
    #         light=light[index]
    #         date1=date1[index]
    #         light_corrected=light_corrected[index]
    #         lighterr=lighterr[index]
    #         allspec=allspec[index,:]
    #         specerr=specerr[index,:]
    #         alldata=alldata[index, :,:]
    #         icount=icount[index]
    #         if ploton==True:
    #             plt.errorbar(date2, light1,lighterr1, fmt='ro')
    #             plt.xlabel('MJD')
    #             plt.ylabel('Total Flux')
    #             plt.errorbar(date1, light,lighterr,  fmt='o',ls='dotted')
    #             plt.show(block=False)
    #         ans2=raw_input('This is the new data, with the red points removed. Is this okay? (Y/N): ')
    #         if ploton==True: plt.close()
    #         if ans2.lower() in ['y','yes']: check=False

    if transit==True:
        fixtime=False
        norandomt=True
        openar=False
        #openinc=True
        openinc=False
    else:
        fixtime=False
        norandomt=True
        openar=True
        openinc=False
    savewl=visit+'/'+direction
    #savewl='hatp41/no_first_orbit'
    #savewl=False

    # no mcmc
    #results=wl_ramp.ramp2018(props, date1, allspec, specerr, count, exptime, plotting=True
    #                         , norandomt=norandomt, openinc=openinc, openar=openar
    #                         , fixtime=fixtime, transit=transit, savewl=savewl)
    # mcmc
    results=wl_ramp2.ramp2018(props, errs, date1, allspec, specerr, count, exptime, plotting=True
                                   , norandomt=norandomt, openinc=openinc, openar=openar
                                   , fixtime=fixtime, transit=transit, savewl=savewl)
    #visit='hatp41/no_first_orbit'

    if savedata:
        visit = visit+'/'+direction
        cols=['Pixel ' + '%03d' % i for i in range(allspec.shape[1])]
        subindex=['Value']*allspec.shape[0] + ['Error']*allspec.shape[0]
        ind=pd.MultiIndex.from_product([[visit], subindex])
        processed_data=pd.DataFrame(np.vstack((allspec, specerr)),columns=cols, index=ind)
        processed_data['Date']=np.append(date1, date1)
        processed_data['Transit']=transit


        sys_p=pd.DataFrame(np.vstack((props_hold, errs)).T, columns=['Properties'
                                                                 , 'Errors'])
        sys_p['Visit']=visit
        sys_p=sys_p.set_index('Visit')

        try:
            cur=pd.read_csv('./processed_ramp.csv', index_col=[0,1])
            cur=cur.drop(visit, level=0)
            cur=pd.concat((cur,processed_data), sort=False)
            cur.to_csv('./processed_ramp.csv', index_label=['Obs', 'Type'])
        except IOError:
            processed_data.to_csv('./processed_ramp.csv',index_label=['Obs', 'Type'])
        try:
            curr=pd.read_csv('./system_params.csv', index_col=0)
            curr=curr.drop(visit)
            curr=pd.concat((curr,sys_p), sort=False)
            curr.to_csv('./system_params.csv')
        except IOError:
            sys_p.to_csv('./system_params.csv', index_label='Obs')

    return [results, user_inputs]

if __name__=='__main__':

    if len(sys.argv) < 4:
        sys.exit('Format: preprocess_ramp.py [planet] [visit] [direction]')
    visit=sys.argv[1]+'/'+sys.argv[2]
    direction=sys.argv[3]
    transit=False
    if len(sys.argv)==5:
        transit=bool(int(sys.argv[4]))

    best_results, inputs = preprocess_ramp(visit, direction
                                                 ,transit=transit, savedata=True)

    print(best_results)
    print("Depth: %f +/- %f" % (best_results[0]*1e6, best_results[1]*1e6))
    print("Central Event Time: %f +/- %f" % (best_results[2], best_results[3]))
    print("Inclination: %f +/- %f" % (best_results[4], best_results[5]))
    print("a/R*: %f +/- %f" % (best_results[6], best_results[7]))
    print("Limb darkening params: ", best_results[8], "+/-", best_results[9])


    inp=pd.DataFrame(inputs, columns=['User Inputs'])
    inp['Visit']=visit+'/'+direction
    inp['Transit']=transit
    inp=inp.set_index('Visit')
    try:
        cur=pd.read_csv('./preprocess_ramp_info.csv', index_col=0)
        cur=cur.drop(visit+'/'+direction, errors='ignore')
        cur=pd.concat((cur,inp), sort=False)
        cur.to_csv('./preprocess_ramp_info.csv')
    except IOError:
        inp.to_csv('./preprocess_ramp_info.csv')


