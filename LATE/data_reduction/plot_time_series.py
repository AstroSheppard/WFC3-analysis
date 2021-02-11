import configparser
import sys
import glob

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


def get_orbits(date):
    """Procedure to organize light curve data by HST orbit"""
    orbit=np.zeros(1).astype(int)

    for i in range(len(date)-1):
        t=date[i+1]-date[i]
        if t*86400 > 1200.:
            orbit=np.append(orbit, i+1) # 1800s is about half an HST orbit
    return np.append(orbit, len(date))


def plot_time_series(visit
                     , direction
                     , x=0
                     , y=0
                     , ploton=True
                     , check=True):

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


    date_order=np.argsort(alldate)
    alldate=alldate[date_order]
    allspec=allspec[date_order,:,:]
    allerr=allerr[date_order,:,:]
    exp_numbers = np.arange(len(alldate))
    # Classify the data by each HST orbit. Returns array (orbit)
    # which contains the indeces for the start of each orbit

    orbit=get_orbits(alldate)

    print("Number of total orbits: %d" % (len(orbit)-1))

    # Choose which orbits to include in the eclipse fitting. 1-2 on either
    # side of the eclipse is recommended

    check2=check

  
    if check == True:
        user_inputs=np.zeros(7)
        while check2==True:
            plt.close()
            if ploton==True:
                err=np.sqrt(np.sum(np.sum(allerr[:,:,:]*allerr[:,:,:], axis=1), axis=1))
                fl= np.sum(allspec[:,:,:], (1,2))
                plt.errorbar(alldate,fl,err, fmt='o')
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
            check_number = input("List exposure number? (y/n) ")
            
            allspec1d=np.ma.sum(allspec,axis=1).data
            allerr1d=np.sqrt(np.ma.sum(allerr*allerr, axis=1)).data

            first_data = orbit[first_orbit]
            last_data=orbit[last_orbit+1]
            date=alldate[first_data:last_data]
            exp_nums = exp_numbers[first_data:last_data]
            spec2d=allspec[first_data:last_data,:,:]
            err2d=allerr[first_data:last_data,:,:]
            spec1d=np.ma.sum(spec2d,axis=1)
            err1d=np.sqrt(np.ma.sum(err2d*err2d, axis=1))
            light = np.ma.sum(spec1d,axis=1)
            lighterr=np.sqrt(np.ma.sum(err1d*err1d, axis=1))
            user_inputs[5], user_inputs[6] = first_data, last_data

            if ploton==True:
                plt.errorbar(date, light/max(light),lighterr/max(light),fmt='o')
                plt.xlabel('MJD')
                plt.ylabel('Total Flux')
                if check_number == 'y':
                    for  x, y, i in zip(date, light/max(light), exp_nums):
                        plt.text(x, y, str(i), color='r', fontsize=12)
                plt.show(block=False)

            ans = input("Is this correct? (Y/N): ")
            if ans.lower() in ['y','yes']: check2=False
            if ploton==True:
                pass

if __name__=='__main__':

    config = configparser.ConfigParser()
    config.read('config.py')
    planet = config.get('DATA', 'planet')
    visit_number = config.get('DATA', 'visit_number')
    direction = config.get('DATA', 'scan_direction')
    if len(sys.argv) != 1:
        sys.exit('Set inputs using config.py file.')
        
    visit = planet + '/' + visit_number
    plot_time_series(visit, direction)
