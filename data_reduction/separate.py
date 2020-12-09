import sys
import os

import numpy as np
import glob
from astropy.io import fits
import shutil
from shutil import move

def separate(planet):

    """ Go to all data file in a directory and separate them based on date
    (so that different visits are separated)"""

    # Read in files
    # Find the date of each
    # if date2-date1 greater than a hubble orbit period * 2, then save
    # files in new directory

    data=np.sort(np.asarray(glob.glob('../planets/%s/*ima.fits' % planet)))
    raw_data=np.sort(np.asarray(glob.glob('../planets/%s/*raw.fits' % planet)))
    date=np.zeros(len(data))


    # Read in dates for each exposure
    for i, img in enumerate(data):
        fit=fits.open(img)
        date[i]=(fit[0].header['EXPSTART']+fit[0].header['EXPEND'])/2.
        fit.close()

    idx = np.argsort(date)
    date = date[idx]
    data = data[idx]
    try:
        raw_data = raw_data[idx]
    except IndexError:
        pass
    time=np.zeros(len(data)-1)
    visit=np.zeros_like(date)
    hst_period=95.47
    
    for i in range(len(date)-1):
        t=np.abs(date[i+1]-date[i])*24*60
        time[i]=t
        
        #  If time between data points is greater than 
        #  3 HST orbits from previous exposure,
        #  then I classify it as a new observation
        
        if t/hst_period > 3: visit[i]=1

    nObs=np.sum(visit)+1
    fNames=np.arange(nObs)
    dirs=['../planets/%s/visit%02i/' % (planet, name) for name in fNames]
    raw_dirs=['../planets/%s/visit%02i/raw/' % (planet, name) for name in fNames]
    for dir in raw_dirs:
        try:
            os.makedirs(dir)
        except OSError:
            if os.path.isdir(dir):
                shutil.rmtree(dir)
                os.makedirs(dir)
            else:
                raise
    
    direct=dirs[0]
    raw_d=raw_dirs[0]
    numV=0
    for i, img in enumerate(data):
        move(img, direct+img.split('/')[-1])
        try:
            move(raw_data[i], raw_d+raw_data[i].split('/')[-1])
        except IndexError:
            pass
        if visit[i] == 1:
            numV=numV+1
            direct=dirs[numV]
            try:
                raw_d=raw_dirs[numV]
            except IndexError:
                pass
    return 

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('%run separate.py [planet]')
    planet=sys.argv[1]
    separate(planet)
