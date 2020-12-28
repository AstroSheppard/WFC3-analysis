import sys
import glob

import numpy as np
from astropy.io import fits
import pandas as pd
import matplotlib.pyplot as plt

import .fullzap 
from .bkg import get_data

def flatfield(visit, direction):
    """ Exps is every exposure in a visit/direction
    wave is wavelength solution of observation
    sub is size of subarray for observation"""

    data=np.sort(glob.glob('../data_reduction/reduced/'
                           + visit +'/'+ direction +'/bkg/*.fits'))

    x1,x2,y1,y2=pd.read_csv('coords.csv'
                            , index_col=0).loc[visit,'Initial Aperture'].values
    
    xd=x2-x1
    yd=y2-y1

    nexp=len(data)
    all_img=np.empty((nexp, xd, yd))
    all_err=all_img.copy()
    all_raw=all_img.copy()
    headers=np.array([])
    for i,obs in enumerate(data):
        exp=fits.open(obs)
        hdr=exp[0].header.tostring(sep='\\n')
        img=exp[0].data
        err=exp[1].data
        raw=exp[2].data
        exp.close()
        headers=np.append(headers,hdr)
        all_img[i, :,:]=img
        all_err[i,:,:]=err
        all_raw[i,:,:]=raw

    return [all_img, headers, all_err, all_raw]

def dq(visit, direction, data):
    x1,x2,y1,y2=pd.read_csv('coords.csv'
                            , index_col=0).loc[visit,'Initial Aperture'].values

    ima, raw=get_data(visit)
    dq_array=np.zeros(data.shape)
    for i, item in enumerate(ima):
        obs=fits.open(item)
        dqi=obs[3].data
        obs.close()
        dqi=dqi[x1:x2,y1:y2]
        dq_array[i, :,:]=dqi
   
    #dq_array=dq_array[:, 25:47, 42:186]
    #for j in range(dq_array.shape[1]):
    #    plt.plot(dq_array[:,j, 106], label='%d'%j, color='b')
    #    plt.legend()
    #    plt.show()
    poor=(dq_array != 0) * (dq_array != 2048) * (dq_array != 8)
    same=np.sum(poor, axis=0)
    mask=(same==poor.shape[0]).astype(bool)

    return mask



if __name__=='__main__':
    if len(sys.argv) < 4 or len(sys.argv) > 6:
        sys.exit('Run using [program.py] [planet] [visit #] [scan direction] [transit?] [plot?]')
    visit=sys.argv[1]+'/'+sys.argv[2]
    direction=sys.argv[3]
    if len(sys.argv) == 6:
        plotting=int(sys.argv[5])
        transit=int(sys.argv[4])
    elif len(sys.argv) == 5:
        plotting=False
        transit=int(sys.argv[4])
    else:
        plotting=False
        transit=False
        

    data, headers, errors, raw=flatfield(visit, direction)
    
    print 'flats done'
    mask=dq(visit, direction, data)
    mask=np.broadcast_to(mask, data.shape)
    data=np.ma.array(data, mask=mask)
    cr_data=fullzap.zapped(data)
    
    print 'cr done'
    filename = './reduced/'+ visit + '/'+direction+'/final/'
    fullzap.bad_pixels(cr_data, headers, errors, raw, filename)
    

