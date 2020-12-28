import sys

import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits

from . import mpfit


def pixel_scale(y):
    """Function (from WFC3 paper),
    to determine max and min scale 
    for given ycen"""
    min=.0028*y+44.68
    max=.0026*y+45.112
    out=np.asarray([min,max])
    return out

def solution(p, flux, error, sensitivity, star_lines, model_wave, coeffs, fjac=None):
    """ Fitting function for mpfit, which will minimize the returned deviates. 
    This compares the model stellar spectra*sensitivity to observed spectrum
    to get wavelength of each pixel."""
    
    # Convert pixels to wavelengths

    xref=coeffs[0]+coeffs[1]*(0-p[0])
    pixel=range(len(flux))
    x=p[1]*(pixel+p[2])+xref
    
    # calculate the model (sensitivity function x stellar spectra)  
    model=star_lines*sensitivity
    model=model/np.max(model)
 
    # Interpolate model to match data wavelength
    theory=np.interp(x, model_wave, model)
    status=0
    return [status,(flux-theory)/error]
  
def orbits(data, phase_curve=False, transit=False, **kwargs):
    """Find OOE or OOT exposure to use as stellar
    spectrum. STILL NEED TO TEST"""
    if len(kwargs) != 2:
        date=np.zeros(len(data))
        wl=np.zeros(len(data))
        for i,img in enumerate(data):
            # Recover dates
            exp=fits.open(img)
            date[i]=(exp[0].header['EXPSTART']+exp[0].header['EXPEND'])/2.
            wl[i]=np.sum(exp[0].data)
            exp.close()
    else:
        wl=kwargs['y']
        date=kwargs['x']

    orbit=np.zeros(1).astype(int)
    nexposure=len(data)
    for i in range(len(date)-1):
        t=date[i+1]-date[i]
        if t*86400 > 1800: orbit=np.append(orbit, i+1)
  
    orbit=np.append(orbit,len(date))

    if transit == True:
        expo=orbit[1]-3
        expos=(orbit[0], orbit[1])
    else:
        nOrbits=len(orbit)-1
        avg_orbit=np.zeros(nOrbits)
        dif=np.zeros(nOrbits-1)
        i=0
        while i < nOrbits:
            avg_orbit[i]=np.median(wl[orbit[i]:orbit[i+1]])
            i=i+1
        for i in range(nOrbits-1):
            dif[i]=avg_orbit[i+1]-avg_orbit[i]
        eclipse=np.argmin(dif)+1
        expos=(orbit[eclipse],orbit[eclipse+1])
        a=np.sort(dif)
        if phase_curve: eclipse=a[phase_curve]+1 #2 or 1
        expo=np.where(wl==np.min(wl[orbit[eclipse]:orbit[eclipse+1]]))
    return [expo, expos]

def wave_solution(visit, dire, level, plotting=False, savename=False, phase=False, transit=False):
    """ Implements MPFIT to determine physical wavelength for each pixel 
    for future spectral fitting."""
    
    ### Model spectra
    # Read in model wavelength
    file='../planets/'+visit[:-8]+'/wave.dat'
    wave=pd.read_table(file, sep=' ', header=None).values.flatten()
    wave=wave[~np.isnan(wave)]
    index=(wave>1060.) * (wave<1700)
    wavelength=wave[index]

    # Read in model continuum
    file='../planets/'+visit[:-8]+'/continuum.dat'
    cont=np.genfromtxt(file, dtype=str)
    nNum=10
 
    # Convert file to double array
    con=[]
    for line in cont:
        for i in range(len(line)/nNum):
            con.append(line[nNum*i:nNum*(i+1)])
    con=np.asarray(con).astype(float)        
    cont=con[index]

    # Read in model lines
    file='../planets/'+visit[:-8]+'/kurucz.dat'
    lines=np.genfromtxt(file,dtype=str)
  
    # Convert to double array
    lines_hold=[]
    for line in lines:
        for i in range(len(line)/nNum):
            lines_hold.append(line[nNum*i:nNum*(i+1)])
    lines=np.asarray(lines_hold).astype(float)        
    line=lines[index]

    # Final model
    f=(line+cont)/wavelength/wavelength

    # Read in sensitivity file
    sfile='./sensitivity.fits'
    sens_fits=fits.open(sfile)
    wssens=sens_fits[1].data.WAVELENGTH
    through=sens_fits[1].data.SENSITIVITY
    sens_fits.close()
    
    # Convert model wavelength to same units (angstroms)
    wavelength=wavelength*10.
    # Interpolate (linear) sensitivity array to model wavelength grid
    result=np.interp(wavelength, wssens, through)
    
  
    ######## Observed Spectrum ########

    # Find an exposure of just the stellar spectrum (either in eclipse or
    # before ingress
    
    data=np.sort(glob.glob('../data_reduction/reduced/'
                   + visit +'/'+ dire +'/'+level+'/*.fits'))

    exp=orbits(data, phase=phase, transit=transit)[0] # may have phase mixed up
    
    # Read in data, and normaliza the spectrum
    
    data=np.sort(glob.glob('../data_reduction/reduced/'+ visit
                           +'/'+ dire +'/'+level+"/%03d*"%exp))

    spec=np.sum(fits.open(data[0])[0].data, axis=0)
    spec[spec<0]=0
    err=np.sqrt(spec)
    # Normalize
    err=err/np.max(spec)
    err[err==0]=np.median(err)
    spec=spec/np.max(spec)

    ### HERE ###
    # Get xcen and ycen for this dataset from the photometry file
    raw=glob.glob('../planets/'+visit+'/*ima.fits')

    # Isolate the photometric data
    xcen,ycen=0,0
    for img in raw:
        exp=fits.open(img)
        hdr1=exp[0].header
        if 'IMAGING' in hdr1['OBSTYPE']:
            # Find the reference pixel
            hdr=exp[1].header
            xcor=hdr['LTV1']
            ycor=hdr['LTV2']
            xref=hdr['CRPIX1']
            yref=hdr['CRPIX2']
            xcen=xref-xcor
            ycen=yref-ycor
            exp.close()
            break
        else:
            exp.close()
   
    # Find range of pixel scales for a given center y
    limits = pixel_scale(ycen)
    scale=np.mean(limits)
        
    ####### FIT MODEL TO DATA ######

    a=np.asarray([8.95431e3, 9.35925e-2])
    xshift=0

    p0=[xcen, scale, xshift]
    fix=np.array([1,0,0])
    parinfo=[]
    for i in range(len(p0)):
        parinfo.append({'value':0., 'fixed':0, 'limited':[0,0],
               'limits':[0.,0.]})
        parinfo[i]['value']=p0[i]
        parinfo[i]['fixed']=fix[i]

    parinfo[1]['limited']=[1,1]
    parinfo[1]['limits']=[limits[0], limits[1]]


    fa={'flux':spec, 'error':err, 'sensitivity':result, 'star_lines':f,
        'model_wave':wavelength, 'coeffs':a}

    m=mpfit.mpfit(solution,parinfo=parinfo, functkw=fa)
    # Common attributes
    nfree=len(fix)-fix.sum()
    dof=len(spec)-nfree
    perror=m.perror
    parameters=m.params
    bestnorm=m.fnorm
    covar=m.covar
    niter=m.niter
    pcerror=perror*np.sqrt(bestnorm/nfree)
        
    ###### Plot results ######
    xlen=len(spec)
    data_wave=np.zeros(xlen)
    xref=a[0]+a[1]*(0-parameters[0]) 
    for i in range(xlen):
        data_wave[i]=(parameters[1]*(i+parameters[2]))+xref
  
    model=f*result
    model=model/np.max(model)
 
    if plotting == True:
        p=plt.plot(data_wave/1e4, spec, 'red')
        plt.xlabel('Wavelength [$\mu$m]')
        plt.ylabel('Normalized Flux')
        p=plt.plot(wavelength/1e4, model, 'blue')
        p=plt.plot(wssens/1e4,through/max(through)/2, color='green')
        p=plt.plot(wavelength/1e4, f/max(f), 'black')
        t=plt.text(1.17,.2,'Observed Spectrum',  color='red')
        t=plt.text(1.17,.1,'Model Spectrum',  color='blue')
        t=plt.text(1.45,.2,'Model Stellar Flux',  color='black')
        t=plt.text(1.45,.1,'G141 Sensitivity',  color='green')
  
    if savename:
        names=visit.split('/')
        plt.savefig('./wave_sol/'+names[0]+names[1][-3:]+'_wavefit.pdf')     
        
        wave_solution=pd.DataFrame(data_wave, columns=['Wavelength Solution [A]'])
        wave_solution['Visit']=visit
        wave_solution['Transit']=transit
        wave_solution=wave_solution.set_index('Visit')
        try:
            current=pd.read_csv('./wave_sol/wave_solution.csv', index_col=0)
            current=current.drop(visit)
            current=pd.concat((current, wave_solution))
            current.to_csv('./wave_sol/wave_solution.csv')
        except IOError:
            wave_solution.to_csv('./wave_sol/wave_solution.csv')
        
        headers=np.asarray(['Model Wavelength [A]', 'Model Spectrum'])
        extras=pd.DataFrame(list(zip(wavelength, model)),columns=headers)
        extras['Xcen']=parameters[0]
        extras['Pixel Scale']=parameters[1]
        extras['Xshift']=parameters[2]
        extras['Coeff_0']=a[0]
        extras['Coeff_1']=a[1]
        extras['Visit']=visit
        extras['Transit']=transit
        extras=extras.set_index('Visit')
        try:
            cur=pd.read_csv('./wave_sol/wave_solution_extras.csv', index_col=0)
            cur=cur.drop(visit)
            cur=pd.concat((cur,extras))
            cur.to_csv('./wave_sol/wave_solution_extras.csv')
        except IOError:
            extras.to_csv('./wave_sol/wave_solution_extras.csv')
    
    if plotting==True: plt.show()
    return data_wave

if __name__ == '__main__':
    if len(sys.argv) < 4 or len(sys.argv) > 6:
        sys.exit('Run using [program.py] [planet] [visit #] [scan direction] [transit?] [plot?]')
    visit=sys.argv[1]+'/'+sys.argv[2]
    direction=sys.argv[3]
    if len(sys.argv) == 6:
        plotting=int(sys.argv[5])
        trans=int(sys.argv[4])
    elif len(sys.argv) == 5:
        plotting=False
        trans=int(sys.argv[4])
    else:
        plotting=False
        trans=False
    
    wave=wave_solution(visit, direction, 'final'
                       , plotting=plotting, savename=True, transit=trans)
