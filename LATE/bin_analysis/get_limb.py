
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import interpolate

def get_limb(planet, x, limb_coeff, filt1='J', filt2='H', load=False):
    #lds=pd.read_table('asu-1.tsv')
    if load==True:
        lds=pd.read_csv('../planets/'+planet+'/lds.csv')
        return lds[limb_coeff].values
    else:
        #lds=pd.read_table('../planets/'+planet+'/3d.tsv')
        lds=pd.read_csv('../planets/'+planet+'/claret2011.csv')
        #lds=pd.read_csv('../planets/'+planet+'/claret2012.csv')

        lds['Filt']=lds['Band'].str.strip()

        key=pd.read_csv('filt_waves.csv')

        data=lds.set_index('Filt').join(key.set_index('Filt'), how='inner')
        data=data.sort_values('wave')
        data=data.loc[filt1:filt2]
        #return np.interp(x, data['wave'].values, data[limb_coeff].values)
        f=interpolate.interp1d(data['wave'].values
                               , data[limb_coeff].values
                               , fill_value='extrapolate')

        #plt.plot(x,  np.interp(x, data['wave'].values, data[limb_coeff].values), 'bo')
        #plt.plot(x, f(x), 'rx')
        #plt.show()
        return f(x)


def test_interp():
    """ Tests if linear interpolation of coefficients is
    equivalent to linear interpolation of limb darkening curves.

    Since I assume the change between coeffs is slow between similar wavelengths
    J and H - aka that it's linear - and since the curve depends linearly on
    the coefficients, it is okay just to interpolate coeffs. The only other
    assumption is that the baseline intensity is approximately the same.

    Testing the average wavelength between J and H confirms this. Interpolating
    the curves is equivalent ro interpolating the coeffs then generating
    a curve. True means they are equivalent. """

    lds=pd.read_table('3d.tsv')
    lds['Filt']=lds['Filt'].str.strip()
    key=pd.read_csv('filt_waves.csv')

    data=lds.set_index('Filt').join(key.set_index('Filt'),how='inner')
    print(data)
    j=data.loc['J'].values[:-1]
    h=data.loc['H'].values[:-1]
    avg=(j+h)/2.
    jcurve=limb_curve(j)
    hcurve=limb_curve(h)
    avgcurve=(jcurve+hcurve)/2
    avgcurve2=limb_curve(avg)
    x=np.linspace(1,0,1000)
    u=np.sqrt(1-x*x)
    plt.plot(u, jcurve, 'b')
    plt.plot(u, hcurve, 'r')
    plt.plot(u, avgcurve, 'g')
    plt.plot(u,avgcurve2, 'pink')
    plt.show()
    status = np.mean(np.abs(avgcurve-avgcurve2)) < 1e-5
    return status

def limb_curve(coeffs):
    """ Given the coefficients, generate a limb darkening curve assuming
    I_0 = 1 """
    c=np.ones(5)
    d=c.copy()
    d[0]=0
    c[1:]=coeffs*-1
    x=np.linspace(1,0,1000)
    exp=np.arange(5)/2.
    intensity=np.zeros(len(x))
    for i, item in enumerate(x):
        intensity[i]=np.sum(c*(1-d*item**exp))
    return intensity
