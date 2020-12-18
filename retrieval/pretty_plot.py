#from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import nestle
#import dynesty
import random
import corner
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d as gf
import time
from matplotlib.ticker import ScalarFormatter
from scipy.io import readsav
import matplotlib.cm as cm

from platon.fit_info import FitInfo
from platon.constants import R_sun, R_jup, M_jup, METRES_TO_UM, G, AMU, k_B
from platon.transit_depth_calculator import TransitDepthCalculator
from platon.eclipse_depth_calculator import EclipseDepthCalculator
from platon.abundance_getter import AbundanceGetter
from new_combined_retriever import CombinedRetriever as Retriever
from retrieve import get_data



def get_aura():
    dir = './figure_pickle_files/aura_model/'
    waves = np.loadtxt(dir + 'Aura_low1sigma.txt')[:,0]/1.0e6
    down2 = np.loadtxt(dir + 'Aura_low2sigma.txt')[:,1]
    down1 = np.loadtxt(dir + 'Aura_low1sigma.txt')[:,1]
    medians=np.loadtxt(dir + 'Aura_median_result.txt')[:,1]
    up1 = np.loadtxt(dir + 'Aura_high1sigma.txt')[:,1]
    up2 = np.loadtxt(dir + 'Aura_high2sigma.txt')[:,1]

    smooth = 5
    medians=gf(medians,smooth)
    up1=gf(up1, smooth)
    up2=gf(up2, smooth)
    down1=gf(down1, smooth)
    down2=gf(down2, smooth)

    aura=pd.DataFrame()
    aura['Wavelength']=waves*1e6
    aura['Depth'] = medians
    aura['e_Depth'] = down1
    aura['E_depth'] = up1
    aura['e_depth2'] = down2
    aura['E_depth2'] = up2
    np.savetxt('../../aura_model.txt', aura.values, fmt=['%.4f','%.6f','%.6f', '%.6f','%.6f', '%.6f'])
    aura.to_csv('../../aura_model.csv', index=False)
    return medians, up1, up2, down1, down2, waves

def get_atmo():
    atmo = readsav('./figure_pickle_files/H41.atmo.HR.sav')
    waves = atmo['ws']
    depths = atmo['model']*atmo['model']
    waves = waves[1:]
    depths = depths[1:]

    atmo=pd.DataFrame()
    atmo['Wavelength'] = waves[::-1]
    atmo['Depth'] = depths[::-1]
    np.savetxt('../../atmo.txt', atmo.values, fmt=['%.4f', '%.6f'])
    atmo.to_csv('../../atmo.csv', index=False)
    return waves, depths
    
def transit_model(params, transit_calc, fit_info,
                  gas=True, scat=True, CIA=True,
                  mie_only=False, ray_only=False, model_offset=False):

    params_dict = fit_info._interpret_param_array(params)
    
    Rp = params_dict["Rp"]
    T = params_dict["T"]
    logZ = params_dict["logZ"]
    CO_ratio = params_dict["CO_ratio"]
    scatt_factor = 10.0**params_dict["log_scatt_factor"]
    scatt_slope = params_dict["scatt_slope"]
    cloudtop_P = 10.0**params_dict["log_cloudtop_P"]
    error_multiple = params_dict["error_multiple"]
    Rs = params_dict["Rs"]
    Mp = params_dict["Mp"]
    T_star = params_dict["T_star"]
    T_spot = params_dict["T_spot"]
    spot_cov_frac = params_dict["spot_cov_frac"]
    frac_scale_height = 10**params_dict["frac_scale_height"]
    number_density = 10.0**params_dict["log_number_density"]
    part_size = 10.0**params_dict["log_part_size"]
    ri = params_dict["ri"]
    

    """
    adding cloud fraction
    """
    cloud_fraction = params_dict["cloud_fraction"]
    if cloud_fraction < 0. or cloud_fraction > 1.:
        return -np.inf
    
    """ Add offset """
    try:
        offset=params_dict["offset"]
    except KeyError:
        offset=0
    try:
        offset2=params_dict["offset2"]
    except KeyError:
        offset2=0
    try:
        offset3=params_dict["offset3"]
    except KeyError:
        offset3=0
    try:
        wfc3_offset=params_dict["wfc3_offset"]
    except KeyError:
        wfc3_offset=0
    try:
        stis_offset=params_dict["stis_offset"]
    except KeyError:
        stis_offset=0
    try:
        stis1_offset=params_dict["stis1_offset"]
    except KeyError:
        stis1_offset=0
    try:
        stis2_offset=params_dict["stis2_offset"]
    except KeyError:
        stis2_offset=0
    try:
        spitzer_offset=params_dict["spitzer_offset"]
    except KeyError:
        spitzer_offset=0 
        
    ln_likelihood = 0
    transit_wavelengths, calculated_transit_depths, info_dict = transit_calc.compute_depths(
        Rs, Mp, Rp, T, logZ, CO_ratio,
        scattering_factor=scatt_factor, scattering_slope=scatt_slope,
        cloudtop_pressure=cloudtop_P, T_star=T_star,
        T_spot=T_spot, spot_cov_frac=spot_cov_frac,
        frac_scale_height=frac_scale_height, number_density=number_density,
        part_size=part_size, ri=ri, add_scattering=scat, add_gas_absorption=gas,
        add_collisional_absorption=CIA,full_output=True,
        mie_only=mie_only, ray_only=ray_only)
    """
    computing clear depth
    """
    if cloud_fraction != 1:
        xxx, clear_calculated_transit_depths, clear_info_dict = transit_calc.compute_depths(
            Rs, Mp, Rp, T, logZ, CO_ratio,
            scattering_factor=scatt_factor, scattering_slope=scatt_slope,
            cloudtop_pressure=10**8, T_star=T_star,
            T_spot=T_spot, spot_cov_frac=spot_cov_frac,
            frac_scale_height=frac_scale_height, number_density=number_density,
            part_size=part_size, ri=None, add_scattering=scat, add_gas_absorption=gas,
            add_collisional_absorption=CIA, full_output=True, mie_only=mie_only,
            ray_only=ray_only)
                
        calculated_transit_depths = cloud_fraction * calculated_transit_depths + (1.-cloud_fraction) * clear_calculated_transit_depths
        info_dict['unbinned_depths']=cloud_fraction*info_dict['unbinned_depths']+(1.-cloud_fraction) * clear_info_dict['unbinned_depths']
                                                               

    wfc3_range=np.where((transit_wavelengths>1.12e-6) & (transit_wavelengths<1.66e-6))
    stis_range=np.where(transit_wavelengths<9.3e-7)
    spitzer_range = np.where((transit_wavelengths>3.2e-6) & (transit_wavelengths<5.0e-6))
    stis_range1=np.where(transit_wavelengths<5.69e-7)
    stis_range2=np.where((transit_wavelengths> 5.263e-7) & (transit_wavelengths<9.3e-7))
    
    index_sb = 29
    index_sr = 46
    index_w = 75
    
    #Tsiaras wfc3 index (less bins)
    #index_w=71
    
    #wfc3_range=np.arange(index_sr,index_w)
    #stis_range1=np.arange(index_sb)
    #stis_range2=np.arange(index_sb,index_sr)
    
    #stis_range1=np.arange(29)
    #stis_range2=np.arange(29,46)

    if model_offset == True:
        calculated_transit_depths[wfc3_range]+=offset*1.0e-5
        # sometimes offset2 is stis, sometimes it's stis blue
        calculated_transit_depths[stis_range]+=offset2*1.0e-5
        #calculated_transit_depths[stis_range1]+=offset2*1.0e-5
        # sometimes offset 3 is spitzer, sometimes it's stis red
        #calculated_transit_depths[spitzer_range]+=offset3*1.0e-5
        calculated_transit_depths[stis_range2]+=offset3*1.0e-5

        calculated_transit_depths[wfc3_range] += wfc3_offset*1.0e-5
        calculated_transit_depths[stis_range] += stis_offset*1.0e-5 
        calculated_transit_depths[spitzer_range]+=spitzer_offset*1.0e-5
        calculated_transit_depths[stis_range1]+=stis1_offset*1.0e-5
        calculated_transit_depths[stis_range2]+=stis2_offset*1.0e-5
    
        wfc3_range=np.where((info_dict['unbinned_wavelengths']>1.12e-6) & (info_dict['unbinned_wavelengths']<1.66e-6))
        stis_range=np.where(info_dict['unbinned_wavelengths']<9.3e-7)
        spitzer_range = np.where((info_dict['unbinned_wavelengths']>3.2e-6) & (info_dict['unbinned_wavelengths']<5.0e-6))
        stis_range1=np.where(info_dict['unbinned_wavelengths']<5.69e-7)
        stis_range2=np.where((info_dict['unbinned_wavelengths']> 5.263e-7) & (info_dict['unbinned_wavelengths']<9.3e-7))

        info_dict['unbinned_depths'][wfc3_range] += wfc3_offset*1.0e-5
        #info_dict['unbinned_depths'][wfc3_range] += offset*1.0e-5
        info_dict['unbinned_depths'][stis_range] += stis_offset*1.0e-5 
        info_dict['unbinned_depths'][spitzer_range]+=spitzer_offset*1.0e-5
        info_dict['unbinned_depths'][stis_range1]+=stis1_offset*1.0e-5
        info_dict['unbinned_depths'][stis_range2]+=stis2_offset*1.0e-5
        #info_dict['unbinned_depths'][stis_range]+=offset2*1.0e-5
        #info_dict['unbinned_depths'][stis_range2]+=offset3*1.0e-5

    #residuals = calculated_transit_depths - measured_transit_depths
    #scaled_errors = error_multiple * measured_transit_errors
    #ln_likelihood += -0.5 * np.sum(residuals**2 / scaled_errors**2 + np.log(2 * np.pi * scaled_errors**2))

    # This code gets water feature size in scale heights
    #wfc3_range=np.where((transit_wavelengths>1.12e-6) & (transit_wavelengths<1.66e-6))
    mu = np.median(info_dict['mu_profile'])
    g = G * Mp / Rp / Rp
    #mu = 5.8
    scale = k_B * T / mu / AMU / g # meters
    #water = calculated_transit_depths[wfc3_range]**.5
    #feature = (water.max() - water.min())*Rs

    #print scale
    #print
    #print feature/scale

    return info_dict, calculated_transit_depths, transit_wavelengths


def get_priors(pkl):
    with open('bestfit/'+pkl+'.pkl') as f:
        _, fit_info = pickle.load(f)


    print(_['samples'].shape)
    columns=[]
    prior=[]
    low=[None]*len(fit_info.all_params.keys())
    high=[None]*len(fit_info.all_params.keys())
    std=[None]*len(fit_info.all_params.keys())
    for i, key in enumerate(fit_info.all_params):
        factor=1.
        columns.append(key)
        if key=='Mp':
            factor=1/M_jup
        if key=='Rp':
            factor=1/R_jup
        if key=='Rs':
            factor=1/R_sun
        try:
            prior.append(fit_info.all_params[key].best_guess*factor)
        except TypeError:
            prior.append(fit_info.all_params[key].best_guess)
        try:
            low[i]=fit_info.all_params[key].low_lim*factor
            high[i]=fit_info.all_params[key].high_lim*factor
        except AttributeError:
            try:
                std[i]=fit_info.all_params[key].std*factor
            except AttributeError:
                pass
    df=pd.DataFrame()
    df['low']=low
    df['prior']=prior
    df['high']=high
    df['std']=std
    df['Index']=columns
    df=df.set_index('Index')
    return df
    

def pretty_corner(pkl, pkl2=None):

    with open('bestfit/'+pkl+'.pkl') as f:
        result, fit_info = pickle.load(f)

    #if pkl2:
    #    with open('bestfit/'+pkl+'.pkl') as f:
    #        result2, fit_info2 = pickle.load(f)
    #equal_samples = nestle.resample_equal(result.samples, result.weights)
    #np.percentile(equal_samples[:, i], .2)
    #sss
    # change parameter names to desired ones for plot
    # change units so the numbers are readable
    # change format so that, ie, temp doesnt use 2 decimal places
    # allow for removing uninteresting columns for corner
    # fix subscripts
    labels=np.zeros_like(fit_info.fit_param_names).astype(object)
    include=(np.ones_like(fit_info.fit_param_names).astype(bool))
 
    for i, name in enumerate(fit_info.fit_param_names):
        if name=='logZ':
            #result.samples[:,i]= 10.0**result.samples[:,i]
            #labels[i]=r'$\log{\frac{Z}{Z_{\odot}}}$ '
            labels[i]=r'$\log{Z/Z_{\odot}}$ '
        if name=='CO_ratio':
            labels[i]=r"$C/O$"
            #include[i]=False
        if name=='T':
            #labels[i]=r"$\frac{T_p}{1000K}$"
            labels[i]=r"$T_p/10^3K$"
            result.samples[:,i]=result.samples[:,i]/1000.
            #include[i]=False
        if name=='cloud_fraction':
            labels[i]=r"$f_c$"
            #include[i]=False
        if name=='log_cloudtop_P':
            labels[i]=r'$\log{P_{cloud}/Pa}$'
            #include[i]=False
        if name=='Rp':
            result.samples[:,i]=result.samples[:,i]/R_jup
            labels[i]=r"$R_p/R_{Jup}$"
            #include[i]=False
        if name=='scatt_slope':
            labels[i]=r"$\gamma$"
        if name=='log_scatt_factor':
            labels[i]=r"$a_0$"
        if name=='Rs':
             result.samples[:,i]=result.samples[:,i]/R_sun
             labels[i]=r"$R_s/R_{\odot}$"
             #include[i]=False
        if name=='Mp':
            result.samples[:,i]=result.samples[:,i]/M_jup
            labels[i]=r"$M_p/M_{Jup}$"
            #include[i]=False
        if name=='offset':
            labels[i]='WFC3 offset'
            #include[i]=False
            result.samples[:,i]=result.samples[:,i]*-10.
        if name=='offset2':
            labels[i]='STIS offset'
            #include[i]=False
            result.samples[:,i]=result.samples[:,i]*-10.
        if name=='offset3':
            labels[i]='G750 offset'
            #include[i]=False
            result.samples[:,i]=result.samples[:,i]*-10.
        if name=='wfc3_offset':
            labels[i]='WFC shift'
            #include[i]=False
            result.samples[:,i]=result.samples[:,i]*-10.
        if name=='stis1_offset':
            labels[i]='G430 shift'
            #include[i]=False
            result.samples[:,i]=result.samples[:,i]*-10.
        if name=='stis2_offset':
            labels[i]='G750 shift'
            #include[i]=False
            result.samples[:,i]=result.samples[:,i]*-10.
        if name=='stis_offset':
            labels[i]='STIS offset'
            #include[i]=False
            result.samples[:,i]=result.samples[:,i]*-10.
        if name == 'log_number_density':
            labels[i]=r"$\log{n}$"
            #include[i]=True
        if name == "log_part_size":
            labels[i]=r"$\log{r_{part}/m}$"
            include[i]=True
            #result.samples[:,i]= 10.0**result.samples[:,i]
        if name == 'frac_scale_height':
            labels[i]=r"$\log{H_{part}/H_{gas}}$"
            include[i]=True
        if name == 'error_multiple':
            labels[i]='Error Multiple'
            #include[i]=False
        if name=='spot_cov_frac':
            labels[i]=r"$f_{fac}$"
            #include[i]=False

    fig=corner.corner(result.samples.T[include,:].T, weights=result.weights,
                      range=[0.99] * np.sum(include),
                      labels=labels[include], show_titles=True
                      ,quantiles=[.16,.5,.84], title_kwargs={"fontsize": 12}
                      , smooth=.5, label_kwargs={"fontsize":16})

    # Should I smooth 2d hist? I can change cmap in the corner_hist2d code, but it isn't sequential.
    #print result.samples.shape
    #plt.savefig('../../mie_pc_corner.pdf')
    plt.show()
    # plt.clf()
    # plt.close()
    return 1


def breakdown(pkl, smooth=False):



    """with open('bestfit/fid_breakdown_noCO.pkl') as f:
       xx, co, x, plot_depths, full = pickle.load(f)
    with open('bestfit/fid_breakdown_noCO2.pkl') as f:
       xx, co2, x, plot_depths, full = pickle.load(f)
    with open('bestfit/fid_breakdown_noH2O.pkl') as f:
       xx, h2o, x, plot_depths, full = pickle.load(f)
    with open('bestfit/fid_breakdown_noTiO.pkl') as f:
       xx, tio, x, plot_depths, full = pickle.load(f)
    with open('bestfit/fid_breakdown_noVO.pkl') as f:
       xx, vo, x, plot_depths, full = pickle.load(f)
    with open('bestfit/fid_breakdown_noMgH.pkl') as f:
       xx, mgh, x, plot_depths, full= pickle.load(f)
    with open('bestfit/pc_breakdown_noNa.pkl') as f:
        xx, na, x, plot_depths, full = pickle.load(f)
    with open('bestfit/pc_breakdown.pkl') as f:
        xx, gas, x, plot_depths, full, xerr, errors = pickle.load(f)


    plt.errorbar(x*1e6, plot_depths,yerr=errors, xerr=xerr, fmt='.', color='b',
                 label="Observed", ecolor='grey', elinewidth=.4)
    #plt.errorbar(xx, h2o, color='b', label='h2o', linewidth=.8)
    #plt.errorbar(xx, co2, color='yellow', label='co2', linewidth=.8)
    #plt.errorbar(xx, co, color='red', label='co', linewidth=.8)
    #plt.errorbar(xx, mgh, color='green', label='MgH', linewidth=.8)
    #plt.errorbar(xx, vo, color='purple', label='vo', linewidth=.8)
    #plt.errorbar(xx, tio, color='red', label='tio', linewidth=.8)
    plt.errorbar(xx, na, color='orange', label='Na', linewidth=.8)
    plt.errorbar(xx,full, color='k', label='Best fit fiducial')
    plt.legend()
    plt.xlim([.3,5.4])
    plt.xscale('log')
    plt.show()
    sys.exit()"""
    

    
    with open('bestfit/'+pkl+'.pkl') as f:
        result, fit_info = pickle.load(f)

    print(result.samples.shape)
    getter = AbundanceGetter()
    bins, depths, errors = get_data()

    transit_calc = TransitDepthCalculator()
    binned_calc=TransitDepthCalculator()
    x=bins.mean(axis=1)
    binned_calc.change_wavelength_bins(bins)
    post=result.logp
    best_params_arr = result.samples[np.argmax(post)]

    hold=result.samples[np.argsort(post),:][-300:,:]

    #hold[:,3]=hold[:,3]/R_sun
    #hold[:,4]=hold[:,4]/M_jup
    #hold[:,5]=hold[:,5]/R_jup
    
    #hold[, -3] = 6
    
    #best_params_arr=[4.01, -6.44, 15.5, 1.86*R_sun, .86*M_jup, 1.76*R_jup, 1620, z, .38, 4.96]
    #best_params_arr=hold[-10]
    #best_params_arr[4]=0
    #best_params_arr[-1]=2.
    
    #best_params_arr=hold[hold[:,4].argmin()]
 
    print(best_params_arr)
    
    """metals=[1,2,2.5, 3.0]
    nsamples=len(metals)
    water = np.zeros((nsamples, 13))
    #vo = np.zeros((nsamples, 13))
    pressures = np.arange(-4, 9)
    pressures = pressures - 5
    for i in range(nsamples):
        abundances = getter.get(metals[i],best_params_arr[5])
        tt = int(round(best_params_arr[3], -2)/100 - 1)
        water[i] = abundances['H2O'][tt,:]
    #    #co[i] = abundances['Na'][tt,:]
    #    #co2[i] = abundances['CO2'][tt,:]
    #    tio[i] = abundances['TiO'][tt,:]
    #    vo[i] = abundances['VO'][tt,:]
        #best_params_arr[2]=1.
        #best_params_arr[0]=8.0
        #best_params_arr[-1]=.2
        plt.plot(water[i], pressures, alpha=.8, label=str(metals[i])+' tio')
    #    plt.plot(vo[i], pressures, alpha=.8, label=str(metals[i]) + ' vo', ls='--')
      
    plt.legend()
    plt.gca().invert_yaxis()
    plt.xscale('log')
    plt.ylabel('P [Bar]')
    #plt.yscale('log')
    plt.xlabel('log(Fractional Abundance)')
    #plt.xscale
    ##plt.xlim([-15,0])
    plt.show()
    #sys.exit()"""
    #best_params_arr[7]=0.0
    #solar_best_partial =  [ 1.26203452e+09,  1.70177384e+27,  1.22158796e+08,  1.40491907e+03, 0
    #                        ,2.13942750e-01, -6.66288297e-01 , 4.37473917e-01]
    #solar_best = [1.24013735e+09, 1.81937243e+27, 1.22208899e+08, 1.40426646e+03, 0.0,
    #              2.48421314e-01, 2.88498407e+00, 1.0]


    #sss
    #best_params_arr[1]=1.1*M_jup

    #abundances = getter.get(best_params_arr[3],best_params_arr[5])

    full_best, full, xx= transit_model(best_params_arr ,transit_calc, fit_info)
    full_best, gas, xx= transit_model(best_params_arr ,transit_calc, fit_info, CIA=False,scat=False)
    
    full_best, cia, xx= transit_model(best_params_arr ,transit_calc, fit_info, gas=False, scat=False)
    #best_params_arr[-1] = 0.0

    fullll, mie, xx= transit_model(best_params_arr ,transit_calc, fit_info, CIA=False, gas=False, mie_only=True)
    fullll, ray, xx= transit_model(best_params_arr ,transit_calc, fit_info, CIA=False, gas=False, ray_only=True)
    #best_params_arr[3]=1200
    #best_params_arr[4] = 0.0
    #best_params_arr[2] = best_params_arr[2]*.99
    #best_params_arr=np.array(solar_best_partial)
    #best_params_arr[-1] = 1.0
    cloudy_best, cloudy, xx= transit_model(best_params_arr ,transit_calc, fit_info)
    full_best, scat, xx= transit_model(best_params_arr ,transit_calc, fit_info, CIA=False, gas=False)
    #best_params_arr[-1] = 0.0
    #best_params_arr=np.array(solar_best)
    clear_best, clear, xx= transit_model(best_params_arr ,transit_calc, fit_info)
    full_best, ray, xx= transit_model(best_params_arr ,transit_calc, fit_info, CIA=False, gas=False)
    #best_params_arr[5]=.5
    #best_params_arr[3]=1700
    #best_params_arr[4]=2.1
    #full_best, full2, xx= transit_model(best_params_arr ,transit_calc, fit_info)

    #best_params_arr[3]=None
    #best_params_arr[5]=None
    #no_x, gas2, xx= transit_model(best_params_arr ,transit_calc, fit_info, CIA=False,scat=False, custom_abundances = abundances)

    if smooth:
        nbins=full.shape[0]
        gas=gf(gas,smooth)
        scat=gf(scat, smooth)
        cia=gf(cia, smooth)
        full=gf(full, smooth)
        #full2=gf(full2, smooth)
        ray=gf(ray, smooth)
        clear=gf(clear, smooth)
        cloudy=gf(cloudy, smooth)
    else:
        nbins=100
        bins1=np.logspace(np.log10(3.001e-7),np.log10(5e-6), nbins+1)
        bins_model=np.stack((bins1[:-1],bins1[1:])).T
        transit_calc.change_wavelength_bins(bins_model)

    xx=xx*1e6
    xerr=np.zeros(x.shape[0])
    xerr[-2]=.4
    xerr[-1]=.5
    scale=1e2

    params_dict = fit_info._interpret_param_array(best_params_arr)
    try:
        offset=params_dict["offset"]
    except KeyError:
        offset=0
    try:
        offset2=params_dict["offset2"]
    except KeyError:
        offset2=0
    try:
        offset3=params_dict["offset3"]
    except KeyError:
        offset3=0
    try:
        wfc3_offset=params_dict["wfc3_offset"]
    except KeyError:
        wfc3_offset=0
    try:
        stis_offset=params_dict["stis_offset"]
    except KeyError:
        stis_offset=0
    try:
        stis1_offset=params_dict["stis1_offset"]
    except KeyError:
        stis1_offset=0
    try:
        stis2_offset=params_dict["stis2_offset"]
    except KeyError:
        stis2_offset=0
    try:
        spitzer_offset=params_dict["spitzer_offset"]
    except KeyError:
        spitzer_offset=0 
        
    wfc3_range=np.where((x>1.11e-6) & (x<1.7e-6))
    stis_range=np.where(x<9.4e-7)
    #stis_range=np.where(x<5.69e-7)
    spitzer_range = np.where((x>3.2e-6) & (x<5.0e-6))
    stis_range1=np.arange(29)
    stis_range2=np.arange(29,46)
    plot_depths = depths.copy()
    plot_depths[wfc3_range] -= offset*1.0e-5
    # sometimes offset2 is stis, sometimes it's stis blue
    plot_depths[stis_range] -= offset2*1.0e-5
    # sometimes offset 3 is spitzer, sometimes it's stis red
    plot_depths[spitzer_range] -= offset3*1.0e-5

    #stis_offset=0
    plot_depths[wfc3_range] -= wfc3_offset*1.0e-5
    plot_depths[stis_range] -= stis_offset*1.0e-5 
    plot_depths[spitzer_range] -= spitzer_offset*1.0e-5
    plot_depths[stis_range1] -= stis1_offset*1.0e-5
    plot_depths[stis_range2] -= stis2_offset*1.0e-5

    plt.errorbar(x*1e6, plot_depths*scale,yerr=errors*scale, xerr=xerr, fmt='.', color='b',
                 label="Observed", ecolor='b', elinewidth=.4)
    plt.errorbar(xx, cia*scale, color='red', label='Cloudy Scattering', linewidth=.8)
    plt.errorbar(xx, ray*scale, color='red', label='Rayleigh Scattering', linewidth=.8
                 , ls='dashed', alpha=.8)
    plt.errorbar(xx, gas*scale, color='green', label='Gas Absorption', linewidth=.8)
    #plt.errorbar(xx,cia*scale, color='orange', label='CIA', linewidth=.8)
    plt.errorbar(xx,full*scale, color='k', label='Best fit fiducial')
    #plt.errorbar(xx,full2*scale, color='pink', label='high c/o')
    #plt.errorbar(xx,clear*scale, color='green', label='Clear Spectrum')
    #plt.errorbar(xx,cloudy*scale, color='g', label='Partial Clouds Forced Solar Z')
    #plt.errorbar(xx, mie*scale, color='red', label='Mie Scattering', linewidth=.8
    #             , ls='dotted', alpha=.8)
    plt.xscale('log')
    #plt.plot([3e-7, 1.7e-6],np.zeros(2))
    plt.xlim([.3,5])
    #plt.ylim(bottom=.86)
    plt.legend(loc='lower right',prop={'size': 8})
    plt.xlabel('Wavelength [$\mu$m]')
    plt.ylabel('Transit depth [%]')
    # plt.savefig('../../breakdown.png')
    # plt.clf()

    #with open('./bestfit/pc_breakdown_noNa.pkl', 'w') as f:
    #    pickle.dump([xx, gas*scale, x, plot_depths*scale, full*scale], f)
    
    plt.show()

def plot_spec(x, measured_transit_depths, measured_transit_errors, waves, medians
              , up1, up2, down1, down2, binned_medians,name, range=[.29, .54], yrange=[]
              ,log='log', size=(8,6), xerr=[]):
    if len(xerr)==0:
        xerr=np.zeros(x.shape[0])
        xerr[-2]=.4
        xerr[-1]=.5
        xerr[0]=.025
        xerr[1]=.01
        xerr[2]=.0085
        xerr[3]=.0085
    #from matplotlib.ticker import ScalarFormatter
    scale=100
    fig, ax = plt.subplots(1, figsize=size)
    
    #ax.get_xaxis().get_major_formatter().labelOnlyBase = False
    #ax.minorticks_off()

    wfc3_range=np.where((x>1.11e-6) & (x<1.7e-6))
    stis_range=np.where(x<9.4e-7)
    spitzer_range = np.where((x>3.2e-6) & (x<5.0e-6))
    stis_range1=np.arange(29)
    stis_range2=np.arange(29,46)
    plot_depths = measured_transit_depths.copy()
    #wfc3_offset=27.0
    #stis_offset=17.2
    #plot_depths[wfc3_range] -= wfc3_offset*1.0e-5
    #plot_depths[stis_range] -= stis_offset*1.0e-5 
    #plot_depths[spitzer_range] -= spitzer_offset*1.0e-5
    #plot_depths[stis_range1] -= stis1_offset*1.0e-5
    #plot_depths[stis_range2] -= stis2_offset*1.0e-5
    # for i, ax in enumerate(axis):
    ax.errorbar(METRES_TO_UM * x, plot_depths*scale, yerr = measured_transit_errors*scale
                , xerr=METRES_TO_UM*xerr, fmt='.', color='b', label="Observed", ecolor='b', elinewidth=.4, markersize=3)
    #plt.plot(METRES_TO_UM * x, binned_median, 'g^', label="Binned median model", ls='', alpha=1.0, markersize=5)
    # ax.plot(METRES_TO_UM * x, binned_best*scale, 'y^', label="Binned best-fit model", ls='', alpha=1.0, markersize=5)
    
    #plt.plot(METRES_TO_UM * full_best["unbinned_wavelengths"], full_best["unbinned_depths"], alpha=0.5, color='r', label="Best")
    #plt.xlim(.29,1.7)
  
    ax.fill_between(waves*1e6, medians*scale, up1*scale, color='r', alpha=.4)
    ax.fill_between(waves*1e6, up1*scale, up2*scale, color='r', alpha=.15)
    ax.fill_between(waves*1e6, down1*scale, medians*scale, color='r', alpha=.4)
    ax.fill_between(waves*1e6, down2*scale,down1*scale, color='r', alpha=.15)
    ax.plot(METRES_TO_UM * waves, medians*scale, color='r', label='PLATON')
    
    ax.set_xlabel("Wavelength [$\mu m$]")
    ax.set_ylabel("Transit depth [%]")
    ax.set_xscale(log)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks([0.3, 0.4,0.5, .7,.9, 1,1.5,2,3,4,5,7,9])
    ax.set_xlim([.29, 5.4])
    #ax.set_xlim([.29, 10])

    #with open('figure_pickle_files/fid_spectrum.pkl') as f:
    #    waves, medians, up1, up2, down1, down2, x, bb, xerr, plot_depths, measured_transit_errors = pickle.load(f)
    """
    medians, up1, up2, down1, down2, waves = get_aura()
    ax.plot(METRES_TO_UM*waves, medians*scale, color='purple', label='AURA', alpha=.5)
    ax.fill_between(waves*1e6, medians*scale, up1*scale, color='purple', alpha=.4)
    ax.fill_between(waves*1e6, up1*scale, up2*scale, color='purple', alpha=.15)
    ax.fill_between(waves*1e6, down1*scale, medians*scale, color='purple', alpha=.4)
    ax.fill_between(waves*1e6, down2*scale,down1*scale, color='purple', alpha=.15)
    
    waves2, medians2 = get_atmo()
    
    ax.plot(METRES_TO_UM * waves2, medians2*scale, color='g', label='ATMO', alpha=.5)"""
    """
    
    with open('figure_pickle_files/best_spectrum.pkl') as f:
        waves, medians, up1, up2, down1, down2, x, bb, xerr, plot_depths, measured_transit_errors = pickle.load(f)

    ax.plot(METRES_TO_UM * waves, medians*scale, color='g', label='Offset Model', alpha=.5)
    ax.fill_between(waves*1e6, medians*scale, up1*scale, color='g', alpha=.4)
    ax.fill_between(waves*1e6, up1*scale, up2*scale, color='g', alpha=.15)
    ax.fill_between(waves*1e6, down1*scale, medians*scale, color='g', alpha=.4)
    ax.fill_between(waves*1e6, down2*scale,down1*scale, color='g', alpha=.15)"""
    #ax.plot(METRES_TO_UM * x, binned_medians*scale, 'g^', label="Binned Median Model", ls='', alpha=.8, markersize=4)

    
    if range[0]==1.1:
        ax.get_xaxis().get_major_formatter().labelOnlyBase = False
        ax.set_xticks([1.1,1.2,1.3,1.4,1.5,1.6,1.7])
        ax.set_xscale('linear')
    if range[1]==1.0:
        ax.get_xaxis().get_major_formatter().labelOnlyBase = False
        #ax.set_xticks([0.])
        ax.set_xscale('linear')
    # fig.tight_layout()

    
    if range[1]==5.4:
        ax.legend(loc='lower right')
    #ax.set_xlim(range)
    #ax.set_ylim(yrange)
    plt.show()
    #plt.savefig('./figure_pickle_files/model_comp_15.pdf')

    
    #plt.show()
    sss
    return 1

def pretty_plot(pkl, smooth=False, ax=None):
    retriever = Retriever()
    getter = AbundanceGetter()
    transit_calc = TransitDepthCalculator()
    binned_calc=TransitDepthCalculator()
    with open('bestfit/'+pkl+'.pkl') as f:
        result, fit_info = pickle.load(f)
    bins, measured_transit_depths, measured_transit_errors = get_data()

    x=bins.mean(axis=1)
    xerr=x-bins[:,0]
    binned_calc.change_wavelength_bins(bins)
    
    post = result.logp

    # parameter set that gives highest posterior
    best_params_arr = result.samples[np.argmax(post)]

    # print best_params_arr
    # sys.exit()

    # 1 and 2 sigma errors
    # sig1=result.samples[np.where(post-post.max()>-.5)]
    # sig2=result.samples[np.where((post-post.max()>-2.) & (post-post.max()<-.5))]
    # also possibly want median values of all params
    equal_samples = nestle.resample_equal(result.samples, result.weights)
    #medians = np.percentile(equal_samples, 50, axis=0)

    #_, binned_median, _=transit_model(medians ,binned_calc, fit_info)
    print('this is the right one')
    _, binned_best, _ = transit_model(best_params_arr ,binned_calc, fit_info, model_offset=True)
    #full_median, calc_median, xx=transit_model(medians ,transit_calc, fit_info)
    full_best, calc_best, xx= transit_model(best_params_arr ,transit_calc, fit_info)

    
    # Calculate chi-squared value for best fit
    dof = len(bins) - fit_info._get_num_fit_params()
    print(dof)
    print((measured_transit_depths - binned_best) / measured_transit_errors)
    chi2 = np.sum((measured_transit_depths - binned_best)**2 / measured_transit_errors**2)
    print("Chi squared = %.3f" % (chi2/dof))

    # Chan method

    if smooth:
        nbins=calc_best.shape[0]
    else:
        nbins=1000
        bins1=np.logspace(np.log10(3.001e-7),np.log10(10e-6), nbins+1)
        bins_model=np.stack((bins1[:-1],bins1[1:])).T
        transit_calc.change_wavelength_bins(bins_model)

    nsamples=50
    chi_i = np.zeros(nsamples)
    randoms=random.sample(equal_samples, nsamples)
    depths=np.zeros((nbins,nsamples))
    binned_depths=np.zeros((x.shape[0],nsamples))

    
    # Extras for mmw and abundance profile distributions

    mus = np.zeros(nsamples)
    presh = np.zeros((nsamples, 250))
    water = np.zeros((nsamples, 13))
    co = np.zeros((nsamples, 13))
    co2 = np.zeros((nsamples, 13))
    tio = np.zeros((nsamples, 13))
    na = np.zeros((nsamples, 13))   
    vo = np.zeros((nsamples, 13))
    hf = np.zeros((nsamples, 13))
    h2 = np.zeros((nsamples, 13))
    he = np.zeros((nsamples, 13))
    hcl = np.zeros((nsamples, 13))   
    sio = np.zeros((nsamples, 13))
    xh = np.zeros((nsamples, 13))
    o3 = np.zeros((nsamples, 13))
    n2 = np.zeros((nsamples, 13))
    h2s = np.zeros((nsamples, 13))   
    h2co = np.zeros((nsamples, 13))
    mgh = np.zeros((nsamples, 13))
    c2h2 = np.zeros((nsamples, 13))
    so2 = np.zeros((nsamples, 13))   
    ph3 = np.zeros((nsamples, 13))
    xk = np.zeros((nsamples, 13))
    xc = np.zeros((nsamples, 13))
    xn = np.zeros((nsamples, 13))
    xo = np.zeros((nsamples, 13))
    ocs = np.zeros((nsamples, 13))   
    o2 = np.zeros((nsamples, 13))
    no2 = np.zeros((nsamples, 13))
    c2h4 = np.zeros((nsamples, 13))
    no = np.zeros((nsamples, 13))
    ch4 = np.zeros((nsamples, 13))   
    sih = np.zeros((nsamples, 13))
    nh3 = np.zeros((nsamples, 13))
    hcn = np.zeros((nsamples, 13))   
    oh = np.zeros((nsamples, 13))

    
    for i, item in enumerate(randoms):
        info,d,waves=transit_model(item, transit_calc, fit_info, model_offset=False)
        _,dd,waves2=transit_model(item, binned_calc, fit_info, model_offset=False)
        _,ddd,www=transit_model(item, binned_calc, fit_info, model_offset=True)
        depths[:,i]=d
        binned_depths[:,i]=dd
        # MMW and abundances
        mus[i] = np.median(info['mu_profile'])
        # check for mie scattering
        if fit_info.all_params['ri'].best_guess != None:
            zz = item[7]
            coco = item[8]
            ttt = item[6]
        else:
            zz= item[4]
            # For solar abundance (i.e, no logZ)
            #zz=0.0
            #coco=item[4]
            coco=item[5]
            ttt=item[3]
        abundances = getter.get(zz, coco)
        tt = int(round(ttt, -2))/100 - 1
        water[i] = abundances['H2O'][tt,:]
        co[i] = abundances['CO'][tt,:]
        co2[i] = abundances['CO2'][tt,:]
        tio[i] = abundances['TiO'][tt,:]
        vo[i] = abundances['VO'][tt,:]
        na[i] = abundances['Na'][tt,:]
        h2[i] = abundances['H2'][tt,:]
        he[i] = abundances['He'][tt,:]
        hcl[i] = abundances['HCl'][tt,:]
        h2co[i] = abundances['H2CO'][tt,:]
        xh[i] = abundances['H'][tt,:]
        h2s[i] = abundances['H2S'][tt,:]
        xk[i] = abundances['K'][tt,:]
        sio[i] = abundances['SiO'][tt,:]
        mgh[i] = abundances['MgH'][tt,:]
        hf[i] = abundances['HF'][tt,:]
        o3[i] = abundances['O3'][tt,:]
        n2[i] = abundances['N2'][tt,:]
        ph3[i] = abundances['PH3'][tt,:]
        so2[i] = abundances['SO2'][tt,:]
        xn[i] = abundances['N'][tt,:]
        xo[i] = abundances['O'][tt,:]
        xc[i] = abundances['C'][tt,:]
        o2[i] = abundances['O2'][tt,:]
        no2[i] = abundances['NO2'][tt,:]
        no[i] = abundances['NO'][tt,:]
        ocs[i] = abundances['OCS'][tt,:]
        ch4[i] = abundances['CH4'][tt,:]
        sih[i] = abundances['SiH'][tt,:]
        nh3[i] = abundances['NH3'][tt,:]
        hcn[i] = abundances['HCN'][tt,:]
        c2h2[i] = abundances['C2H2'][tt,:]
        c2h4[i] = abundances['C2H4'][tt,:]
        oh[i] = abundances['OH'][tt,:]

        chi_i[i] = np.sum((measured_transit_depths - ddd)**2 / measured_transit_errors**2)
        print(i)


    print("Complexity = %.2f" % (chi_i.mean() - chi2))

    abund_samples = {}
    abund_samples['H2O'] = water
    abund_samples['CO'] = co
    abund_samples['CO2'] = co2
    abund_samples['TiO'] = tio
    abund_samples['VO'] = vo
    abund_samples['Na'] = na
    abund_samples['H2'] = h2
    abund_samples['He'] = he
    abund_samples['HCl'] = hcl
    abund_samples['H'] = xh
    abund_samples['H2CO'] = h2co
    abund_samples['H2S'] = h2s
    abund_samples['K'] = xk
    abund_samples['SiO'] = sio
    abund_samples['O3'] = o3
    abund_samples['MgH'] = mgh
    abund_samples['HF'] = hf
    abund_samples['N2'] = n2
    abund_samples['PH3'] = ph3
    abund_samples['SO2'] = so2
    
    abund_samples['C2H2'] = c2h2
    abund_samples['C2H4'] = c2h4
    abund_samples['OCS'] = ocs
    abund_samples['O2'] = o2
    abund_samples['N'] = xn
    abund_samples['C'] = xc
    abund_samples['O'] = xo
    abund_samples['NO'] = no
    abund_samples['NO2'] = no2
    abund_samples['HCN'] = hcn
    abund_samples['SiH'] = sih
    abund_samples['NH3'] = nh3
    abund_samples['OH'] = oh
    abund_samples['CH4'] = ch4
    
    scale = 1
    pressures = np.arange(-4, 9)
    pressures = pressures - 5
    aura_pressures = np.arange(-6,3)
 
    i = 0
    #colors = ['red', 'blue', 'grey', 'green']
    colors=['red']*34
    aura_colors=['blue']*34
    #fig, axes = plt.subplots(2,2, figsize=(7,6), sharey=True)
    #fig, ax = plt.figure(figsize=(7,5))
    
    plt.rc('xtick',labelsize=12)
    plt.rc('ytick',labelsize=12)
    
    med_labels=[]
    med_array=[]
    
    for key in sorted(abund_samples.keys()):
        fig, ax = plt.subplots(figsize=(8,8))
        #ax=axes[i%2,int(i>1)]
        #fig, ax = plt.subplots(figsize=(4.5,3))
        #ax=axes[i]
        c=colors[i]
        medians=np.log10(np.percentile(abund_samples[key], 50, axis=0))        
        up1=np.log10(np.percentile(abund_samples[key], 84, axis=0))
        #up2=np.log10(np.percentile(abund_samples[key], 97.7, axis=0))
        down1=np.log10(np.percentile(abund_samples[key], 16, axis=0))
        #down2=np.log10(np.percentile(abund_samples[key], 2.3, axis=0))
        ax.fill_betweenx(pressures, medians*scale, up1*scale, alpha=.3, color=c
                          , label='PLATON')#, hatch='|')
        #ax.fill_betweenx(pressures, up1*scale, up2*scale, alpha=.15, color=c)
        ax.fill_betweenx(pressures, down1*scale, medians*scale, alpha=.3, color=c)#, hatch='|')
        #ax.fill_betweenx(pressures, down2*scale, down1*scale, alpha=.15, color=c)
        ax.plot(medians, pressures, color='k')
     
        c=aura_colors[i]
        # Madhus abundances: Co2, Na, TiO, VO, H2O
        if key == 'H2O':
            ax.fill_betweenx(aura_pressures,[-1.91]*9,[-1.38]*9,color=c, alpha=.3)#, hatch='-')
            ax.fill_betweenx(aura_pressures,[-2.59]*9,[-1.91]*9,color=c, label='AURA', alpha=.3)#, hatch='-')
            ax.plot([-1.91]*9, aura_pressures, color='k', ls='--')
            label=r'$\log{X_{H_2O}}$'
        if key == 'VO':
            ax.fill_betweenx(aura_pressures,[-8.37]*9,[-7.28]*9,color=c, alpha=.3)#, hatch='-')
            ax.fill_betweenx(aura_pressures,[-10.18]*9,[-8.37]*9,color=c, label='AURA', alpha=.3)#, hatch='-')
            ax.plot([-8.37]*9, aura_pressures, color='k', ls='--')
            label=r'$\log{X_{VO}}$'
        if key == 'TiO':
            ax.fill_betweenx(aura_pressures,[-9.46]*9,[-7.99]*9,color=c, alpha=.3)#, hatch='-')
            ax.fill_betweenx(aura_pressures,[-11.04]*9,[-9.46]*9,color=c, label='AURA', alpha=.3)#, hatch='-')
            ax.plot([-9.46]*9, aura_pressures, color='k', ls='--')
            label=r'$\log{X_{TiO}}$'
        if key == 'Na':
            ax.fill_betweenx(aura_pressures,[-2.38]*9,[-1.57]*9,color=c, alpha=.3)#, hatch='-')
            ax.fill_betweenx(aura_pressures,[-3.71]*9,[-2.38]*9,color=c, label='AURA', alpha=.3)#, hatch='-')
            ax.plot([-2.38]*9, aura_pressures, color='k', ls='--')
            label=r'$\log{X_{Na}}$'
        if key == 'CO2':
            ax.fill_betweenx(aura_pressures,[-5.8]*9,[-3.52]*9,color=c, alpha=.3)#, hatch='-')
            ax.fill_betweenx(aura_pressures,[-9.72]*9,[-5.8]*9,color=c, label='AURA', alpha=.3)#, hatch='-')
            ax.plot([-5.8]*9, aura_pressures, color='k', ls='--')
            label=r'$\log{X_{CO_2}}$'
        if key == 'CO':
            ax.fill_betweenx(aura_pressures,[-2.67]*9,[-1.67]*9,color=c, alpha=.3)#, hatch='-')
            ax.fill_betweenx(aura_pressures,[-4.58]*9,[-2.67]*9,color=c, label='AURA', alpha=.3)#, hatch='-')
            ax.plot([-2.67]*9, aura_pressures, color='k', ls='--')
            label=r'$\log{X_{CO}}$'
        if key == 'He':
            label=r'$\log{X_{He}}$'
        if key == 'H2':
            label=r'$\log{X_{H_2}}$'
        
        label = key

 
        if np.any(10**medians > 0.01):
            med_array.append(10**medians)
            med_labels.append(label)
        i = i + 1

        ax.invert_yaxis()
        #ax.set_ylim([2,-6])

        fract = 10**medians[pressures==-3][0]*100
        if key=='H2O':
            ax.legend(loc='upper left', prop={'size': 22})
        #ax.legend()
        ax.set_ylabel('P [Bar]', fontsize=22)
        ax.set_xlabel(label, fontsize=22)
        ax.text(.05,.05, 'Fractional abundance = %.2f%%' % (fract),size=14, transform=ax.transAxes)
      
        #plt.savefig('./figure_pickle_files/solar_metal_abundances/'+key+'.pdf', bbox_inches='tight')
        #plt.show()
        plt.clf()
        plt.close()
        
        ##plt.xlim([-15,0])
    #fig.tight_layout()
    #axes[1,1].set_ylabel('')
    #axes[0,1].set_ylabel('')
    #axes[0,0].set_xlabel('')
    #axes[0,1].set_xlabel('')
    #axes[0,0].legend(loc='upper left')
    fig, ax = plt.subplots()
    med_array = np.array(med_array)
    cstack = iter(cm.hsv(np.linspace(0.05, .95, med_array.shape[0])))
    #hatches = np.zeros_like(med_array[:,0]).astype(str)
    #hatches[0::2] = '\\'
    #hatches[1::2] = '//'
    stacks = ax.stackplot(pressures, med_array, labels=med_labels, colors=list(cstack))

    ax.set_xlabel('log Pressure [Bar]')
    ax.set_ylabel('Fractional Abundance')
    ax.set_title('PLATON Preferred Model')
    ax.legend(loc='lower left', ncol=1)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper right')
    ax.margins(0,0)
    #plt.show()
    #plt.savefig('./figure_pickle_files/stack_fid.pdf', bbox_inches='tight')
    #sys.exit()
    plt.show()
    #fig.tight_layout()
    #plt.savefig('../../abundances2.pdf')
    #plt.show()
    #sys.exit()
    nbins=20
    plt.hist(np.log10(mus),density=True, bins=nbins)
    plt.xlabel(r'$\log{\frac{MMW}{[AMU]}}$)')
    plt.ylabel('PDF')
    plt.show()
    plt.hist(mus,density=True, bins=nbins)
    plt.xlabel(r'$\frac{MMW}{[AMU]})$')
    plt.ylabel('PDF')
    plt.show()
    al=.4
    #sys.exit()

   

    # Hist for single pressure level value
    # plt.hist(water, label='water', density=True, bins=nbins, alpha=al)
    # plt.hist(co, label='co', density=True, bins=nbins, alpha=al)
    # plt.hist(h2, label='ch4', density=True, bins=nbins, alpha=al)
    # plt.hist(co2, label='co2', density=True, bins=nbins, alpha=al)
    # plt.hist(tio, label='tio', density=True, bins=nbins, alpha=al)
    # plt.hist(vo, label='vo', density=True, bins=nbins, alpha=al)
    # plt.legend()
    # plt.show()
    

    if smooth:
        medians=gf(np.percentile(depths, 50, axis=1),smooth)
        #std=depths.std(axis=1)
        up1=gf(np.percentile(depths, 84, axis=1),smooth)
        up2=gf(np.percentile(depths, 97.7, axis=1),smooth)
        down1=gf(np.percentile(depths, 16, axis=1),smooth)
        down2=gf(np.percentile(depths, 2.3, axis=1),smooth)
    #std=depths.std(axis=1)
    else:
        medians=np.percentile(depths, 50, axis=1)
        up1=np.percentile(depths, 84, axis=1)
        up2=np.percentile(depths, 97.7, axis=1)
        down1=np.percentile(depths, 16, axis=1)
        down2=np.percentile(depths, 2.3, axis=1)
    binned_medians=np.percentile(binned_depths, 50, axis=1)
    #plt.plot(METRES_TO_UM * waves, medians, color='b', alpha=1.0)
    #plt.plot(METRES_TO_UM * waves, up1, color='b', alpha=.5)
    #plt.plot(METRES_TO_UM * waves, up2, color='b', alpha=.1)
    #plt.plot(METRES_TO_UM * waves, down1, color='b', alpha=.5)
    #plt.plot(METRES_TO_UM * waves, down2, color='b', alpha=.1)

    # plot bestfit/median and data
    # lower transparency, plot all 1 sigma
    # lower again, plot all 2 sigma
   
    #plt.figure(1)
    #for item in sig1:
    #    full_i, calc_i, xx=transit_model(item ,transit_calc, fit_info)
    #    calc_i=gf(calc_i, 15)
    #    plt.plot(METRES_TO_UM * xx, calc_i, color='grey', alpha=.08)
        #plt.plot(METRES_TO_UM * full_i["unbinned_wavelengths"], full_i["unbinned_depths"], alpha=0.05, color='blue')
    #for i, item in enumerate(sig2):
    #    print i
    #    full_i, calc_i,xx=transit_model(item ,transit_calc, fit_info)
    #    calc_i=gf(calc_i, 15)
        #      #plt.plot(METRES_TO_UM * full_i["unbinned_wavelengths"], full_i["unbinned_depths"], alpha=0.02, color='blue')
    #    plt.plot(METRES_TO_UM * xx, calc_i, color='grey', alpha=.005)
        #plt.plot(METRES_TO_UM * full_median["unbinned_wavelengths"], full_median["unbinned_depths"], color='r', label="Median")
    #calc_median=gf(calc_median,smooth)
    #plt.plot(METRES_TO_UM * xx, calc_median, color='r', label="Median model")
    #calc_best=gf(calc_best,smooth)
    #plt.plot(METRES_TO_UM * xx, calc_best, color='r', label="Best")

    params_dict = fit_info._interpret_param_array(best_params_arr)
    try:
        offset=params_dict["offset"]
    except KeyError:
        offset=0
    try:
        offset2=params_dict["offset2"]
    except KeyError:
        offset2=0
    try:
        offset3=params_dict["offset3"]
    except KeyError:
        offset3=0
    try:
        wfc3_offset=params_dict["wfc3_offset"]
    except KeyError:
        wfc3_offset=0
    try:
        stis_offset=params_dict["stis_offset"]
    except KeyError:
        stis_offset=0
    try:
        stis1_offset=params_dict["stis1_offset"]
    except KeyError:
        stis1_offset=0
    try:
        stis2_offset=params_dict["stis2_offset"]
    except KeyError:
        stis2_offset=0
    try:
        spitzer_offset=params_dict["spitzer_offset"]
    except KeyError:
        spitzer_offset=0 
        
    wfc3_range=np.where((x>1.11e-6) & (x<1.7e-6))
    stis_range=np.where(x<9.4e-7)
    spitzer_range = np.where((x>3.2e-6) & (x<5.0e-6))
    stis_range1=np.arange(29)
    stis_range2=np.arange(29,46)
    plot_depths = measured_transit_depths.copy()
    plot_depths[wfc3_range] -= offset*1.0e-5
    # sometimes offset2 is stis, sometimes it's stis blue
    plot_depths[stis_range1] -= offset2*1.0e-5
    # sometimes offset 3 is spitzer, sometimes it's stis red
    plot_depths[stis_range2] -= offset3*1.0e-5

    #stis_offset=0
    plot_depths[wfc3_range] -= wfc3_offset*1.0e-5
    plot_depths[stis_range] -= stis_offset*1.0e-5 
    plot_depths[spitzer_range] -= spitzer_offset*1.0e-5
    plot_depths[stis_range1] -= stis1_offset*1.0e-5
    plot_depths[stis_range2] -= stis2_offset*1.0e-5

    #plot_depths = measured_transit_depths
    ranges=[[.29, 5.4], [.29,1.0],[1.1,1.7]]#,[3.0,5.4]]
    yranges=[[.96,1.2],[.96, 1.1], [1.0,1.08]]
    scales = ['log', 'linear', 'linear']
    names= ['full', 'stis', 'wfc3']
    #fig.set_size_inches(10, 4.44)
    sizes=[(10,4.44),(5,4),(5,4)]
    
    #with open('./figure_pickle_files/mie_pc_spectrum.pkl', 'w') as f:
    #    pickle.dump([waves, medians, up1, up2, down1, down2, x, binned_medians, xerr, plot_depths, measured_transit_errors], f)
        
    for i, r in enumerate(ranges):
        plot_spec(x, plot_depths, measured_transit_errors, waves, medians,
                  up1, up2, down1, down2, binned_medians, range=r
                  , yrange=yranges[i], log=scales[i], size=sizes[i], name=names[i], xerr=xerr)
    """
    
    
        
    xerr=np.zeros(x.shape[0])
    xerr[0]=.025
    xerr[1]=.01
    #xerr[2]=.0085
    xerr[-2]=.4
    xerr[-1]=.5
    scale=100
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(7,5))
  
    # for i, ax in enumerate(axis):
    ax.errorbar(METRES_TO_UM * x, measured_transit_depths*scale, yerr = measured_transit_errors*scale
                , xerr=xerr, fmt='.', color='b', label="Observed", ecolor='b', elinewidth=.4)
        #plt.plot(METRES_TO_UM * x, binned_median, 'g^', label="Binned median model", ls='', alpha=1.0, markersize=5)
        # ax.plot(METRES_TO_UM * x, binned_best*scale, 'y^', label="Binned best-fit model", ls='', alpha=1.0, markersize=5)
   
        #plt.plot(METRES_TO_UM * full_best["unbinned_wavelengths"], full_best["unbinned_depths"], alpha=0.5, color='r', label="Best")
        #plt.xlim(.29,1.7)

    ax.fill_between(waves*1e6, medians*scale, up1*scale, color='r', alpha=.4)
    ax.fill_between(waves*1e6, up1*scale, up2*scale, color='r', alpha=.15)
    ax.fill_between(waves*1e6, down1*scale, medians*scale, color='r', alpha=.4)
    ax.fill_between(waves*1e6, down2*scale,down1*scale, color='r', alpha=.15)
    ax.plot(METRES_TO_UM * waves, medians*scale, color='r', label='Median Model')
    ax.plot(METRES_TO_UM * x, binned_medians*scale, 'g^', label="Binned Median Model", ls='', alpha=1.0, markersize=5)"""
    
    #ax.set_xlabel("Wavelength [$\mu m$]")
    #ax.set_ylabel("Transit depth [%]")
    #ax.set_xscale('log')
    # fig.tight_layout()
    #ax.legend(loc='lower right')

    #from matplotlib.ticker import ScalarFormatter
    #ax.xaxis.set_major_formatter(ScalarFormatter())
    #ax.get_xaxis().get_major_formatter().labelOnlyBase = False
    #ax.minorticks_off()
    #ax.set_xticks([0.3, 0.4,0.5,1,1.5,2,3,4,5])
    #ax.set_xlim([.29, 5.4])
    #ax.set_xlim([1.115,1.666])
    # axes[1].set_ylim([1.0,1.08])
    # axes[0].legend(loc='upper left',prop={'size': 8})
    # axes[0].text(.9, 1.08, 'STIS', color='k')
    # axes[1].text(1.6, 1.07, 'WFC3', color='k')
  
    #plt.savefig('../../spectra.png')
    #plt.show()
    return 1

    #plt.errorbar(x, (measured_transit_depths-binned_median)/measured_transit_depths
    #             , yerr=measured_transit_errors/measured_transit_depths, fmt='.')
    #plt.show()


    
def hpd(pkl, histtype='step', alpha=1, pkl_label='test', param='logZ', ax=None, color='b'):
    """ Get highest posterior density credible interval. Also get 
    median and quantiles, and the marginalized posterior histogram 
    for log metallicity. Save all to a pickle file for that model
    """
    with open('bestfit/'+pkl+'.pkl') as f:
        result, fit_info = pickle.load(f)

    labels=np.zeros_like(fit_info.fit_param_names).astype(object)
    for i, name in enumerate(fit_info.fit_param_names):
        if name=='logZ':
            labels[i]=r'$\log{Z/Z_{\odot}}$ '
        if name=='CO_ratio':
            labels[i]=r"$C/O$"
        if name=='T':
            labels[i]=r"$T_p/10^3K$"
            result.samples[:,i]=result.samples[:,i]/1000.
        if name=='cloud_fraction':
            labels[i]=r"$f_c$"
        if name=='log_cloudtop_P':
            labels[i]=r'$\log{P_{cloud}/Pa}$'
        if name=='Rp':
            result.samples[:,i]=result.samples[:,i]/R_jup
            labels[i]=r"$R_p/R_{Jup}$"
        if name=='scatt_slope':
            labels[i]=r"$\gamma$"
        if name=='log_scatt_factor':
            labels[i]=r"$a_0$"
        if name=='Rs':
             result.samples[:,i]=result.samples[:,i]/R_sun
             labels[i]=r"$R_s/R_{\odot}$"
        if name=='Mp':
            result.samples[:,i]=result.samples[:,i]/M_jup
            labels[i]=r"$M_p/M_{Jup}$"
        if name=='offset':
            labels[i]='WFC3 offset'
        if name=='offset2':
            labels[i]='STIS offset'
        if name == 'log_number_density':
            labels[i]=r"$\log{n}$"
        if name == "log_part_size":
            labels[i]=r"$\log{r_{part}/m}$"
        if name == 'frac_scale_height':
            labels[i]=r"$H_{part}/H_{gas}$"
        if name=='spot_cov_frac':
            labels[i]=r"$f_{fac}$"
        
    mass_frac=.68
    nbins=result.samples.shape[0]
    nbins=30
    #nbins=50
    
    dist = nestle.resample_equal(result.samples, result.weights)
    for i, s in enumerate(result.samples.T):
        name = fit_info.fit_param_names[i]
        label = labels[i]
        # plt.clf()
        # plt.close()
        if name == param:
            if ax is None:
                fig, ax = plt.subplots()
            xx, yy, _ = ax.hist(s, bins=nbins, weights=result.weights,
                                density=True, histtype=histtype,
                                label=pkl_label, alpha=alpha, linewidth=1.1, color=c)

            centers = (yy[1:]+yy[:-1])/2.
            best = centers[np.argmax(xx)]
            yrange=[0,np.max(xx)]
            #print centers
        test = dist[:, i]
        d = np.sort(test)
        n = len(d)
        n_samples = np.floor(mass_frac * n).astype(int)
        int_width = d[n_samples:] - d[:n-n_samples]
        min_int = np.argmin(int_width)
        low = d[min_int]
        high = d[min_int+n_samples]
        span = np.arange(low, high, nbins)
        q1=np.percentile(dist[:,i],50)
        q2=np.percentile(dist[:,i],16)
        q3=np.percentile(dist[:,i],84)
    
        if name == param:
             if ax is None:
                 fig, ax = plt.subplots()
             #plt.hist(d, bins=nbins, histtype='step', density=True)
             #plt.plot([best, best], yrange, ls='dashed', color='red')
             #plt.plot([low, low], yrange, ls='dashed', color='red')
             #plt.plot([high, high], yrange, ls='dashed', color='red', label='HPD')
             #plt.plot([q1, q1], yrange, ls='dashed', color='black')
             #plt.plot([q2, q2], yrange, ls='dashed', color='black')
             #plt.plot([q3, q3], yrange, ls='dashed', color='black' , label='Quantiles')
             #print "Mode %s: %.2f - %.2f - %.2f" % (name, low, (low+high)/2.0, high)
             #print "Mode %s: %.2f - %.2f - %.2f" % (name, low, best, high)
             #print "Median %s: %.2f - %.2f - %.2f" % (name, q2, q1, q3)
             # print "%s: %d -- %d" % (name, low*1000, high*1000)
             # print result.samples.shape[0]
             #ax.set_ylabel('PDF')
             #.set_xlabel(label)
             #ax.legend()
             #plt.show()
             #plt.close()
        # Save marginalized posterior metallicity as a pickle with quantiles, hpd ranges.

    return 1

if __name__ == '__main__':
    
    pkl=sys.argv[1]
    smooth=False
    
    if len(sys.argv)==3:
        smooth=float(sys.argv[2])
        
    # 'f_part_full_cloud_prior', ,'tspot6060'    
    #breakdown(pkl, smooth=smooth)
    #breakdown('oct_double_offset_pc', smooth=smooth)
    #plt.show()

    print(get_priors(pkl))
    #sys.exit()
    pretty_corner(pkl)
    sys.exit()
    #sys.exit()
    #plt.close()
    pretty_plot(pkl, smooth=smooth)
 
    sys.exit()
    #hpd(pkl)
    #sys.exit()
 
    #sys.exit()
    # fig, axes = plt.subplots(1)
    # fig = plt.figure()
    # axes = []
    #axes.append(fig.add_subplot(211))
    #axes.append(fig.add_subplot(212))
    #axes.append(fig.add_subplot(22))
    #axes=np.array(axes)
    # pretty_plot(pkl, smooth=smooth, ax=axes)
    # fig.tight_layout()
    # plt.show()
    # sys.exit()
    #pkl_list = ['fid_march', 'fid+pc+scatter_march', 'fid+pc+mie_march']
    pkl_list = ['oct_fid',  'july20_tess+pc_scatter', 'july20_tess+pc+triple_offset', 'july20_tess+pc+mie',  'oct_triple_gauss_offset+pc',]
    #label = ['Fiducial', 'F + PC + Parametric Scattering', 'F + PC + Gaussian Offsets',
    #         'F + PC + Uniform Offsets']
    label = ['Fiducial', 'F + PC + Parametric Scattering', 'F + PC + 3 Uniform Offsets', 'F + PC + Mie',  'F + PC + Gaussian Offsets']
    #letter=['(a)', '(b)', '(c)']
    params=['logZ', 'T', 'CO_ratio']
    colors = [None, None,  None, None, 'k']
    fig = plt.figure(figsize=((7,5)))
    axes=[]
    axes.append(fig.add_subplot(121))
    axes.append(fig.add_subplot(222))
    axes.append(fig.add_subplot(224))
    axes=np.array(axes)

    
    #fig, axes = plt.subplots(3, sharex=True, gridspec_kw={'hspace': 0}, figsize=(7,6))
    for j in range(len(params)):
        param=params[j]
        ax=axes[j]
        for i, item in enumerate(pkl_list):
            c = colors[i]
            if item=='oct_fid':
                typ='bar'
                alpha=.2
            else:
                typ='step'
                alpha=1
            hpd(item, histtype=typ, alpha=alpha,
                pkl_label=label[i], param=param, ax=ax, color=c)

        #pretty_plot(item, smooth=smooth, ax=axes[i])
        #axes[i].label_outer()
        #axes[i].text(2.5, 1.08, letter[i] + ' ' + label[i], color='k')
    #axes[0].legend(prop={'size': 8}, loc='upper left')
    ### for posterior fig
    #axes[1].set_ylabel('Transit depth [%]')
    axes[2].set_ylabel('PDF')
    axes[1].set_ylabel('PDF')
    axes[0].legend(loc='upper left', prop={'size': 8})
    axes[0].set_xlabel('$\log{Z/Z_{\odot}}$')
    #axT = axes[0].twinx()
    #locs = axes[0].get_xticks()
    #labels = axes[0].get_xticklabels()
    #axT.set_xticks(locs-.37)
    #axT.set_xticklabels(labels)
    
    
    axes[1].set_xlabel('T/$10^3$K')
    axes[2].set_xlabel('C/O')
    axes[1].set_yticks([],[])
    axes[2].set_yticks([],[])
    axes[0].set_ylabel('PDF')
    axes[2].set_xlim(right=1.1)
    axes[1].set_xlim(1.1,2.3)
    fig.tight_layout()
    ###
    #plt.show()
    # ax.set_xlabel('$\log{Z/Z_{\odot}}$')
    # axes[0].set_ylabel('')
    # axes[2].set_ylabel('')
    #plt.show()
    fig.savefig('../../model_post_oct.pdf')
    
    # ax.legend(loc='upper left')
    # ax.set_title('Marginalized Temperature Posteriors For Different Models')
    #plt.show()
    #print get_priors(pkl)
    sys.exit()
    #pretty_corner(pkl)
    #pretty_plot(pkl, smooth=smooth)

