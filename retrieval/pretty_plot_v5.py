from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
#import nestle
import dynesty
import random
import corner
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d as gf
import time
from matplotlib.ticker import ScalarFormatter

from platon.fit_info import FitInfo
from platon.constants import R_sun, R_jup, M_jup, METRES_TO_UM, G, AMU, k_B
from platon.transit_depth_calculator import TransitDepthCalculator
from platon.eclipse_depth_calculator import EclipseDepthCalculator
from platon.abundance_getter import AbundanceGetter
from platon.combined_retriever import CombinedRetriever as Retriever
from retrieve_v5 import get_data


def transit_model(params, transit_calc, fit_info,
                  gas=True, scat=True, CIA=True,
                  mie_only=False, ray_only=False):

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
    
    offset=params_dict["offset"]
    try:
        offset2=params_dict["offset2"]
    except KeyError:
        offset2=0
    try:
        offset3=params_dict["offset3"]
    except KeyError:
        offset3=0
        
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
    #stis_range1=np.where(transit_wavelengths<5.69e-7)
    #stis_range2=np.where((transit_wavelengths>5.69e-7) & (transit_wavelengths<9.3e-7))

    calculated_transit_depths[wfc3_range]+=offset*1.0e-5
    calculated_transit_depths[stis_range]+=offset2*1.0e-5
    calculated_transit_depths[spitzer_range]+=offset3*1.0e-5
    #calculated_transit_depths[stis_range1]+=offset2*1.0e-5
    #calculated_transit_depths[stis_range2]+=offset3*1.0e-5

    
    wfc3_range=np.where((info_dict['unbinned_wavelengths']>1.12e-6) & (info_dict['unbinned_wavelengths']<1.66e-6))
    stis_range=np.where(info_dict['unbinned_wavelengths']<9.3e-7)
    spitzer_range = np.where((info_dict['unbinned_wavelengths']>3.2e-6) & (info_dict['unbinned_wavelengths']<5.0e-6))
    #stis_range1=np.where(info_dict['unbinned_wavelengths']<5.69e-7)
    #stis_range2=np.where((info_dict['unbinned_wavelengths']>5.69e-7) & (info_dict['unbinned_wavelengths']<9.3e-7))

    info_dict['unbinned_depths'][wfc3_range]+=offset*1.0e-5
    info_dict['unbinned_depths'][stis_range]+=offset2*1.0e-5
    info_dict['unbinned_depths'][spitzer_range]+=offset3*1.0e-5
    #info_dict['unbinned_depths'][stis_range1]+=offset2*1.0e-5
    #info_dict['unbinned_depths'][stis_range2]+=offset3*1.0e-5
 
    
    #residuals = calculated_transit_depths - measured_transit_depths
    #scaled_errors = error_multiple * measured_transit_errors
    #ln_likelihood += -0.5 * np.sum(residuals**2 / scaled_errors**2 + np.log(2 * np.pi * scaled_errors**2))

    # This code gets water feature size in scale heights
    #wfc3_range=np.where((transit_wavelengths>1.12e-6) & (transit_wavelengths<1.66e-6))
    mu = np.median(info_dict['mu_profile'])
    g = G * Mp / Rp / Rp
    scale = k_B * T / mu / AMU / g # meters
    #water = calculated_transit_depths[wfc3_range]**.5
    #feature = (water.max() - water.min())*Rs
    print(scale)
    #print
    #print feature/scale

    return info_dict, calculated_transit_depths, transit_wavelengths


def get_priors(pkl):
    #with open('bestfit/'+pkl+'.pkl') as f:
    #    _, fit_info = pickle.load(f)
    
    # for drake's
    with open('bestfit/'+pkl+'.pkl', 'rb') as f:
        result = pickle.load(f)
    fit_info = result.fit_info

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

    #with open('bestfit/'+pkl+'.pkl') as f:
    #    result, fit_info = pickle.load(f)

    # for drake's (and maybe all of v5?
    with open('bestfit/'+pkl+'.pkl', 'rb') as f:
        result = pickle.load(f)
    fit_info = result.fit_info

    
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
            labels[i]=r"$T_p/K$"
            result.samples[:,i]=result.samples[:,i]
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
            result.samples[:,i]=result.samples[:,i]*10.
        if name=='offset2':
            labels[i]='STIS offset'
            #include[i]=False
            result.samples[:,i]=result.samples[:,i]*10.
        if name=='offset3':
            labels[i]='STIS red offset'
            #include[i]=False
            result.samples[:,i]=result.samples[:,i]*10.
        if name == 'log_number_density':
            labels[i]=r"$\log{n}$"
            #include[i]=True
        if name == "log_part_size":
            labels[i]=r"$\log{r_{part}/m}$"
            include[i]=True
            #result.samples[:,i]= 10.0**result.samples[:,i]
        if name == 'frac_scale_height':
            labels[i]=r"$H_{part}/H_{gas}$"
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
                      , smooth=0.0, label_kwargs={"fontsize":16}, color='magenta')

    # Should I smooth 2d hist? I can change cmap in the corner_hist2d code, but it isn't sequential.
    #print result.samples.shape
    plt.savefig('../../drake_corner.png')
    #plt.show()
    # plt.clf()
    # plt.close()
    return 1


def breakdown(pkl, smooth=False):
    with open('bestfit/'+pkl+'.pkl', 'rb') as f:
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

    hold=result.samples[np.argsort(post),:][-1000:,:]
    
    #hold[:,3]=hold[:,3]/R_sun
    #hold[:,4]=hold[:,4]/M_jup
    #hold[:,5]=hold[:,5]/R_jup
    
    #print hold[-1]
    
    #best_params_arr=[4.01, -6.44, 15.5, 1.86*R_sun, .86*M_jup, 1.76*R_jup, 1620, z, .38, 4.96]
    best_params_arr=hold[-1]
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
    best_params_arr[-1] = 1.0
    cloudy_best, cloudy, xx= transit_model(best_params_arr ,transit_calc, fit_info)
    full_best, scat, xx= transit_model(best_params_arr ,transit_calc, fit_info, CIA=False, gas=False)
    best_params_arr[-1] = 0.0
    #best_params_arr=np.array(solar_best)
    clear_best, clear, xx= transit_model(best_params_arr ,transit_calc, fit_info)
    full_best, ray, xx= transit_model(best_params_arr ,transit_calc, fit_info, CIA=False, gas=False)
    #best_params_arr[5]=.5
    #best_params_arr[3]=1700
    #best_params_arr[4]=2.1
    #full_best, full2, xx= transit_model(best_params_arr ,transit_calc, fit_info)
   

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

    plt.errorbar(x*1e6, depths*scale,yerr=errors*scale, xerr=xerr, fmt='.', color='b',
                 label="Observed", ecolor='b', elinewidth=.4)
    plt.errorbar(xx, scat*scale, color='red', label='Cloudy Scattering', linewidth=.8)
    #plt.errorbar(xx, ray*scale, color='red', label='Rayleigh Scattering', linewidth=.8
    #             , ls='dashed', alpha=.8)
    #plt.errorbar(xx, gas*scale, color='green', label='Gas Absorption', linewidth=.8)
    #plt.errorbar(xx,cia*scale, color='orange', label='CIA', linewidth=.8)
    plt.errorbar(xx,full*scale, color='k', label='Best fit fiducial')
    #plt.errorbar(xx,full2*scale, color='pink', label='high c/o')
    plt.errorbar(xx,clear*scale, color='green', label='Clear Spectrum')
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
    plt.show()

def plot_spec(x, measured_transit_depths, measured_transit_errors, waves, medians
              , up1, up2, down1, down2, binned_medians,name, range=[.29, .54], yrange=[]
              ,log='log', size=(6,4)):
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
    ax.plot(METRES_TO_UM * x, binned_medians*scale, 'g^', label="Binned Median Model", ls='', alpha=1.0, markersize=5)
    
    ax.set_xlabel("Wavelength [$\mu m$]")
    ax.set_ylabel("Transit depth [%]")
    ax.set_xscale(log)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks([0.3, 0.4,0.5,1,1.5,2,3,4,5])
    ax.set_xlim([.29, 5.4])
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
    ax.set_xlim(range)
    ax.set_ylim(yrange)
    plt.savefig('../../'+name+'.png')
    #plt.show()
   
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
    _, binned_best, _ = transit_model(best_params_arr ,binned_calc, fit_info)
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
        nbins=100
        bins1=np.logspace(np.log10(3.001e-7),np.log10(5e-6), nbins+1)
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
    ch4 = np.zeros((nsamples, 13))
    for i, item in enumerate(randoms):
        info,d,waves=transit_model(item, transit_calc, fit_info)
        _,dd,waves2=transit_model(item, binned_calc, fit_info)
        depths[:,i]=d
        binned_depths[:,i]=dd
        # MMW and abundances
        mus[i] = np.median(info['mu_profile'])
        abundances = getter.get(item[4],item[5])
        tt = int(round(item[3], -2))/100 - 1
        water[i] = abundances['H2O'][tt,:]
        #co[i] = abundances['Na'][tt,:]
        co2[i] = abundances['CO2'][tt,:]
        tio[i] = abundances['TiO'][tt,:]
        vo[i] = abundances['VO'][tt,:]
        na[i] = abundances['NH3'][tt,:]
        #ch4[i] = abundances['CH4'][tt,:]

        chi_i[i] = np.sum((measured_transit_depths - dd)**2 / measured_transit_errors**2)
        print(i)

    
    print("Complexity = %.2f" % (chi_i.mean() - chi2))
    
    abund_samples = {}
    abund_samples['H2O'] = water
    #abund_samples['CO'] = co
    #abund_samples['CH4'] = ch4
    #abund_samples['CO2'] = co2
    #print np.median(co)
    abund_samples['TiO'] = tio
    abund_samples['VO'] = vo
    abund_samples['NH3'] = na
    scale = 1
    pressures = np.arange(-4, 9)
    pressures = pressures - 5
    aura_pressures = np.arange(-6,3)
 
    i = 0
    #colors = ['red', 'blue', 'grey', 'green']
    colors=['red']*4
    aura_colors=['blue']*4
    fig, axes = plt.subplots(2,2, figsize=(7,6), sharey=True)
    for key in sorted(abund_samples.keys()):
        ax=axes[i%2,int(i>1)]
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
            ax.fill_betweenx(aura_pressures,[-1.85]*9,[-1.46]*9,color=c, alpha=.3)#, hatch='-')
            ax.fill_betweenx(aura_pressures,[-2.28]*9,[-1.85]*9,color=c, label='AURA', alpha=.3)#, hatch='-')
            ax.plot([-1.85]*9, aura_pressures, color='k', ls='--')
            label=r'$\log{X_{H_2O}}$'
        if key == 'VO':
            ax.fill_betweenx(aura_pressures,[-8.34]*9,[-7.34]*9,color=c, alpha=.3)#, hatch='-')
            ax.fill_betweenx(aura_pressures,[-10.04]*9,[-8.34]*9,color=c, label='AURA', alpha=.3)#, hatch='-')
            ax.plot([-8.34]*9, aura_pressures, color='k', ls='--')
            label=r'$\log{X_{VO}}$'
        if key == 'TiO':
            ax.fill_betweenx(aura_pressures,[-9.42]*9,[-8.1]*9,color=c, alpha=.3)#, hatch='-')
            ax.fill_betweenx(aura_pressures,[-11.0]*9,[-9.42]*9,color=c, label='AURA', alpha=.3)#, hatch='-')
            ax.plot([-9.42]*9, aura_pressures, color='k', ls='--')
            label=r'$\log{X_{TiO}}$'
        if key == 'Na':
            ax.fill_betweenx(aura_pressures,[-2.57]*9,[-1.77]*9,color=c, alpha=.3)#, hatch='-')
            ax.fill_betweenx(aura_pressures,[-3.79]*9,[-2.57]*9,color=c, label='AURA', alpha=.3)#, hatch='-')
            ax.plot([-2.57]*9, aura_pressures, color='k', ls='--')
            label=r'$\log{X_{NH_3}}$'
        if key == 'CO2':
            ax.fill_betweenx(aura_pressures,[-3.94]*9,[-2.58]*9,color=c, alpha=.3)#, hatch='-')
            ax.fill_betweenx(aura_pressures,[-7.41]*9,[-3.94]*9,color=c, label='AURA', alpha=.3)#, hatch='-')
            ax.plot([-3.94]*9, aura_pressures, color='k', ls='--')
            label=r'$\log{X_{CO_2}}$'
        i = i + 1

        #ax.set_ylim([2,-6])
        if i==1:
            ax.invert_yaxis()
        #ax.legend()
        ax.set_ylabel('P [Bar]')
        ax.set_xlabel(label)
      
        #plt.savefig('../../abundances'+key+'.pdf')
        #plt.show()
        ##plt.xlim([-15,0])
    #fig.tight_layout()
    axes[1,1].set_ylabel('')
    axes[0,1].set_ylabel('')
    #axes[0,0].set_xlabel('')
    #axes[0,1].set_xlabel('')
    axes[0,0].legend(loc='upper left')
    
    fig.tight_layout()
    #plt.savefig('../../abundances2.pdf')
    plt.show()
    sys.exit()
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
   
    plt.figure(1)
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

    
    ranges=[[.29, 5.4], [.29,1.0],[1.1,1.7]]#,[3.0,5.4]]
    yranges=[[.96,1.2],[.96, 1.1], [1.0,1.08]]
    scales = ['log', 'linear', 'linear']
    names= ['full', 'stis', 'wfc3']
    sizes=[(9,5),(5,4),(5,4)]
    for i, r in enumerate(ranges):
        plot_spec(x, measured_transit_depths, measured_transit_errors, waves, medians,
                  up1, up2, down1, down2, binned_medians, range=r
                  , yrange=yranges[i], log=scales[i], size=sizes[i], name=names[i])
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


    
def hpd(pkl, histtype='step', alpha=1, pkl_label='test', param='logZ', ax=None):
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
                                label=pkl_label, alpha=alpha, linewidth=2)
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
    print(get_priors(pkl))
    pretty_corner(pkl)
   
    #plt.close()
    #pretty_plot(pkl, smooth=smooth)
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
    pkl_list = ['fid_march', 'fid+pc+scatter_march', 'fid+pc+mie_march']
    label = ['Fiducial', 'F + PC + Parametric Scattering', 'F + PC + Gaussian Offsets', 'F + PC + Uniform Offsets']
    label = ['Fiducial', 'F + PC + Parametric Scattering', 'F + PC + Mie']
    letter=['(a)', '(b)', '(c)']
    params=['logZ', 'T', 'CO_ratio']
    fig = plt.figure(figsize=((7,5)))
    axes=[]
    axes.append(fig.add_subplot(121))
    axes.append(fig.add_subplot(222))
    axes.append(fig.add_subplot(224))
    axes=np.array(axes)

    
    fig, axes = plt.subplots(3, sharex=True, gridspec_kw={'hspace': 0}, figsize=(7,6))
    #for j in range(len(params)):
    #    param=params[j]
    #    ax=axes[j]
    for i, item in enumerate(pkl_list):
        #    if item=='fid_march':
        #        typ='bar'
        #        alpha=.2
        #    else:
        #        typ='step'
        #        alpha=1
        #    hpd(item, histtype=typ, alpha=alpha,
        #        pkl_label=label[i], param=param, ax=ax)

        pretty_plot(item, smooth=smooth, ax=axes[i])
        axes[i].label_outer()
        axes[i].text(2.5, 1.08, letter[i] + ' ' + label[i], color='k')
    axes[0].legend(prop={'size': 8}, loc='upper left')
    ### for posterior fig
    axes[1].set_ylabel('Transit depth [%]')
    """axes[2].set_ylabel('PDF')
    axes[1].set_ylabel('PDF')
    axes[0].legend(loc='upper left', prop={'size': 8})
    axes[0].set_xlabel('$\log{Z/Z_{\odot}}$')
    axes[1].set_xlabel('T/$10^3$K')
    axes[2].set_xlabel('C/O')
    axes[1].set_yticks([],[])
    axes[2].set_yticks([],[])
    axes[0].set_ylabel('PDF')
    axes[2].set_xlim(right=1.1)
    axes[1].set_xlim(1.1,2.3)"""
    fig.tight_layout()
    ###
    plt.show()
    # ax.set_xlabel('$\log{Z/Z_{\odot}}$')
    # axes[0].set_ylabel('')
    # axes[2].set_ylabel('')
    #plt.show()
    fig.savefig('../../model_post_march.png')
    
    # ax.legend(loc='upper left')
    # ax.set_title('Marginalized Temperature Posteriors For Different Models')
    #plt.show()
    #print get_priors(pkl)
    sys.exit()
    #pretty_corner(pkl)
    #pretty_plot(pkl, smooth=smooth)

