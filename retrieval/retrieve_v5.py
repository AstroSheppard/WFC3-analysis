from __future__ import print_function
import sys

import numpy as np
from scipy.io import readsav
import pandas as pd
import matplotlib.pyplot as plt
import corner
#import nestle
import pickle

from platon.fit_info import FitInfo
# from platon.retriever import Retriever
from platon.constants import R_sun, R_jup, M_jup, AMU, k_B, G
from platon.transit_depth_calculator import TransitDepthCalculator
from platon.abundance_getter import AbundanceGetter
# from platon.combined_retriever import CombinedRetriever
from platon.TP_profile import Profile
from platon.eclipse_depth_calculator import EclipseDepthCalculator
from platon.combined_retriever import CombinedRetriever as Retriever
from platon.retrieval_result import RetrievalResult
import copy


# def get_best_fit(pkl):
# Would have to interpret fit_info._interpret_param_array first,
# too much effort for now
#    with open('bestfit/'+pkl+'.pkl') as f:
#        result, fit_info = pickle.load(f)
#    post=result.logp
#    best_params_arr = result.samples[np.argmax(post)]
#    return best_params_arr


def normalise_abundances(abundances, molecule):
    """ Takes abundance dictionary and a molecule.
    Removes that molecule, then renormalizes
    so abundances still add to 1 for every pressure
    and temperature point. Returns normalized
    abundances dict with molecule set to zero.
    """
    ab = copy.deepcopy(abundances)
    a1 = np.sum([abundances[item] for item in abundances], axis=0)
    # ab[molecule] *= 1e-99
    a2 = np.sum([ab[item] for item in ab], axis=0)
    factor = a1 / a2
    for item in ab:
        ab[item] *= factor
    return ab


def get_refractive_index(mol, wmin=.3, wmax=5.0):
    headers = ['Wave', 'n', 'k']
    data = pd.DataFrame(np.genfromtxt('optical_constants_mie/'+mol+'.dat'),
                        columns=headers)
    averages = data[(data['Wave'] > wmin) &
                    (data['Wave'] < wmax)].median(axis=0)
    n = averages['n']
    k = averages['k']
    return complex(n, -k)


def get_data(wfc3=True, stis=True, spitz=True, eclipse=False):
    """For each instrument, return 3 arrays:
    bins: 2d array with starting and ending wavelength of each bin
    depths: depth at each bin
    errors: errors at each bin"""

    if eclipse == True:
        wfc = pd.read_csv('../bin_analysis/spectra.csv',
                          index_col=[0, 1, 2]).sort_index()
        wfc = wfc.loc[('hatp41/visit00/reverse', 'marg', 4)]
        bin1_wfc = wfc['Central Wavelength'] - wfc['Wavelength Range']
        bin2_wfc = wfc['Central Wavelength'] + wfc['Wavelength Range']
        bin_wfc = np.stack((bin1_wfc, bin2_wfc)).T * 1e-6
        depth_wfc = wfc['Depth'] / 1e6
        error_wfc = wfc['Error'] / 1e6

        bin_spitzer = np.asarray([[3.2, 4.0], [4.0, 5.0]]) * 1e-6
        #depth_spitzer = np.asarray([1367, 2307]) / 1e6
        #error_spitzer = np.asarray([189, 210]) / 1e6
        depth_spitzer = np.asarray([1829, 2278]) / 1e6
        error_spitzer = np.asarray([319, 177]) / 1e6
        dilution = np.asarray([1.0069, 1.0111])
        error_spitzer = error_spitzer * dilution
        depth_spitzer = depth_spitzer * dilution
        bins = np.vstack((bin_wfc, bin_spitzer))
        depths = np.hstack((depth_wfc, depth_spitzer))
        errors = np.hstack((error_wfc, error_spitzer))
    else:
        # First, read in nikolay's sav file for stis
        nan = float('NaN')
        if stis:
            stis_dict = readsav('HAT41_STIS.sav')
            bin1_stis = stis_dict['w_g430'] - stis_dict['we_g430']
            bin2_stis = stis_dict['w_g430'] + stis_dict['we_g430']
            depth_blue = stis_dict['rprs_g430']**2
            error_blue = stis_dict['rprs_err_g430'] * 2 * np.sqrt(depth_blue)
            # This is specific to hatp41
            outlier = np.argmin(depth_blue)
            depth_blue = np.delete(depth_blue,outlier)
            error_blue = np.delete(error_blue, outlier)
            bin1_stis = np.delete(bin1_stis, outlier)
            bin1_stis[0] = 3.0e3 + 1e-11
            bin2_stis = np.delete(bin2_stis, outlier)
            bin_blue = np.stack((bin1_stis, bin2_stis)).T * 1e-10

            bin1_stis = stis_dict['w_g750'] - stis_dict['we_g750']
            bin2_stis = stis_dict['w_g750'] + stis_dict['we_g750']
            bin_red = np.stack((bin1_stis, bin2_stis)).T * 1e-10
            depth_red = stis_dict['rprs_g750']**2
            error_red = stis_dict['rprs_err_g750'] * 2 * np.sqrt(depth_red)

            bin_stis = np.vstack((bin_blue, bin_red))
            depth_stis = np.hstack((depth_blue, depth_red))
            error_stis = np.hstack((error_blue, error_red))
  
        else:
            bin_stis = np.empty((1, 2)) + nan
            depth_stis = np.empty(1) + nan
            error_stis = np.empty(1) + nan
        if spitz:
            bin_spitzer = np.asarray([[3.2, 4.0], [4.0, 5.0]]) * 1e-6
            # bin_spitzer=np.asarray([[, 4.0], [4.0,5.0]])*1e-6
            
            rprs_spitzer = np.asarray([.1001, .1028])
            dilution = np.asarray([1.0171, 1.0106])
            depth_spitzer = rprs_spitzer**2 * dilution
            error_spitzer = np.asarray([5e-4, 7e-4])
     
            error_spitzer = error_spitzer * 2.0 * rprs_spitzer * dilution

            #depth_spitzer = np.asarray([0.00992,.01028])
            #error_spitzer = np.asarray([8.e-5,1.3e-4])
        else:
            bin_spitzer = np.empty((1, 2)) + nan
            depth_spitzer = np.empty(1) + nan
            error_spitzer = np.empty(1) + nan

        if wfc3:
            wfc = pd.read_csv('../bin_analysis/spectra.csv',
                              index_col=[0, 1, 2]).sort_index()
            wfc = wfc.loc[('hatp41/visit01/reverse', 'marg', 4)]

            # wfc=pd.read_csv('../bin_analysis/tsiaras2.csv')
            bin1_wfc = wfc['Central Wavelength'] - wfc['Wavelength Range']
            bin2_wfc = wfc['Central Wavelength'] + wfc['Wavelength Range']
            bin_wfc = np.stack((bin1_wfc, bin2_wfc)).T * 1e-6
            depth_wfc = wfc['Depth'] / 1e6  # -.0002 #### Adjusted depth here!
            error_wfc = wfc['Error']/1e6
            # plt.errorbar(bin1_wfc,depth_wfc, error_wfc, ls='')
            # plt.show()
        else:
            bin_wfc = np.empty((1, 2)) + nan
            depth_wfc = np.empty(1) + nan
            error_wfc = np.empty(1) + nan

        bins = np.vstack((bin_stis, bin_wfc, bin_spitzer))
        depths = np.hstack((depth_stis, depth_wfc, depth_spitzer))
        errors = np.hstack((error_stis, error_wfc, error_spitzer))

        bins = bins[~np.isnan(bins)]
        bins = np.reshape(bins, (len(bins)//2, 2))
        depths = depths[~np.isnan(depths)]
        errors = errors[~np.isnan(errors)]
    return [bins, depths, errors]


if __name__ == '__main__':
    eclipse = bool(int(sys.argv[1]))
    try:
        inp = sys.argv[2]
        if inp == 0:
            inp = False
    except IndexError:
        print("DataFrame of 3 sigma ranges will not be saved," \
            "since no filename was given")
        inp = False
    try:
        condensate = sys.argv[3]
        ri = True
    except IndexError:
        print('No condensate given, Mie scattering will not be allowed')
        ri = None

    bins, depths, errors = get_data()
    # bins=bins[:-1]
    # depths=depths[:-1]
    # errors=errors[:-1]
    # depths[46:75]-=1.5e-4
    if eclipse == True:
        # cloud height? log pressure or pressure?
        ebins, edepths, eerrors = get_data(eclipse=True)
        retriever = Retriever()
        Rs = 1.84 * R_sun
        Mp = .96 * M_jup
        Rp_guess = 1.76 * R_jup
        T_guess = 1750
        Ts = 5960

        # test forward model
        tp = Profile()
        # T0, P1, alpha1, alpha2, P3, T3
        # 250, 1.25e5
        # make sure alpha1 and 2 are defined right
        #tp.set_parametric(1300, 250, .4, .6, 1.e6, 1800)

        calc = EclipseDepthCalculator()
        #wave, model = calc.compute_depths(tp, Rs, Mp, Rp_guess, Ts, logZ=1.6, CO_ratio=.4)
        #plt.plot(wave[wave<5e-6], model[wave<5e-6])
        #plt.errorbar(bins.mean(axis=1)[:-2], depths[:-2], errors[:-2], ls='', color='b', marker='o')
        #plt.xlim([1e-6, 5e-6])
        #plt.show()

        #sys.exit()

        T0 = 1500
        T3 = 1500
        P1 = 2.4
        P3 = 6.
        alpha1 = .4
        alpha2 = .6
        fit_info = retriever.get_default_fit_info(Rs=Rs, Mp=Mp, Rp=Rp_guess,
                                                  T=T_guess, T0=T0, log_P1=P1,
                                                  alpha1=alpha1,
                                                  alpha2=alpha2, log_P3=P3,
                                                  T3=T3,
                                                  logZ=0.0,
                                                  CO_ratio=.53,
                                                  log_cloudtop_P=np.inf,
                                                  log_scatt_factor=0,
                                                  T_star=Ts, add_H_minus_absorption=True,
                                                  profile_type="parametric")

        print('Setting Priors')
        fit_info.add_uniform_fit_param('Rp', 0.5*Rp_guess, 1.5*Rp_guess)
        # fit_info.add_uniform_fit_param('T', 0.5*T_guess, 1.5*T_guess)
        fit_info.add_uniform_fit_param('T0', 1000, 3000)
        fit_info.add_uniform_fit_param('T3', 1000, 3000)
        fit_info.add_uniform_fit_param('log_P1', -1, 7)
        fit_info.add_uniform_fit_param('log_P3', 3, 7)
        fit_info.add_uniform_fit_param('alpha1', .02, 2)
        fit_info.add_uniform_fit_param('alpha2', 0.02, 2)
        # fit_info.add_uniform_fit_param("log_scatt_factor", 0, 1)
        fit_info.add_uniform_fit_param("logZ", -1, 3)
        fit_info.add_uniform_fit_param("CO_ratio", .05, 2)
        # fit_info.add_uniform_fit_param("log_cloudtop_P", -.99, 5)
        print('starting retrieval')
        result = retriever.run_multinest(None, None, None, ebins,
                                         edepths, eerrors, fit_info,
                                         plot_best=True, nlive=50)
        result.plot_spectrum()
        plt.show()
        result.plot_corner()
        plt.show()
        #fig = corner.corner(result.samples, weights=result.weights,
         #                   range=[0.99] * result.samples.shape[1],
         #                   labels=fit_info.fit_param_names)
        #fig.show()
        sys.exit()
    retriever = Retriever()

    # Fixed: Rs, Mp, Rp
    Mp = .96 * M_jup
    # Rs=1.19*R_sun # possible other value
    Rs = 1.84 * R_sun  # gaia dr2
    Rp_guess = 1.84 * R_jup
    Rp_guess = 1.76 * R_jup  # from previous runs
    Ts = 5960
    T_guess = 1700
    co = .4
    # Rp_guess=1.58*R_jup  # value found by stis only with solar z
    ############################ Discovery values ##################3
    #Rs=1.683*R_sun
    #Rp_guess=1.685*R_jup
    #Mp=.8*M_jup
    #Ts=6390
    ########################### Tsantaki paper ###########
    #Rs = 1.19*R_sun
    #Rp_guess = 1.21 * R_jup
    #Mp = .79*M_jup # total guess from figure
    #Ts = 6500
    ##########

    # Open: T, logZ, CO_ratio, log_cloudtop_P
    test_abundance = False
    if test_abundance == True:
        getter = AbundanceGetter()
        calculator = TransitDepthCalculator()
        """
        #pkl = 'removed_mie'
        # get_best_fit(pkl)
        # sss
       
        #lsf = .0735189
        #Rs = 1179203609.32
        #rp = 116475464.
        CO = .27392921
        logz = 2.40125
        T = 1730.7394
        lcp = 3.87190445

        Rp = 1.75 * R_jup
        Rs = 1.878 * R_sun
        # temp=Rp/Rs
        # Rs=1.7*R_sun
        # Rp=temp*Rs*.99
        Mp = .824 * M_jup
        logZ = 1.75
        CO = .37
        T = 1581
        log_part_size = -6.55
        log_number_density = 4.71
        lcp = 6.83
        condensate = 'al2o3'
        ri = get_refractive_index(condensate)
        print(ri)
        # ri=None
        fsh = 8.3
        pss = .5

        # GJ stuff

        # Rp=.374*R_jup
        # Rs=.48*R_sun
        # Mp=.04*M_jup
        # logZ=0
        # CO=.54
        # T=800
        # log_part_size=np.log10(5e-7)
        # condensate='kcl'
        # ri=get_refractive_index(condensate)
        # fsh=3
        # log_number_density=6
        # lcp=8

        abundances = getter.get(logZ, CO)
        # abundances['H2O']*=0
        spot = Ts + 100
        cf = 0.12
        lsf = 0

        calculator.change_wavelength_bins(bins)
        w8, d8 = calculator.compute_depths(Rs, Mp, Rp, T, logZ=None,
                                           CO_ratio=None,
                                           cloudtop_pressure=10**lcp,
                                           T_star=Ts, ri=ri,
                                           frac_scale_height=fsh,
                                           part_size=10**log_part_size,
                                           number_density=10**log_number_density,
                                           add_scattering=False,
                                           add_collisional_absorption=False,
                                           T_spot=spot, spot_cov_frac=cf,
                                           custom_abundances=abundances,
                                           scattering_factor=10.**lsf)
        w9, d9 = calculator.compute_depths(Rs, Mp, Rp, T, logZ=None,
                                           CO_ratio=None,
                                           cloudtop_pressure=10**lcp,
                                           T_star=Ts, ri=ri,
                                           frac_scale_height=fsh,
                                           part_size=10**log_part_size,
                                           number_density=10**log_number_density,
                                           add_collisional_absorption=False,
                                           add_gas_absorption=False,
                                           part_size_std=pss, T_spot=spot,
                                           spot_cov_frac=cf,
                                           custom_abundances=abundances,
                                           scattering_factor=10.**lsf)
        w10, d10 = calculator.compute_depths(Rs, Mp, Rp, T, logZ=None,
                                             CO_ratio=None,
                                             cloudtop_pressure=10**lcp,
                                             T_star=Ts, ri=ri,
                                             frac_scale_height=fsh,
                                             part_size=10**log_part_size,
                                             number_density=10**log_number_density,
                                             add_gas_absorption=False,
                                             add_scattering=False,
                                             T_spot=spot, spot_cov_frac=cf,
                                             custom_abundances=abundances,
                                             scattering_factor=10.**lsf)
        w11, d11, mie_full = calculator.compute_depths(Rs, Mp, Rp, T, logZ=None,
                                                 CO_ratio=None,
                                                 cloudtop_pressure=10**lcp,
                                                 T_star=Ts, ri=ri,
                                                 frac_scale_height=fsh,
                                                 part_size=10**log_part_size,
                                                 number_density=10**log_number_density,
                                                 part_size_std=pss, T_spot=spot,
                                                 spot_cov_frac=cf,
                                                 custom_abundances=abundances,
                                                 full_output=True,
                                                 scattering_factor=10.**lsf)
        # Alter abundances here
        abundances = normalise_abundances(abundances, 'CO2')
        # mus=ss['mu_profile']

        mu = np.median(mie_full['mu_profile'])
        temp = np.mean(mie_full['T_profile'])
        g = G * Mp / Rp / Rp
        scale = k_B * temp / mu / AMU / g
        print('mie:', scale)"""
        
        # print mus.mean()
        mus = None
        Rp = 1.758 * R_jup
        Rs = 1.83 * R_sun
        # temp=Rp/Rs
        # Rs=1.7*R_sun
        # Rp=temp*Rs*.99
        Mp = .95 * M_jup
        logZ = 2.2
        CO = .25
        T = 1700
        lcp = 4.0
        Ts=5960
        #cf = 0.0
        #spot = 6060
        #f=0
        ri = None

        w11, d12, ss = calculator.compute_depths(Rs, Mp, Rp, T, logZ=logZ,
                                                 CO_ratio=CO,
                                                 cloudtop_pressure=10**lcp,
                                                 T_star=Ts, ri=ri,
                                                 full_output=True)
        """w11, d12, ss = calculator.compute_depths(Rs, Mp, Rp, T, logZ=logZ,
                                                 CO_ratio=CO,
                                                 cloudtop_pressure=np.inf,
                                                 T_star=Ts, ri=ri,
                                                 frac_scale_height=fsh,
                                                 part_size=10**log_part_size,
                                                 number_density=10**log_number_density,
                                                 part_size_std=pss, T_spot=spot,
                                                 spot_cov_frac=cf,
                                                 full_output=True,
                                                 scattering_factor=10.**lsf)

        f = 0
        d13 = f*d10 + (1-f)*d12
        logZ = 2.3
        f = 0.0
        # CO=.75
        # cf=0.05
        # T=1500
        # Mp=.8*M_jup
        Rp = 1.76 * R_jup
        w10, d10, ss = calculator.compute_depths(Rs, Mp, Rp, T, logZ=logZ,
                                                 CO_ratio=CO,
                                                 cloudtop_pressure=10**lcp,
                                                 T_star=Ts, ri=ri,
                                                 frac_scale_height=fsh,
                                                 part_size=10**log_part_size,
                                                 number_density=10**log_number_density,
                                                 part_size_std=pss, T_spot=spot,
                                                 spot_cov_frac=cf,
                                                 full_output=True,
                                                 scattering_factor=10.**lsf,
                                                 mus=mus)
        w11, d12, fid_full = calculator.compute_depths(Rs, Mp, Rp, T, logZ=logZ,
                                                 CO_ratio=CO,
                                                 cloudtop_pressure=np.inf,
                                                 T_star=Ts, ri=ri,
                                                 frac_scale_height=fsh,
                                                 part_size=10**log_part_size,
                                                 number_density=10**log_number_density,
                                                 part_size_std=pss, T_spot=spot,
                                                 spot_cov_frac=cf,
                                                 full_output=True,
                                                 scattering_factor=10.**lsf)

        d12 = f*d10 + (1-f)*d12
        spec_star = ss['stellar_spectrum']

        # Get scale height
        mu = np.median(fid_full['mu_profile'])
        temp = np.mean(fid_full['T_profile'])
        g = G * Mp / Rp / Rp
        scale = k_B * temp / mu / AMU / g # meters
        print('fid:', scale)
        print(d12[46:-2]*100)
    
        # mu=ss['mu_profile']

        # d8=spec_star*(1-d8)
        # d9=spec_star*(1-d9)
        # d10=spec_star*(1-d10)
        # d11=spec_star*(1-d11)
        # T_spot=Ts-300
        # cov=0.0

        #  w9, d9=calculator.compute_depths(Rs, Mp, Rp, T, logZ=logZ, CO_ratio=CO
        #                                 ,cloudtop_pressure=10**lcp
        #                                 ,T_star=Ts, T_spot=T_spot, spot_cov_frac=cov)

        # T_spot=Ts+300
        # cov=0.0
        # w6, d6=calculator.compute_depths(Rs, Mp, Rp, T, logZ=logZ, CO_ratio=CO
        #                                  ,cloudtop_pressure=10**lcp
        #                                  ,T_star=Ts, T_spot=T_spot, spot_cov_frac=cov)

        Rp = .99 * Rp
        logZ = 0
        lcp = 7
        log_part_size = -7
        log_number_density = 12
        fsh = .8
        # for item in abundances:
        #     abundances[item]*=0
        # print abundances
        # abundances['H2']+=.999
        abundances['H2O'] *= .01
        abundances['HCN'] *= 100
        # abundances['CH4']*=0
        # w9, d9=calculator.compute_depths(Rs, Mp, Rp, T, logZ=None, CO_ratio=None
        #                                  ,cloudtop_pressure=10**lcp
        #                                  ,T_star=Ts, custom_abundances=abundances)
        # w10, d10=calculator.compute_depths(Rs, Mp, Rp, T, logZ=None, CO_ratio=None
        #                                  ,cloudtop_pressure=10**lcp
        #                                  ,T_star=Ts, ri=ri, frac_scale_height=fsh
        #                                  , part_size=10**log_part_size
        #                                  ,number_density=10**log_number_density
        #                                  , custom_abundances=abundances)

        # w6, d6,xx=calculator.compute_depths(Rs, Mp, Rp, T, logZ=logZ
        #                                  , CO_ratio=CO
        #                                  ,cloudtop_pressure=10**8
        #                                     ,T_star=Ts
        #                                 , full_output=True)
        # w7,d7,xx=calculator.compute_depths(Rs, Mp, Rp, T, logZ=logZ
        #                                 , CO_ratio=CO
        #                                 ,cloudtop_pressure=10**lcp
        #                                 ,T_star=Ts
        #                                 , full_output=True)
        # f=.4
        # d6=f*d7+(1-f)*d6

        # logZ=2.2
        lcp = 2.0
        T = 1650
        # T=1550
        Rs = 1.26e9
        Rp = 1.238e8
        Mp = 1.8e27
        logZ = 2.1
        f = .5
        CO = .85
        # slope=8.14
        # lsf=-2.32
        # Rp=1.224e8
        # Mp=1.8e27
        abundances = getter.get(logZ, CO)
        tt = int(round(T, -2))/100 - 1
        pp = 8  # representative pressure of 1e4 Pa or 100 mbar
        # print np.log10(abundances['CO'][tt,:])
        # print np.log10(abundances['CO2'][tt,:])
        # print np.log10(abundances['H2'][tt,:])
        # abundances['H2']*=0
        # abundances['CO2']*=0
        # w6, d6=calculator.compute_depths(Rs, Mp, Rp, T, logZ=None
        #                                  , CO_ratio=None
        #                                  ,cloudtop_pressure=10**8, T_star=Ts
        #                                  , custom_abundances=abundances)
        # w7, d7=calculator.compute_depths(Rs, Mp, Rp, T, logZ=None
        #                                  , CO_ratio=None
        #                                  ,cloudtop_pressure=10**lcp, T_star=Ts
        #                                  , custom_abundances=abundances)
        # w6, d6=calculator.compute_depths(Rs, Mp, Rp, T, logZ=logZ
        #                                  , CO_ratio=CO
        #                                  ,cloudtop_pressure=10**8
        #                                  ,T_star=Ts)
        # w7,d7=calculator.compute_depths(Rs, Mp, Rp, T, logZ=logZ
        #                                 , CO_ratio=CO
        #                                 ,cloudtop_pressure=10**lcp
        #                                 ,T_star=Ts)
        # f=0.4
        # d6=f*d7+(1-f)*d6

        # plt.plot(w5,d5, color='orange', label='Low metal')
        # plt.plot(w6, d6, color='blue', label='High metal')

        # d8[46:75]+=32e-5
        # d6[46:75]+=26e-5
        # plt.errorbar(w8, d8, color='orange', label='Gas')
        # plt.errorbar(w9, d9, color='green', label='Scattering')
        # plt.errorbar(w10,d10, color='red', label='CIA')
        # plt.errorbar(w11,d11, color='k', label='Full')"""
        plt.errorbar(w11, d12, color='green', label='Clear', ls='', marker='x')
        #plt.errorbar(w11, d13, color='red', label='Partial', ls='', marker='x')
        plt.xscale('log')
        # plt.errorbar(w6,d7/d6-.99, color='red', label='Ratio', marker='x', ls='')
        # r8=d8-depths
        # r7=d9-depths
        # r5=d5-depths
        like1 = 0
        like2 = 0
        like3 = 0
        # like1 += -0.5 * np.sum(r8[6:]**2 / errors[6:]**2 + np.log(2 * np.pi * errors[6:]**2))
        # like2 += -0.5 * np.sum(r7[6:]**2 / errors[6:]**2 + np.log(2 * np.pi * errors[6:]**2))
        # like3 += -0.5 * np.sum(r5[6:]**2 / errors[6:]**2 + np.log(2 * np.pi * errors[6:]**2))
        print(like1)
        print(like2)
        # print like3

        # plt.plot(w3,d3)
        # depths[46:75]-=34.e-5
        # depths[:46]+=34.e-5
        # depths=stellar_spectrum*(1-depths)
        # errors=stellar_spectrum*errors
        plt.errorbar((bins[:, 0] + bins[:, 1])/2., depths[:], errors[:],
                     xerr=(bins[:, 1] - bins[:, 0])/2., marker='o',
                     color='b', ecolor='b', ls='')
        # plt.plot([3e-7, 1.7e-6],np.zeros(2))
        plt.xlim([.3e-6, 5e-6])
        # plt.ylim([.009,.0115])
        plt.legend()
        plt.show()
        sys.exit()

    if ri is not None:
        ri = get_refractive_index(condensate,
                                  wmin=bins.min()*1e6,
                                  wmax=bins.max()*1e6)
        fit_info = retriever.get_default_fit_info(Rs=Rs, Mp=Mp, Rp=Rp_guess,
                                                  T=T_guess, logZ=0,
                                                  CO_ratio=co,
                                                  log_cloudtop_P=0,
                                                  T_star=Ts,
                                                  cloud_fraction=1., offset=0,
                                                  offset2=0, offset3=0,
                                                  log_number_density=4.5, ri=ri,
                                                  log_part_size=-6.5,
                                                  frac_scale_height=0.5,
                                                  T_spot=Ts+100,
                                                  spot_cov_frac=0.0) 
        # Mie theory stuff
        fit_info.add_uniform_fit_param('log_number_density', 1, 15) # 1-15
        fit_info.add_uniform_fit_param('log_part_size', -8, -6)
        # fit_info.add_uniform_fit_param('frac_scale_height', .2, 10)
        # If log uniform
        fit_info.add_uniform_fit_param('frac_scale_height', -1, 1)

    else:
        fit_info = retriever.get_default_fit_info(Rs=Rs, Mp=Mp, Rp=Rp_guess,
                                                  T=T_guess, logZ=0,
                                                  CO_ratio=co,
                                                  log_cloudtop_P=0,
                                                  T_star=Ts,
                                                  cloud_fraction=1., offset=0,
                                                  offset2=0, offset3=0,
                                                  T_spot=Ts+100,
                                                  spot_cov_frac=0.0,
                                                  log_scatt_factor=0) # back to 0

    print('Setting priors')
    # Add gaussian or uniform priors
    #fit_info.add_gaussian_fit_param('Rs', 0.09*R_sun)
    #fit_info.add_gaussian_fit_param('Mp', 0.1*M_jup)
    fit_info.add_gaussian_fit_param('Rs', 0.2*R_sun)
    fit_info.add_gaussian_fit_param('Mp', 0.2*M_jup)

    fit_info.add_uniform_fit_param('Rp', 0.5*Rp_guess, 1.1*Rp_guess)
    # fit_info.add_uniform_fit_param('Rp', 0.985*Rp_guess, 1.015*Rp_guess)
    fit_info.add_uniform_fit_param('T', 0.5*T_guess, 1.5*T_guess)
    #fit_info.add_uniform_fit_param('T', 900,2500)
    #fit_info.add_uniform_fit_param("log_scatt_factor", -4, 8)
    #fit_info.add_uniform_fit_param("scatt_slope",-2,20)
    fit_info.add_uniform_fit_param("logZ", -1, 3)
    fit_info.add_uniform_fit_param("CO_ratio", .05, 2)
    # Change back to 3 for clouds above spectrum
    fit_info.add_uniform_fit_param("log_cloudtop_P", -3, 8) # change to 8
    #fit_info.add_uniform_fit_param("cloud_fraction", 0.0, 1.0)
    # fit_info.add_uniform_fit_param("spot_cov_frac",0.0,.2)
    # fit_info.add_uniform_fit_param("T_spot", 6150, 6900)
    # whitelight error from marg is 5 (50 ppm), and gaussian
    #fit_info.add_uniform_fit_param("offset", -10, 48)
    #fit_info.add_uniform_fit_param("offset2", -50, 30)
    #fit_info.add_uniform_fit_param("offset3", -48, 10)
    #fit_info.add_gaussian_fit_param("offset", 5.)
    #fit_info.add_gaussian_fit_param("offset2", 8.8)
    # fit_info.add_gaussian_fit_param("offset3", 8.5)
    # fit_info.add_uniform_fit_param("error_multiple", 0.5, 3)

    print('Starting fit')
    print
    # result = retriever.run_multinest(bins, depths, errors
    #                                  , fit_info, plot_best=True, npoints=500)

    # Emcee alternative
    emcee = False
    if emcee == True:
        result = retriever.run_emcee(bins, depths, errors, None,
                                     None, None,
                                     fit_info, plot_best=True)
        equal_samples = result.chain[:, 100:, :].reshape((-1, result.chain.shape[2]))

        fig = corner.corner(equal_samples,
                            range=[0.999] * equal_samples.shape[1],
                            labels=fit_info.fit_param_names, show_titles=True,
                            quantiles=[.16, .5, .84],
                            title_kwargs={"fontsize": 8})

        bestfit = result.flatchain[np.argmax(result.flatlnprobability)]
        plt.show()
    else:
        result = retriever.run_multinest(bins, depths, errors, None,
                                         None, None,
                                         fit_info, plot_best=True,
                                         npoints=100)
        plt.show()
        fig = corner.corner(result.samples, weights=result.weights,
                            range=[0.999] * result.samples.shape[1],
                            labels=fit_info.fit_param_names, show_titles=True,
                            quantiles=[.16, .5, .84], title_kwargs={"fontsize": 8})

        fig.show()
        equal_samples = nestle.resample_equal(result.samples, result.weights)
        bestfit = result.samples[np.argmax(result.logp)]
    sigm3 = []
    sigp3 = []
    sigm24 = []
    sigp24 = []
    sigm2 = []
    sigp2 = []
    sigm1 = []
    sigp1 = []
    median = []
    best = []

    for i, name in enumerate(fit_info.fit_param_names):
        if name == 'logZ':
            logz = np.percentile(equal_samples[:, i], 50)
        if name == 'CO_ratio':
            co = np.percentile(equal_samples[:, i], 50)
        if name == 'T':
            temp = np.percentile(equal_samples[:, i], 50)
        median.append(np.percentile(equal_samples[:, i], 50))
        best.append(bestfit[i])
        sigm3.append(np.percentile(equal_samples[:, i], .2))
        sigm24.append(np.percentile(equal_samples[:, i], 1.))
        sigm2.append(np.percentile(equal_samples[:, i], 2.3))
        sigm1.append(np.percentile(equal_samples[:, i], 16))
        sigp1.append(np.percentile(equal_samples[:, i], 84))
        sigp2.append(np.percentile(equal_samples[:, i], 97.7))
        sigp24.append(np.percentile(equal_samples[:, i], 99.))
        sigp3.append(np.percentile(equal_samples[:, i], 99.8))
        # sig3 is actually 2.4 sigma, I think the
        # limited sampling makes edge extreme.
        # ie, .2% is solar, but .7% is 68x solar

    out = pd.DataFrame()
    out['-3'] = sigm3
    out['-2.4'] = sigm24
    out['-2'] = sigm2
    out['-1'] = sigm1
    out['Median'] = median
    out['1'] = sigp1
    out['2'] = sigp2
    out['2.4'] = sigp24
    out['3'] = sigp3
    out['Best'] = best
    out['Params'] = fit_info.fit_param_names
    out = out.set_index('Params')
    outt = out.T
    try:
        outt['Rp'] = outt['Rp'] / R_jup
    except KeyError:
        pass
    try:
        outt['Rs'] = outt['Rs'] / R_sun
    except KeyError:
        pass
    try:
        outt['z'] = 10**outt['logZ']
    except KeyError:
        pass
    try:
        outt['Mp'] = outt['Mp'] / M_jup
    except KeyError:
        pass
    out = outt.T
    # Get representative water abundance
    try:
        getter = AbundanceGetter()
        abundances = getter.get(logz, co)
        tt = int(round(temp, -2))/100 - 1
        pp = 8  # representative pressure of 1e4 Pa or 100 mbar
        water = abundances['H2O'][tt, pp]
        out['log(H$_2$O)'] = np.log10(water)
    except NameError:
        pass
    out['Evidence'] = result.logz
    print(out)
    if inp:
        out.to_csv('./bestfit/'+inp+'.csv')
        with open('./bestfit/'+inp+'.pkl', 'w') as f:
            pickle.dump([result, fit_info], f)
