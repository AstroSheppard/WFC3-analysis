from __future__ import print_function
import sys
import configparser


import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import norm

import wlpaper
import binpaper as bp
sys.path.append('../bin_analysis')
from compare_spectra import compare_spectra

if __name__=='__main__':

    config = configparser.ConfigParser()
    config.read('./config.py')
    # Read in data.
    planet = config.get('DATA', 'planet')
    visit_number = config.get('DATA', 'visit_number')
    direction = config.get('DATA', 'scan_direction')
    visit = planet + '/' + visit_number + '/' + direction
    transit = config.getboolean('DATA', 'transit')
    binsize = config.getint('DATA', 'bin_width')
    bin_number = config.getint('DATA', 'bin_number')

    # Which white-light fit to get residuals from.
    ignored_exposures = config.getboolean('DATA', 'ignored_exposures')
    inflated_errors = config.getboolean('DATA', 'inflated_errors')
    openar = config.getboolean('DATA', 'openar')
    include_first_orbit = config.getboolean('DATA', 'include_first_orbit')
    include_removed_points = config.getboolean('DATA', 'include_removed_points')

    # Specify spectral model inputs.
    method = config.get('DATA', 'method')
    mcmc = config.getboolean('DATA', 'mcmc')  
    include_error_inflation = config.getboolean('DATA', 'include_error_inflation')
    include_wl_adjustment = config.getboolean('DATA', 'include_wl_adjustment')
    include_residuals = config.getboolean('DATA', 'include_residuals')
    ld_type = config.get('DATA', 'limb_type')
    linear_slope = config.getboolean('DATA', 'linear_slope')
    quad_slope = config.getboolean('DATA', 'quad_slope')
    exp_slope = config.getboolean('DATA', 'exp_slope')
    log_slope = config.getboolean('DATA', 'log_slope')
    # Something about get chi-squared, normality values
    get_chi = config.getboolean('FIGURES', 'get_chi')
    get_normality = config.getboolean('FIGURES', 'get_normality')
    get_autocorrelations = config.getboolean('FIGURES', 'get_autocorrelations')
    get_spectral_curves = config.getboolean('FIGURES', 'get_spectral_curves')
    get_wl_curve = config.getboolean('FIGURES', 'get_wl_curve')
    get_spectra_compare = config.getboolean('FIGURES', 'get_spectra_compare')
    weighted = config.getboolean('FIGURES', 'get_weighted_spectrum')
    get_parameter_values = config.getboolean('FIGURES', 'get_parameter_values')
    ###
    save_comp = config.getboolean('COMPARE SPECTRA', 'save_comparison')
    save_comparison_name = config.get('COMPARE SPECTRA', 'save_comparison_name')
    visits = config.get('COMPARE SPECTRA', 'visits').split(';')
    methods = config.get('COMPARE SPECTRA', 'methods').split(';')
    bin_widths = config.get('COMPARE SPECTRA', 'bin_widths').split(';')
    # Specify save info.
    save_plots = config.getboolean('SAVE', 'save_plots')

    
    # Use config file inputs to define the full visit
    # name and spec method (specs).
    visit = planet + '/' + visit_number + '/' + direction
    planet_index = visit
    if inflated_errors == False:
        visit = visit + '_no_inflation'
    if ld_type == 'linear':
        visit = visit + '_linearLD'
    if ignored_exposures == True:
        visit = visit + '_no_first_exps'
    if openar == True:
        visit = visit + '_openar'
    if quad_slope == True:
        visit = visit + '_quad'
    full_method = method + str(int(include_first_orbit)) + str(int(include_removed_points)) \
                  + str(int(include_residuals)) + str(int(include_wl_adjustment)) \
                  + str(int(include_error_inflation))


    save_name = visit.replace('/', '_') + '_' + full_method + '_' + str(binsize)
    
    # Spectrum itself, especially 4 overlapping for b (maybe normalized c by scale height too?)
    if get_spectra_compare == True:
        #visits.append(visit)
        #methods.append(full_method)
        #bin_widths.append(str(binsize))
  
        compare_spectra(bin_widths
                        , visits
                        , methods
                        , save_comp=save_comp
                        , save_name=save_comparison_name
                        , weighted=weighted)
    

    # Plots:
    # wlpaper: raw and de-trended whitelight curve
    # correalte: autocorrelation of whitelight and bins
    # correlate: correlated noise figure of whitelight and bins
    # binvis: spectral light curve figure
    # compare_spectra: Spectrum itself, especially 4 overlapping for b (maybe normalized c by scale height too?)
    # Other results:
    # binvis/correlated: reduced chi-squared
    # correlated: normality test values

    # Maybe depth/parameter value for tables
    # 

    # First, get raw, de-trended, and residuals of whitelight curve
    # to show example light curve and data fitting.
    if get_wl_curve==True:
        wlpaper.wlpaper(visit, method, savefig=save_plots)
    # All whitelight consistent with normality, consistent with chi-squared =1. Some red noise
    # for certain combos. This is acceptable as removable structure. 


    # Bin paper stuff
    bin_file = '../bin_analysis/outputs/'
    datafile = bin_file + 'bin_data.csv'
    pfile = bin_file + 'bin_params.csv'
    phot = pd.read_csv(pfile, index_col=[0, 1, 2, 3, 4]).sort_index()
    try:
        phot_error=phot.loc[(visit, full_method, binsize, bin_number, 'Values')
                            , 'Photon Error'].values[0]
    except AttributeError:
        phot_error=phot.loc[(visit, full_method, binsize, bin_number, 'Values')
                            , 'Photon Error']

    # Get wavelengths from spectra file.
    spectra = bin_file + 'spectra.csv'
    sp = pd.read_csv(spectra, index_col=[0,1,2]).sort_index()
    spec = sp.loc[(visit, full_method, binsize), 'Central Wavelength'].values
    wave = spec[bin_number]

    # autocorrelation of whitelight and bins
    # correlated noise figure of whitelight and bins
 
    # Other results:
    # reduced chi-squared
    # normality test values

    # spectral light curve figure
    if get_spectral_curves==True:
        bp.binvis(visit
                  , full_method
                  , binsize
                  , spec
                  , save_plots=save_plots
                  , save_name=save_name)

    if get_autocorrelations==True:
        
        marg = pd.read_csv(datafile, index_col=[0, 1, 2, 3]).sort_index()
        marg = marg.loc[(visit, full_method, binsize)]

        nrow = int(np.ceil((len(spec)+1)/3.))
        f, axes = plt.subplots(nrow, 3, sharex='col', sharey='row', figsize=(12, 12))
        ax = f.add_subplot(111, frame_on=False)
        f2, axes2 = plt.subplots(nrow, 3, sharex='col', sharey='row', figsize=(12, 12))
        ax2 = f2.add_subplot(111, frame_on=False)
        ax.set_xlabel('Exposures Per Bin', labelpad=30, fontsize=15)
        ax.set_ylabel('Normalized RMS', labelpad=40, fontsize=15)
        ax2.set_xlabel('Lag', labelpad=30, fontsize=15)
        ax2.set_ylabel('Autocorrelation', labelpad=40, fontsize=15)
        s = visit.split('/')
        #ax.set_title('WASP-19b Correlated Noise Analysis\n', fontsize=18)
        #ax.set_title('%s %s %s Correlated Noise Analysis\n' % (s[0],s[1],s[2]) , fontsize=18)
        #ax.set_title('Marginalization Correlated Noise Analysis\n', fontsize=18)
        ax.tick_params(bottom=False, labelbottom=False, top=False, labeltop=False
                       ,left=False, labelleft=False,  right=False, labelright=False)
        ax2.tick_params(bottom=False, labelbottom=False, top=False, labeltop=False
                       ,left=False, labelleft=False,  right=False, labelright=False)
        plt.subplots_adjust(hspace=0)
        plt.subplots_adjust(wspace=0)
        colors = iter(cm.rainbow(np.linspace(0.2, .9, len(np.ravel(axes)))))
        beta = np.empty(len(spec))

        # Get free params
        params = pd.read_csv(pfile, index_col=[0,1,2,3,4]).sort_index()
        params = params.loc[(visit, full_method, binsize)]
        
        rchis = []
        for i, ax in enumerate(np.ravel(axes)):
            ax2 = np.ravel(axes2)[i]
            if i == len(spec):
                #ax.set_xscale('log')
                #ax.set_yscale('log')

                #ax.xaxis.set_major_formatter(ScalarFormatter())
                #ax.get_xaxis().get_major_formatter().labelOnlyBase = False
                #ax.yaxis.set_major_formatter(ScalarFormatter())
                #ax.minorticks_off()
                #ax.set_yticks([.1,1])
                #ax.set_xticks([1,2,3,4,5,6,7,8,9])
                modelfile = '../wl_preprocess/data_outputs/wl_models_info.csv'
                model_info = pd.read_csv(modelfile, index_col=[0, 1]).loc[visit]
                best_model = model_info.loc['Weight', :'Model 124'].astype(float).idxmax()
                # best_model = model_info.loc['Weight'].iloc[:-1].astype(float).idxmax()
                best = model_info.loc[:, best_model]
                resids = best.loc['Residuals'].values*1e6


                # Test what normal residuals look like
                #mresids= np.median(errors)/np.median(mresids)*mresids
                nGen = 250
                shape2 = resids.shape[0] // 2
                #shape2=9
                binsize = np.arange(shape2) + 1
                rmss = np.zeros((nGen, shape2))
                for j in range(nGen):
                    gen_resids = norm.rvs(scale=np.std(resids), size=len(resids))
                    rmss[j, :] = bp.correlated(gen_resids, wave
                                               , ax, 'grey', alpha=.1, plot=False)
                medians = np.percentile(rmss, 50, axis=0)
                up1 = np.percentile(rmss, 84, axis=0)
                up2 = np.percentile(rmss, 97.7, axis=0)
                down1 = np.percentile(rmss, 16, axis=0)
                down2 = np.percentile(rmss, 2.3, axis=0)

                ax.plot(binsize, medians, color='white', alpha=1.0, ls='--')
                ax.fill_between(binsize, medians, up1, color='grey', alpha=.7)
                ax.fill_between(binsize, medians, down1, color='grey', alpha=.7)
                ax.fill_between(binsize, up1, up2, color='grey', alpha=.4)
                ax.fill_between(binsize, down1, down2, color='grey', alpha=.4)

                bp.correlated(resids, 0, ax, 'grey')
                bp.autocorr(resids, 0, ax2, 'grey')
                break
            mresids = marg.loc[i,'Residuals'].values
            errors = marg.loc[i, 'Corrected Flux Error'].values
            nfree = (params.loc[(i, 'Errors'),'rprs':'WL coeff'].values != 0).sum()
            dof = len(mresids) - nfree
            chi2 = np.sum(mresids*mresids/errors/errors)
            rchi2 = chi2 / dof
            rchis.append(rchi2)
            print('Bin %2d' % i)
            if get_chi==True:
                print('Chi squared:  %.2f' % chi2)
                print('Reduced Chi squared:  %.2f' % rchi2)
                print('DOF:  %d' % dof)
            wave=spec[i]
            if get_normality==True:
                bp.adtest(mresids, phot_error/1e6)


            # Test what normal residuals look like
            #mresids= mresids/errors
            nGen = 250
            shape2 = mresids.shape[0] // 2
            #shape2=9
            binsize = np.arange(shape2) + 1
            rmss = np.zeros((nGen, shape2))
            for j in range(nGen):
                gen_resids = norm.rvs(scale = np.std(mresids), size=len(mresids))
                #gen_resids = norm.rvs(scale = errors, size=len(mresids))
                rmss[j, :] = bp.correlated(gen_resids, wave
                                           , ax, 'grey', alpha=.1, plot=False)
            medians = np.percentile(rmss, 50, axis=0)
            up1 = np.percentile(rmss, 84, axis=0)
            up2 = np.percentile(rmss, 97.7, axis=0)
            down1 = np.percentile(rmss, 16, axis=0)
            down2 = np.percentile(rmss, 2.3, axis=0)
            ax.plot(binsize, medians, color='white', alpha=1.0, ls='--')
            ax.fill_between(binsize, medians, up1, color='grey', alpha=.7)
            ax.fill_between(binsize, medians, down1, color='grey', alpha=.7)
            ax.fill_between(binsize, up1, up2, color='grey', alpha=.4)
            ax.fill_between(binsize, down1, down2, color='grey', alpha=.4)
            
            beta[i] = bp.correlated(mresids, wave, ax, next(colors))
            bp.autocorr(mresids, wave, ax2, 'r')

        print("Median reduced chi squared of all bins: %.3f" % np.median(rchis))
        print("Mean reduced chi squared of all bins: %.3f"  % np.mean(rchis))
        # name='rednoise_'+visit.replace('/','_')+'_marg.pdf'
        # name = '../../rednoise_'+visit.replace('/','_')+'_marg.pdf'
        #name = '../../autocorr_'+visit.replace('/','_')+'_marg.pdf'
        # name = '../../rednoise_sim.pdf'
        # With an empty figure, do this
        # axes[-1,2].legend(*axes[0,2].get_legend_handles_labels(), prop={'size': 10})
        # axes2[-1,2].legend(*axes2[0,2].get_legend_handles_labels(), prop={'size': 10})
        # When there are no empty figures, do below
        axes[0,2].legend(prop={'size': 10})
        axes2[0,2].legend(prop={'size': 10})
        if save_plots==True:
            f.savefig('./outputs/' + save_name + '_rednoise.pdf', bbox_inches='tight')
            f.clf()
            plt.close(f)
            f2.savefig('./outputs/' + save_name + '_acor.pdf', bbox_inches='tight')
            f2.clf()
            plt.close(f2)
        else:
            plt.show()
            plt.close('all')

        # sp.loc[(visit, 'marg', binsize), 'Beta Max']=beta
        # sp.to_csv('../bin_analysis/spectra.csv', index_label=['Obs', 'Method', 'Bin Size'])
        # print(beta)
        # print(np.mean(beta))
        # print(np.median(beta))
