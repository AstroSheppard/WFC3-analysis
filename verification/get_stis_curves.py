from scipy.io import readsav

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
import sys


def get_lc_files(visit):
    period = 2.694047
    tc=0
    if visit ==  '83':
        tc = 58000.69582
    elif visit == '84':
        tc = 58245.85414
    elif visit == '85':
        tc = 58280.87484

    print tc
    folder = 'HAT41_STIS_LCs_V2/SPECLC/visit' + str(visit) +'/*'
    data=np.sort(np.asarray(glob.glob(folder)))


    stisd = pd.DataFrame()
    stism = pd.DataFrame()
    fig = plt.figure(figsize=(7, 12))
    colors = iter(cm.inferno(np.linspace(0.1, .8, len(data))))
    i=0
    for savfile in data:
        c = next(colors)
        con = 0.0035
        waves = np.asarray(savfile[:-4].split('/')[-1].split('_')).astype(float)
        wave = waves.mean()/1e4
        print wave
        if wave == .56065:
            pass
        else:

            stist = pd.DataFrame()
            stist2 = pd.DataFrame()
            info = readsav(savfile)
            cor = info['flux_cr']
            corerr = info['flux_err']
            date = info['mjd']
            model = info['flux_md']
            modelx = info['mjd_md']

            phase = (date-tc)/period
            phase = phase - np.floor(phase)
            phase[phase > 0.5] = phase[phase > 0.5] - 1.0
            xmin=np.min(phase)-0.005
            xmax=np.max(phase)+.02

            modelx = (modelx - tc) / period
            modelx = modelx - np.floor(modelx)
            modelx[modelx > 0.5] = modelx[modelx > 0.5] - 1.0

            stist['Bin']=[i]*len(cor)
            stist['Wave']=[wave]*len(cor)
            stist['Phase']=phase
            stist['Flux']=cor
            stist['Error']=corerr
            stist2['Bin']=[i]*len(model)
            stist2['Wave']=[wave]*len(model)
            stist2['phase']=modelx
            stist2['Model']=model
         
          
            stisd=stisd.append(stist)
            stism=stism.append(stist2)
            
            #xmin=np.min(date)-0.01
            #xmax=np.max(date)+.052
            if i % 2 == 0 :
                alpha = .35
                marker='o'
            else:
                alpha = 1.0
                marker='o'
            plt.errorbar(phase, cor-i*con, corerr, color=c,
                         ls='', marker='o', ecolor=c, markersize = 3, alpha=alpha)
            #plt.errorbar(date, cor-i*con, corerr, color=c,
            #             ls='', marker=marker, ecolor=c, markersize = 3, alpha=alpha)
            plt.plot(modelx, model-i*con, color=c, alpha=alpha)
            plt.xlim([xmin, xmax])
            plt.ylabel('Normalized Flux - Constant')
            plt.xlabel('Orbital Phase')
            #plt.xlabel('Date [MJD]')
            plt.text(xmax-.018, 0.9976-i*con, r'%.3f$\mu$m' % wave, color=c)
            #plt.text(-.06, 1.002-i*con, r'$\chi^2_{red}$=%.2f' % rchi2, color=c)
            i += 1
    plt.ylim([1-i*con-.011, 1.0027])
    #plt.savefig('../../h41_lc_stis'+str(visit)+'_oct.pdf')

    np.savetxt('../../mrt/stis_data'+str(visit)+'.txt', stisd.values, fmt=['%02d', '%.5f','%+.6f','%.6f', '%.6f'])
    np.savetxt('../../mrt/stis_model'+str(visit)+'.txt', stism.values, fmt=['%02d','%.5f','%+.6f','%.6f'])
    stisd.to_csv('../../mrt/stid_data'+str(visit)+'.csv', index=False)
    stism.to_csv('../../mrt/stid_model'+str(visit)+'.csv', index=False)
if __name__ == '__main__':
    visit = sys.argv[1]
    get_lc_files(visit)
