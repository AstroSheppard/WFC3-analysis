import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import marg_mcmc as wl
#import whitelight2018 as wl

# Note to Kyle: This is originally designed to loop through each
# visit once, use that to get a weighted avg of transit time,
# a/r, inc, etc., then use that avg value to either be fixed
# for each visit or as a prior. However, only have dealt with
# single visits lately, so this is superfluous.

# Unclear if this method is better than fitting all simultaneously,
# via phase folding or something (hard because so many systematic variables.
# Emcee seems overkill here. Could use marg only on each, then use resulting
# values as priors for emcee individually. Anyway, ignoring for now.

def wlanalysis(planet, visits, dirs):
    """Take preprocessed data and perform a whitelight fit to 
    determine the physical parameters and get residuals for
    each model. 

    INPUTS: planet, visit

    DOES: 3 or 4 fits. The first 2 or 3, in order, determine priors
    for center time, inclination, and a/r* (transits). Uses priors
    in a final totally open fit. 

    RETURNS: Residuals for each model, best fit system parameters
    with errors. 

    TODO: Compare this with one open fit. Also try not limiting
    other parameters at all. They should always be limited anyway 
    (3 sigma form lit)
    
    Maybe try open, and if error is greater than like 30%, limit to 
    lit value. Do for every visit once. Then, determine best system
    parameters: For inc and a/r*, weighted mean
    of constrained visits (though for weighted mean may not matter.
    For epoch, use lower error of 1) lowest error + period 2) Error
    from fit for that visit 3) literature value + period error.
    Then run again, with all visits fixed
    to those values for consistency. NOTE: For bidirectional scans,
    both must use same time.

    """
    visit=planet+'/'+visits+'/'+dirs
    results=find_priors(visit)
    # array to save params for each visit
    # for visit in visits:

        ### Function that does this
       
        # limb darkening always open? Shouldn't matter much for transits.
        # Fit for time open + random, all else limited.
        # then inc
        # then a/r* (transit only)
        # Finally once with all open
        # returns values and errors for each
        ### Function that takes two 2D arrays (params or errors by visits) and returns
        # best values
        ### Function that determines time for each visit
        ### Rerun each visit with those values and save WLresults, params
    final_sys_params=results
    return final_sys_params
def find_priors(visit):

    # Read in csv
    proc='processed_data.csv'
    par='system_params.csv'
    df=pd.read_csv(proc, index_col=[0,1]).loc[visit]
    df2=pd.read_csv(par, index_col=0).loc[visit]
    pre=pd.read_csv('preprocess_info.csv', index_col=0).loc[visit]
    
    first = pre['User Inputs'].values[-2].astype(int)
    last = pre['User Inputs'].values[-1].astype(int)
    transit=df['Transit'].values[0]
    date=df.loc['Value','Date'].values[first:last]
    spectra=df.loc['Value'].iloc[first:last,1:-2].dropna(axis=1).values
    # spectra=spectra[~np.isnan(spectra)]
    spec_err=df.loc['Error'].iloc[first:last,1:-2].dropna(axis=1).values

    # spec_err=spec_err[~np.isnan(spec_err)]
    inputs=df2['Properties'].values
    errors=df2['Errors'].values
    # Get time for this visit, with error
    inputs[1], errors[1] = event_time_prior(date, inputs, errors)

    # make errors an input here? Not now...
    # find best tcenter prior with errors
 #   params = wl.whitelight2018(inputs, date, spectra, spec_err
 #                              ,transit=transit)
    #depth, depth_err, tcenter, tc_err, inc, inc_err, MpMsR (ar)
    #, MpMsR_err (ar_err), c, c_err

    #inputs: rp/rs, event time, inclin, a/r, period, eclipse depth, limb1, limb 2
    # if transit == True:
    #     inputs[0]=np.sqrt(params[0])
    # else:
    #     inputs[5]=params[0]
    # print params[3]
    # print errors[1]
    # if params[3] < errors[1]:
    #     inputs[1]=params[2]
    #     errors[1]=params[3]
    #     print 'switched'

    # find best inc prior with errors, limiting tcenter to within 3/5 sigma
    # params = wl.whitelight2018(inputs, date, spectra, spec_err, fixtime=True
    #                            , openinc=True, transit=transit)
    # print params[5]
    # print errors[2]
    # plt.plot(params[0:2], params[0:2])
    # plt.show()
    # if params[5] < errors[2]:
    #     inputs[2]=params[4]
    #     errors[2]=params[5]
    #     print 'switched'
    #a/r* for transits too, with both above limited to 3/5 sigma
    # params = wl.whitelight2018(inputs, date, spectra, spec_err, fixtime=True
    #                            , openar=True, transit=transit)
    # if transit == True:
    #     inputs[0]=np.sqrt(params[0])
    # else:
    #     inputs[5]=params[0]
    # print params[7]
    # print errors[3]
    # if params[7] < errors[3]:
    #     inputs[3]=params[6]
    #     errors[3]=params[7]
    #     print 'switched'
    # run with all open, maybe try with all limited?
    openar=False
    fixtime=False
    emcee = False
    #visit='no_inflation_hatp41'
    results = wl.whitelight2020(inputs, date, spectra, spec_err,
                                openar=openar, fixtime=fixtime, savewl=visit
                                , transit=transit, plotting=False)

    return results
  
"""

   
   
        savfile2='./WLresults/best_params/constrained/'+planet+'/'+inpfile[i]+'.sav'
        SAVE, filename=savfile2, final_results, starting_params
"""

# Program to determine the expected eclipse time
def event_time_prior(date, inputs, errors):
    #  Inputs
    #  date: 1D array of the date of each exposure (MJD)
    #  properties: 1D array containing the last observed eclipse 
    #  and the period. (MJD, days)
    #  NOTE: If transit has been observed but eclipse hasn't, then 
    #  use last observed transit and add period/2
    time=inputs[1]
    period=inputs[4]
    i=0
    if date[0] > time:
        while date[0] - time > period/2.:
            time+=period
            i+=1

    while time - date[0] > period/2.:
        time-=period
        i+=1

    err=(errors[1]**2+(i*errors[4])**2)**.5
    return time, err

if __name__ == '__main__':
    if len(sys.argv) < 5:
        sys.exit('Incorrect input')
    planet=sys.argv[1]
    visits=sys.argv[2]
    dirs=sys.argv[3]
    transit=bool(int(sys.argv[4]))
    
    sys=wlanalysis(planet, visits, dirs)
    # Save system params for planet
    data=[sys[2:8]]
    cols=['Event Time', 'Event Time Error', 'Inc', 'Inc Error', 'AR*'
             , 'AR* Error']
    df=pd.DataFrame(data, columns=cols)
    df['Transit']=transit
    df['Planet']=planet
    df=df.set_index(['Planet', 'Transit'])

    try:
        cur=pd.read_csv('./fit_sys_params.csv', index_col=[0,1])
        cur=cur.drop((planet, transit), errors='ignore')
        cur=pd.concat((cur,df))
        cur.to_csv('./fit_sys_params.csv', index_label=['Planet', 'Type'])
    except IOError:
        df.to_csv('./fit_sys_params.csv', index_label=['Planet','Type'])
