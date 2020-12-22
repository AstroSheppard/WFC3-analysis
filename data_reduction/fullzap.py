from __future__ import print_function
import glob
import sys

import numpy as np
from astropy.io import fits
from tqdm import tqdm
from tqdm import trange

from scipy.special import erfinv
import matplotlib.pyplot as plt

def bad_pixels(data, headers, errors, raws, savefile):
    """ Used to fix bad pixels, but that's no
    longer an issue with flat field correction.
    Now, this 1) gets aperture and 2) zeros negative
    pixels"""
    n=len(data)
    x=np.zeros(3)                   #[apt,min,max]
    y=np.zeros(3)
    if n > 1:
        ####Find base aperture####
        #x,y=aperture(data[1])
        x,y=ap(data[1])
        #x,y=ap(data[1])

        # Use larger aperture to include all light
        update=np.asarray([1,-1,1])
        xl=x+0*update
        yl=y+0*update
        for i,img in enumerate(data):
            expo=img[xl[1]:xl[2], yl[1]:yl[2]]
            error=errors[i,xl[1]:xl[2], yl[1]:yl[2]]
            raw=raws[i, xl[1]:xl[2], yl[1]:yl[2]]
            np.place(expo, expo<0, 0)
            zapped=expo
            #zapped=pixel_zapping(expo)
            # save zapped images in folder named by dataset
            hdr=fits.Header.fromstring(headers[i], sep='\\n')
            filename=savefile+ "%03d.fits" % i
            # Add data mask file
            prim=fits.PrimaryHDU(zapped.data, header=hdr)
            ext=fits.ImageHDU(zapped.mask.astype(int))
            err=fits.ImageHDU(error)
            r=fits.ImageHDU(raw)
            hdul=fits.HDUList([prim, ext, err,r])
            hdul.writeto(filename, overwrite=True)

def ap(frame):
    """Return index of maximum flux row"""
    f=np.mean(frame,axis=1)
    #f1=np.argmax(f)
    # Define max value as the median of the top 5
    # highest flux rows to avoid oddities from
    # cosmic rays
    norm = np.median(np.sort(f)[-5:])
    per=0.02
    #for per in perc:
        # Get all rows with flux > 2% of norm flux --- this can be adjustable
    index = np.where(f > per*norm)[0]
    #if check_continuity(index):
    #    print "Percent of max flux used for cut-off: %.3f" % per
    #    #break

    xmin = np.min(index)
    xmax = np.max(index)
    xapt = (xmax - xmin) / 2

    f=np.mean(frame,axis=0)
    #f1=np.argmax(f)
    # Define max value as the median of the top 5
    # highest flux rows to avoid oddities from
    # cosmic rays
    norm = np.median(np.sort(f)[-5:])
    per=0.05
    #for per in perc:
        # Get all rows with flux > 2% of norm flux --- this can be adjustable
    index = np.where(f > per*norm)[0]
    #if check_continuity(index):
    #    print "Percent of max flux used for cut-off: %.3f" % per
    #    #break

    ymin = np.min(index)
    ymax = np.max(index)
    yapt = (ymax - ymin) / 2

    x=np.asarray([xapt,xmin,xmax])
    y=np.asarray([yapt,ymin,ymax])
    return (np.vstack((x,y)))

    return low, high
def aperture(exposure):
    """Given a sample exposure data, return the basic aperture."""

    xlen,ylen = exposure.shape
    center=np.zeros(2).astype(int)
    exposure_xcen = exposure[(xlen/2),:]

    # Use 10% of max pixel as gauge for edge cutoff
    max_count=np.max(exposure)
    scan = np.where(exposure_xcen > max_count/10.)[0]
    scan_width = len(scan)
    scan_center = int(np.median(scan))

    exposure_ycen = exposure[:,scan_center]
    height = np.where(exposure_ycen > max_count/10.)[0]
    scan_height = len(height)
    height_cen=int(np.median(height))

    # Redo width for new y-center to make sure
    # center is accurate
    exposure_xcen=exposure[height_cen,:]
    scan2=np.where(exposure_xcen > max_count/10.)[0]
    scan_width= len(scan2)
    scan_center=int(np.median(scan2))
    center[0]=height_cen
    center[1]=scan_center
    xapt=scan_height/2
    yapt=scan_width/2
    xmin=center[0]-xapt
    xmax=center[0]+xapt
    ymin=center[1]-yapt
    ymax=center[1]+yapt
    x=np.asarray([xapt,xmin,xmax])
    y=np.asarray([yapt,ymin,ymax])
    return (np.vstack((x,y)))


def pixel_zapping(allspec, plot=False):
    """ Removes bad pixels by comparing each pixel
    to its column's median. This is outdated. No bad pixels
    exist after flat field fix. Keep nloop at 0"""
    nloop=0
    # Pixels expected to be outside sigma factor in image
    np.place(allspec, allspec<0, 0)
    print('Negative pixels = ', np.sum(allspec<0))

    xapt, yapt=allspec.shape
    # Zoom in to ignore lower-flux pixels in median calculation
    xapt1=xapt-8
    nextra=5.
    inp=1-nextra/yapt/xapt1
    n=(2.)**(.5)*erfinv(inp)
    # Sigma rejection factor chosen such that only 5 pixels in image can be expected
    # to naturally lie outside the sigma range. I then can correct the pixels
    # outside the sigma range without worry about overcorrecting

    for j in range(nloop):
        # Take care of any other bad pixels by comparing each pixel to the
        # median of it's own column
        allspec1=allspec.copy()
        allspec1.mask[0:4,:]=True
        allspec1.mask[-4:,:]=True
        column_med=np.ma.median(allspec1,axis=0)
        col_med=np.broadcast_to(column_med, allspec.shape)
        denom=xapt-np.sum(allspec1.mask, axis=0)
        sigma=np.sqrt(np.sum((allspec1-col_med)**2,axis=0)/denom)
        sigma=np.broadcast_to(sigma, allspec.shape)
        dif=np.abs(allspec-col_med)

        # Account for edge effects: If the median of the row is much much
        # different then the median of all rows, then don't change the pixels
        row_m=np.broadcast_to(np.ma.median(allspec, axis=1), (yapt, xapt)).T
        ycut=np.ma.median(row_m)*.9
        yhigh=np.ma.median(row_m)*1.1
        # Replace the bad pixels with the median of their column
        index=(dif > n*sigma)* (row_m >= ycut) * (row_m <= yhigh)
        allspec.data[index]=col_med[index]
    #print 'Bad pixels found =',np.sum(index)
    return allspec

def zapped(allspec):
        """Input is 3D numpy array of all bkg-removed exposure.
        Median values and sigma cuts are used to remove cosmic rays"""
        #dims=np.empty_like(allspec)
        #aspec=allspec.copy()
        nspec, nspat, nwave = allspec.shape
        nloop = [8, 5]
        for sig in nloop:
            nzap = 0
            # Get row means, exluding top and bottom 15 to ignore CRs
            rows=np.ma.mean(np.sort(allspec, axis=2)[:,:,15:-15],axis=2)
            medians=np.tile(np.ma.median(allspec, axis=0),(nspec, 1,1))

            ### Median sigma check
            #sigmas=np.sqrt(np.sum((allspec-medians)*(allspec-medians), axis=0)/(nspec-np.sum(allspec.mask, axis=0)))
            #np.place(sigmas, sigmas==0, 1)
            #crs=np.abs(allspec-medians)/sigmas
            ###
            temp=allspec.copy()
            x=np.asarray(range(nspec))
            for i in range(nspat):
                y=temp[:,i,:].sum(axis=1)
                y/=np.median(y)
                m, b=np.polyfit(x, y, 1)
                shift=m*x+b
                shift=np.tile(shift, (temp.shape[2],1)).T
                np.place(shift, shift==0, 1)
                temp[:,i,:]=temp[:,i,:]/shift

            # correct for uneven scan rate before cr check
            row_anom=np.moveaxis(np.tile(rows.anom(axis=0), (nwave,1,1)), 0, 2)
            #row_anom2=np.moveaxis(np.tile(rows2.anom(axis=0), (nwave,1,1)), 0, 2)

            sigma=np.tile(temp.std(axis=0), (nspec, 1, 1))
            np.place(sigma, sigma==0, 1)
            crs=temp.anom(axis=0)/sigma
            #np.place(sigma2, sigma2==0, 1)
            #crs2=allspec.anom(axis=0)/sigma2
            #plt.imshow(allspec[69,:,:])
            #plt.show()
            #plt.imshow(allspec[73,:,:])
            #plt.show()
            #plt.imshow(crs[:,24,:])
            #plt.show()
            #plt.imshow(crs2[:,24,:])
            #plt.show()

            uneven_scan_row=np.moveaxis(np.tile(np.abs(rows.anom(axis=0)/rows.std(axis=0))>2.
                                                , (nwave,1,1)), 0, 2)


            #plt.imshow(test/np.mean(test))
            #plt.show()
            #plt.imshow(np.abs(rows.anom(axis=0)/rows.std(axis=0)))
            #plt.show()
            #plt.imshow(test/np.mean(test)>3.3)
            #plt.show()
            #plt.imshow(sigma[31,:,:])
            #plt.show()
            #plt.imshow(crs[73,:,:])
            #plt.show()
            #plt.imshow(uneven_scan_row[:,:,1])
            #plt.show()
            index=(crs>sig) * (~uneven_scan_row)
            #plt.imshow(index[73,:,:].astype(int))
            #plt.show()
            nzap=np.ma.sum(index)
            allspec.data[index]=medians[index]
            print('Cosmic rays corrected %d' % nzap)
            print('Total pixels %d' % (nspat * nwave))
            print('Percent %.4f' % (100*(nzap + 0.0)/nspat/nwave))

        #sss
        return allspec

if __name__=='__main__':
    if len(sys.argv) != 3:
        sys.exit('Use python2 [fullzap.py] [planet] [visit]')
    else:
        visit=sys.argv[1]+'/'+sys.argv[2]
        bad_pixels(visit, 'forward')
        bad_pixels(visit, 'reverse')
