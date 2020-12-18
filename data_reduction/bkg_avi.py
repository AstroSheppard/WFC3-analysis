from __future__ import print_function
import glob
import sys
import os
import shutil

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import tqdm
import pandas as pd

coords = []
bkg_coords = []
bkg_coords2 = []

def onclick(event):
    global ix, iy
    ix=event.xdata
    iy=event.ydata
    print('x = %d, y = %d' % (ix, iy))

    global coords
    coords.append((ix, iy))
    
    if len(coords) == 2:
        fig.canvas.mpl_disconnect(cid)

def onclick_bkg(event):
    global iix, iiy
    iix=event.xdata
    iiy=event.ydata
    print('x = %d, y = %d' % (iix, iiy))

    global bkg_coords
    bkg_coords.append((iix, iiy))
    
    if len(bkg_coords) == 2:
        fig.canvas.mpl_disconnect(cid)

def onclick_bkg2(event):
    global iiix, iiiy
    iiix=event.xdata
    iiiy=event.ydata
    print('x = %d, y = %d' % (iiix, iiiy))

    global bkg_coords2
    bkg_coords2.append((iiix, iiiy))
    print(bkg_coords2)
    if len(bkg_coords2) == 2:
        fig.canvas.mpl_disconnect(cid)

        
def test(img, window, final_img,get_window=True):
        """ Test to make sure window size does not cut 
        out important data"""
        #scale=46.7
        raw_fit=bkg(img, window, get_window=get_window, test=True)
        final_img=final_img#*scale
	fits.writeto('./test_zapping/raw.fits', final_img, overwrite=True)
	sys.exit('Compare background-subtracted image with input image (raw.fits)')

def get_data(visit, direction=None):
        """ Extract only quality, spectroscopic fits files """
        # Read in data and decare arrays
        ima=np.sort(np.asarray(glob.glob('../planets/'+visit+'/*ima.fits')))
        raw=np.sort(np.asarray(glob.glob('../planets/'+visit+'/*ima.fits')))
        fit=fits.open(ima[0])
        example=fit[1].data
        xsize, ysize=np.shape(example)

        fit.close()

	# Make sure the data is spectroscopic and SPARS10 (scan, not rapid)
        obstype=np.zeros(len(ima)).astype(str)
        rate=np.zeros(len(ima)).astype(str)
        quality=np.zeros(len(ima)).astype(str)
  	for i,img in tqdm(enumerate(ima)):
                fit=fits.open(img)
                header=fit[0].header
                dire=header['POSTARG2'] + 5.0
                if direction == 'forward':
                    if dire > 0:
      	                obstype[i]=header['OBSTYPE'] 
                        rate[i]=header['SAMP_SEQ']
            	        quality[i]=header['QUALITY']
                elif direction == 'reverse':
                    if dire < 0:
      	                obstype[i]=header['OBSTYPE'] 
                        rate[i]=header['SAMP_SEQ']
            	        quality[i]=header['QUALITY']
                else:
      	            obstype[i]=header['OBSTYPE'] 
                    rate[i]=header['SAMP_SEQ']
            	    quality[i]=header['QUALITY']   
                    fit.close()
        index=(obstype == 'SPECTROSCOPIC')* (
           (rate ==  'SPARS10') + (rate == 'SPARS25')) * (quality != 'LOCKLOST')
        og=len(ima)
        ima=ima[index]
        raw=raw[index]
        print(len(ima), " images remain out of", og, "originals")
        return [ima, raw]


def maxrow(frame):
        """Return index of maximum flux row"""
	f=np.sum(frame,axis=1)
  	#f1=np.argmax(f)
        # Define max value as the median of the top 5
        # highest flux rows to avoid oddities from
        # cosmic rays
        norm = np.median(np.sort(f)[-5:])
        perc = 0.01#, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
        #for per in perc:
        #    # Get all rows with flux > 2% of norm flux --- this can be adjustable
        index = np.where(f > perc*norm)[0]
        index = check_continuity(index)
                #print "Percent of max flux used for cut-off: %.3f" % per
                #break
        
        low = np.min(index)
        high = np.max(index)
        mid = (low+high)/2
        
  	return low, mid, high

def check_continuity(index):
    flag = np.zeros_like(index)
    for i in range(len(index)):
        if index[len(index)/2] - index[i] != len(index)/2 - i:
            flag[i] = 1
    new_index = index[flag == 0]
    print("Flags: %f" % flag.sum())
    
    return new_index
    #return np.all(np.arange(len(index))+min(index) == index)

# Input 1 fits file, make array with each extension (3-D: [ext, x, y])
def bkg(raw, size_window, test=False, get_window=False):
        """ Use differential frames (Deming 2013) to remove
        background."""
        with fits.open(raw) as exp:
                xlen, ylen = np.shape(exp[1].data)
                header=exp[0].header
                corr=header['UNITCORR']
                nFrames=exp[-1].header['EXTVER']
                err=exp[2].data
  	        frames = np.zeros((nFrames, xlen, ylen))
  	        times=np.zeros(nFrames)
  	        frame_diffs = np.zeros((nFrames-1, xlen, ylen))
  	        count=0
                # Iterate through extensions in fits file
  	        for item in exp[1:]:
                    if 'SCI' in item.header['EXTNAME']:
                        frames[count,:,:]=item.data
     		        times[count]=item.header['SAMPTIME']
     		        count= count + 1
  	

	# Check if the units are in electrons or in electrons/s. If per
	# second, multiply frame by samptime. 
	# Zero out background that is far enough from source to be noise (or
	# is from secondary source)

        removed=0
        new_coords = False
  	if 'OMIT' in corr:
     		# Last frame is for tsamp=0. If brightness not in units of e/s, then
		# this last frame will give large negatives. Set it to 0

                # Adjust window size here, since "dir" is not a perfect correlation.
     		#window=np.floor(size_window).astype(int)

     		frames[-1,:,:]=0.0
     		ny=frames.shape[2]
    		# window=40
    		# window=12 ; hatp41
     		for j in range(count-1):
        		f1=frames[j,:,:]      
        		f2=frames[j+1,:,:]     
        		frame_diffs[j,:,:]=f1-f2
                        if len(bkg_coords) == 2:
                            new_coords = True
                        
                        if get_window==True and j == 0:
                            #centroid=maxrow(f1-f2)
                            centroid=maxrow(f1-f2)[1]
                            if new_coords == False:
                                fig=plt.figure()
                                ax=plt.imshow(f1-f2)
                                cid = fig.canvas.mpl_connect('button_press_event', onclick_bkg)
                                cid = fig.canvas.mpl_connect('button_press_event', onclick_bkg)
                                print("Click the top-left then the bottom-right corners")
                                plt.show()
                                bkgc= [int(i) for item in bkg_coords for i in item]
                                x1=bkgc[1]
                                x2=bkgc[3]
                                y1=bkgc[0]
                                y2=bkgc[2]
                            elif new_coords == True:
                                fig=plt.figure()
                                ax=plt.imshow(f1-f2)
                                cid = fig.canvas.mpl_connect('button_press_event', onclick_bkg2)
                                cid = fig.canvas.mpl_connect('button_press_event', onclick_bkg2)
                                print("Click the top-left then the bottom-right corners")
                                plt.show()
                                bkgc= [int(i) for item in bkg_coords2 for i in item]
                                x1=bkgc[1]
                                x2=bkgc[3]
                                y1=bkgc[0]
                                y2=bkgc[2]
                            high=x2-centroid
                            low=centroid-x1
                            size_window = low, high, y1, y2
                        else:
                            low, high, y1, y2 = size_window
        		#mrow=maxrow(frame_diffs[j,:,:])
                        low, mrow, high=maxrow(frame_diffs[j,:,:])
                        low = mrow - low
                        high = high - mrow
                        
                        # prevent window outside of data
                        # if mrow+high > ny: window = ny-mrow
                        # Mask data, find median of background,
                        # subtract from image, zero out bkg
                        mask=np.zeros_like(f1-f2).astype(bool)
                        mask[mrow-low:mrow+high, y1:y2]=True
                        zero=np.ma.array(f1-f2, mask=mask)
                        med=np.ma.median(zero)
                        zero=zero.data-med
                        zero[~mask]=0
                        frame_diffs[j,:,:]=zero
                        removed+=med
                        
                        #bg=np.concatenate((frame_diffs[j,:mrow-low,:],
                        #                   frame_diffs[j, mrow+high:,:]),axis=0)
                        #med=np.median(bg)
                     
                      
       			#frame_diffs[j,:,:]=frame_diffs[j,:,:]-med
        		#frame_diffs[j,:mrow-window,:]=0 
        		#frame_diffs[j,mrow+window:,:]=0
                        if test:
        	                file1='./test_zapping/dif'+str(j)+'.fits'
        	                file2='./test_zapping/frame'+str(j)+'.fits'
                                fits.writeto(file1,frame_diffs[j,:,:], overwrite=True)
                                fits.writeto(file2,f1, overwrite=True)

  	if 'COMPLETE' in corr:
     	        ny=frames.shape[2]
     		for j in range(count-1):
        		f1=frames[j,:,:]*times[j]
        		f2=frames[j+1,:,:]*times[j+1]
        		frame_diffs[j,:,:]=f1-f2
                        
                        if get_window==True and j == 0:
                            centroid=maxrow(f1-f2)
                            fig=plt.figure()
                            ax=plt.imshow(f1-f2)
                            cid = fig.canvas.mpl_connect('button_press_event', onclick_bkg)
                            cid = fig.canvas.mpl_connect('button_press_event', onclick_bkg)
                            print("Click the top-left then the bottom-right corners")
                            plt.show()
                            bkgc= [int(i) for item in bkg_coords for i in item]
                            x1=bkgc[1]
                            x2=bkgc[3]
                            y1=bkgc[0]
                            y2=bkgc[2]
                            high=x2-centroid
                            low=centroid-x1
                            size_window= low, high, y1, y2
                          
                        else:
                            low, high, y1, y2 = size_window
        		mrow=maxrow(frame_diffs[j,:,:])
                        
                        # prevent window outside of data
                        if mrow+high > ny: window = ny-mrow
                        # Mask data, find median of background,
                        # subtract from image, zero out bkg
                        mask=np.zeros_like(f1-f2).astype(bool)
                        mask[mrow-low:mrow+high, y1:y2]=True
                        zero=np.ma.array(f1-f2, mask=mask)
                        med=np.ma.median(zero)
                        zero=zero.data-med
                        zero[~mask]=0
                        frame_diffs[j,:,:]=zero
                        removed+=med
                        
                        if test:
        	                file1='./test_zapping/dif'+str(j)+'.fits'
        	                file2='./test_zapping/frame'+str(j)+'.fits'
                                fits.writeto(file1,frame_diffs[j,:,:], overwrite=True)
                                fits.writeto(file2,f1, overwrite=True)
  
 
  
  	output=frame_diffs.sum(0)
        #print low, high, y1, y2
        #plt.imshow(output)
        #plt.show()
        
        # Removed = bkg error squared, add in quadrature
        err=np.sqrt(removed+np.square(err))

        if test: fits.writeto('./test_zapping/fbkg.fits', output, overwrite=True)
        if get_window == True:
            return [output, err, [low, high, y1, y2]]
        else:
            return [output, err]
  
if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit('Please use python [bkg.py] [planet] [visit]')
    visit=sys.argv[1]+'/'+sys.argv[2]
    ima, raw=get_data(visit)
    n_forward, n_reverse=0,0
    direction=np.zeros(len(ima))

    ### Each img: Get total count, exp time, and start time and save to header
    for i,img in tqdm(enumerate(ima), desc='Getting data'):
        exp=fits.open(img)
        header=exp[0].header
        dire=header['POSTARG2'] + 5.0 
        direction[i]=dire
        if dire > 0:
       	    n_forward+=1
            if n_forward == 1:
                forward_img=exp[1].data
            exp.close()
        else:
            n_reverse+=1
       	    if n_reverse == 1: 
                reverse_img=exp[1].data
            exp.close()
    # Center the source                               
    if n_forward == 0:
        img=reverse_img
    else:
        img=forward_img            
    fig=plt.figure()
    ax=plt.imshow(img)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    print("Click the top-left then the bottom-right corners")
    plt.show()
    # need to make sure x and y are being extracted correctly
    coords= [int(i) for item in coords for i in item]
    x1=coords[1]
    x2=coords[3]
    y1=coords[0]
    y2=coords[2]
    xlen=x2-x1
    ylen=y2-y1
    coords=[]
    if n_forward > 0:
        allimages_f = np.zeros((n_forward, xlen, ylen))
        allheader_f = np.asarray([])
        fzdir='./reduced/'+visit+'/forward/bkg/'
        f2dir='./reduced/'+visit+'/forward/final/'
        try:
            os.makedirs(fzdir)
        except OSError:
            if os.path.isdir(fzdir):
                shutil.rmtree(fzdir)
                os.makedirs(fzdir)
            else:
                raise
        try:
            os.makedirs(f2dir)
        except OSError:
            if os.path.isdir(f2dir):
                shutil.rmtree(f2dir)
                os.makedirs(f2dir)
            else:
                raise
    if n_reverse > 0:
        allimages_r = np.zeros((n_reverse, xlen, ylen))
        allheader_r = np.asarray([])
        rzdir='./reduced/'+visit+'/reverse/bkg/'
        r2dir='./reduced/'+visit+'/reverse/final/'
        try:
            os.makedirs(rzdir)
        except OSError:
            if os.path.isdir(rzdir):
                shutil.rmtree(rzdir)
                os.makedirs(rzdir)
            else:
                raise
        try:
            os.makedirs(r2dir)
        except OSError:
            if os.path.isdir(r2dir):
                shutil.rmtree(r2dir)
                os.makedirs(r2dir)
            else:
                raise

    # determine window size for both forward and reverse images
    # direction array contains something related to scan rate for each image
    #w1=direction[0]
    #w2=direction[1]
    #rwindow=0
    #fwindow=0
        
    # If direction[0] is negative, then it is a reverse scan, and
    # we set the window to be a size that typically captures all
    # source photons
    #if w1 < 0:
    #    rwindow=np.ceil(2*np.abs(w1))*xw
    # If it's positive, then it's a forward scan and we set the
    # forward window instead
    #else:
    #    fwindow=np.ceil(2*np.abs(w1))*xw
    # Now we check the second scan, which can be negative for a
    # bi-directional, in which case we set the reverse window. 
    #if w2 < 0:
    #    rwindow=np.ceil(2*np.abs(w2))*xw
    # For uni-directional, the rate doesnt change so this does nothing
    #else:
    #    fwindow=np.ceil(2*np.abs(w2))*xw
    # For one data set the scan direction (I use as a proxy for rate)
    # was super low, so we can set it to be either the other directions
    # window or 1, in case the other direction is 0 (ie,
    # unidirectional and low scan "rate")
    #if rwindow <= 1: rwindow = max(fwindow,1)
    #if fwindow <= 1: fwindow = max(rwindow,1)

   

    # For each exposure, check if reverse or forward scan.
    # Subtract background
    # Apply window to center source
    # Save image to all images, save header to all headers 
    f=0
    r=0
    fwindow=[1,1,1,1]
    rwindow=[1,1,1,1]
    if len(sys.argv)==5: test(ima[2], rwindow, reverse_img)
    for i,expo in tqdm(enumerate(ima), desc='Subtracting background'):
        obs=fits.open(expo)
        hdr=obs[0].header#.tostring(sep='\\n')
        image=obs[1].data
        if obs[1].header['BUNIT'] == 'ELECTRONS':
            hdr['Count']=np.mean(image)/hdr['EXPTIME']
        else:
            hdr['Count']=np.mean(image)
        obs.close()
        if direction[i] > 0:
            if f==0:
                print(" getting forward direction window")
                get_window=True
                raw_fit, err_array, fwindow=bkg(expo, fwindow, get_window=get_window)
                #print "fwindow ", fwindow
                #print "done"
            else:
                get_window=False
                #print "iteration ", i
                #print "fwindow ", fwindow
                raw_fit, err_array=bkg(expo, fwindow, get_window=get_window)
                
           
	    img = raw_fit[x1:x2,y1:y2]
            raw_file=fits.open(raw[i])
            raw_img=raw_file[1].data[x1:x2,y1:y2]
            raw_file.close()
            errors=err_array[x1:x2, y1:y2]
            prim=fits.PrimaryHDU(img, header=hdr)
            err=fits.ImageHDU(errors)
            rawi=fits.ImageHDU(raw_img)
            hdul=fits.HDUList([prim, err, rawi])
            filename = fzdir + "%03d"%f + '.fits'
            hdul.writeto(filename, overwrite=True)
	    f+=1
        else:
            if r==0:
                print("getting reverse window")
                get_window=True
                raw_fit, err_array, rwindow=bkg(expo, rwindow, get_window=get_window)
                #print "rwindow ", rwindow
                #print "done"
            else:
                get_window=False
                #print "iteration", i
                #print "rwindow ", rwindow
                raw_fit, err_array=bkg(expo, rwindow, get_window=get_window)
 
	    img = raw_fit[x1:x2,y1:y2]
            raw_file=fits.open(raw[i])
            raw_img=raw_file[1].data[x1:x2,y1:y2]
            raw_file.close()
            errors=err_array[x1:x2, y1:y2]
            prim=fits.PrimaryHDU(img, header=hdr)
            err=fits.ImageHDU(errors)
            rawi=fits.ImageHDU(raw_img)
            hdul=fits.HDUList([prim, err, rawi])
            filename = rzdir + "%03d"%r + '.fits'
            hdul.writeto(filename, overwrite=True)
	    r+=1

    # Now save aperture coordinates
    cols=['Initial Aperture']
    coords=pd.DataFrame([x1,x2,y1,y2],columns=cols)
    coords['Visit']=visit
    coords=coords.set_index('Visit')

    try:
        cur=pd.read_csv('./coords.csv', index_col=0)
        cur=cur.drop(visit)
        cur=pd.concat((cur,coords))
        cur.to_csv('./coords.csv')
    except IOError:
        coords.to_csv('./coords.csv')   

    print('Finished removing background ' + visit + ' visit')    

    # Write reduced data to a directory
    # for k in range(n_forward):
    #     filename = fzdir + "%03d"%i + 'f.fits'
    #     image = allimages_f[k,:,:]
    #     hdr=fits.Header.fromstring(allheader_f[k], sep='\\n')
    #     fits.writeto(filename, image, header=hdr, overwrite=True)
    # for k in range(n_reverse):
    #     filename = rzdir + "%03d"%k + 'r.fits'
    #     image = allimages_r[k,:,:]
    #     hdr=fits.Header.fromstring(allheader_r[k], sep='\\n')
    #     fits.writeto(filename, image, header=hdr)    



  
 


