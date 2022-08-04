# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created/Last Edited: December 30, 2019

@author: Rajat Kumar
@maintainer: Rajat Kumar
Notes:
Script for calculating the cartridge masks and the center of rotation. 

"""

# %% All imports
import numpy as np
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
import nibabel as nib, nilearn as nil
import matplotlib.patches as mpatches
from skimage.draw import circle_perimeter,line, polygon,disk, line_aa
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.measure import find_contours
from skimage.transform import (rotate as rt,rescale,downscale_local_mean)
from skimage import filters



# %%
# =============================================================================
# Finding the inner cartridge
# =============================================================================
def findcartridge(data,slice_num,volume_num,sig=2,lt=0,ht=100,rad1=8,rad2=52, step=1, n =3):
    image = data.get_data()[:,:,slice_num,volume_num]
    edges = canny(image, sigma=sig, low_threshold=lt, high_threshold=ht)
    hough_radii = np.arange(rad1, rad2, step)
    hough_res = hough_circle(edges, hough_radii)
    accums, cr, cc, radii = hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=n)
    return [image,edges,cr,cc,radii]

# %%
# =============================================================================================================
# Mask calculation for the inner cartridge (Obsolete!! Changed to the new algorith based on finding the notch!)
# =============================================================================================================


#def inner_mask(data,findcartridge_parameters,slice_num,volume_num):
    
    #data_best_slices = data
    #temp_ind = findcartridge_parameters
    #count = 0
    #choice = 0
    
    #while(choice != 1):
        #if count == 0:
            #r_cord_ind = np.argmin(temp_ind[4])
            #r_cord = temp_ind[4][r_cord_ind]
            #x_cord = temp_ind[2][r_cord_ind]
            #y_cord = temp_ind[3][r_cord_ind]
            
        #else:
            #user_input = [float(p) for p in input('Enter x,y,r with a space').split()]
            #x_cord = user_input[0]
            #y_cord = user_input[1]
            #r_cord = user_input[2]
        
        #mask_image = np.zeros(data_best_slices.get_data()[:,:,slice_num,volume_num].shape) 
        #patch = mpatches.Wedge((y_cord,x_cord),r_cord,0,360)  
        #vertices = patch.get_path().vertices
        #x=[]
        #y=[]
        
        #for k in range(len(vertices)):
            #x.append(int(vertices[k][0]))
            #y.append(int(vertices[k][1]))
        #x.append(x[0])
        #y.append(y[0])
        #rr,cc = polygon(x,y)
        #mask_image[rr, cc] = 1

        
        #plt.figure()
        #plt.imshow(mask_image*np.mean(data_best_slices.get_data()[:,:,slice_num,volume_num].flatten())*5 + data_best_slices.get_data()[:,:,slice_num,volume_num])
        #plt.show()
        #print('Currently used x,y,r',[x_cord,y_cord,r_cord])
        #choice_list = [int(x) for x in input('Enter 1 to go to next slice, 0 to change x,y,r').split()]
        #choice = choice_list[0]
    
        #if choice == 1:
            #mask = mask_image
            
            #center = [x_cord,y_cord,r_cord] # this is the center of the mask
        #count +=1
    
    #return mask,center

    
# %%
# =================================================================================================
# Mask calculation for the inner cartridge 
# =================================================================================================

def inner_mask(data_path,slice_num,volume_num=0,filt = 'sobel',stepNum = 24, scaleFactor = 2, rad1=7,rad2=50,step=1):
    
    # Process arguments 
    filt = filt.lower()
    stepNum = round(stepNum)
    
    # filter options
    filts = {}
    filts['sobel'] = filters.sobel
    filts['scharr'] = filters.scharr
    filts['prewitt'] = filters.prewitt 
    # filts['roberts'] = filters.roberts # excluded due to bad performance
    # filts['hysteresis'] = filters.apply_hysteresis_threshold # For cleaner data if too much small contours found
    # hysteresis filter usage (image, low, high) - low threshold and high threshold 
    
    # Load data
    imOri = nib.load(data_path).get_data()[:,:,slice_num,volume_num]
    
    # Make sure the stepNum augment is greater than 1
    while stepNum <= 1:
        stepNum = input('Please use a stepNum that is greater than 1:')
    
    # Obtain the range value 
    lowTemp = imOri.min() # can be range based on the histogram 
    highTemp = imOri.max()
    stepTemp = (highTemp-lowTemp)/stepNum
    Step = round(stepTemp)
    lowRange = round(lowTemp + Step) # Subtract the first step to not include baseline
    highRange = round(highTemp + Step)
    # highRange = highTemp + Step # To include the highest intensity (not significant)
    lvls = range(lowRange,highRange,Step)
    leng = len(lvls)
    
    repeat = 1
    
    while(repeat):
        imFiltered = filts[filt](imOri)
        im = rescale(imFiltered, scaleFactor, preserve_range = True)
        # Upsample due to the difficulty in detection the inner cycle 
        #     upscale after applying filters 
        #     clearer inner cycle detection in this order and filter result will not be affect by the upsample

    
        # Set up subplots 
        fig = plt.figure(figsize=(leng+10,leng+15))  
        fig.subplots_adjust(hspace=0.1, wspace=0.001)      
        ax = fig.add_subplot(int(leng/4 +1) ,4, 1) 
        
        # The first subplot without specifying level
        # ax.imshow(im)
        contours =find_contours(im,fully_connected='high') # no level input
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], -contour[:, 0], linewidth=2)
        ax.set_title('Default level',fontsize=18)
        ax.set_axis_off()
        
        count = 2
        
        for lvl in lvls:
            # Set up subplots      
            ax = fig.add_subplot(int(leng/4 +1) ,4, count) 
            
            # Plot different subplots using different threshold 
            # ax.imshow(im)
            contours =find_contours(im,level = lvl,fully_connected='high')
            for n, contour in enumerate(contours):
                ax.plot(contour[:, 1], -contour[:, 0], linewidth=2)
            ax.set_title('Level: '+str(lvl),fontsize=18)
            ax.set_axis_off()
            count+=1
        plt.show()
        
        
        
        print('If none of the plots show clear circles, you need to choose different scaling factor and edge filter to find contours again. Fail to detect clear circuit will cause error in the subsequent estimation')
        repeat = int(input('Do you want to repeat and change the values? 1 for yes, 0 for no'))
        if repeat:
            print(f'The current filter is {filt}')
            filt = input('Enter the new filter (string); choose from \'sobel\',\'scharr\',\'prewitt\'')
            print(f'The current scaling factor is {scaleFactor}')
            scaleFactor = int(input('Enter the new scaling (float); recommended range (1~3)'))
        else: 
            lvl = input('Which level value results in clear circles?')
            if lvl.isnumeric():
                contours =find_contours(im,level = lvl, fully_connected='high')
            else: 
                contours =find_contours(im, fully_connected='high')
    
    
    smallest_circle = [] #detects the inner circle with notch
    for i in range(len(contours)):
        smallest_circle.append(contours[i].shape[0])
    temp_var = np.array(smallest_circle)
    temp_var = np.delete(temp_var,np.argmax(temp_var))
    temp_var = np.delete(temp_var,np.argmax(temp_var))
    temp_var = np.delete(temp_var,np.argmax(temp_var))
    index = np.argwhere(np.array(smallest_circle)==np.max(temp_var))[0][0]
    
    fig, ax = plt.subplots()
    fig.set_size_inches(12,12)
    ax.imshow(im)
    ax.plot(contours[index][:, 1], contours[index][:, 0], linewidth=3)
    ax.set_axis_off()
    plt.show()
    
    img = np.zeros(im.shape)
    img[(contours[index][:,0]).astype('int'),(contours[index][:,1]).astype('int')]=1
    imgDownsampled = downscale_local_mean(img, (scaleFactor, scaleFactor))
     
    hough_radii = np.arange(rad1, rad2, step)
    hough_res = hough_circle(imgDownsampled, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=2)    
    radii_complete = radii[np.argmax(radii)]  # complete refers to the full circle with notch
    cx_complete = cx[np.argmax(radii)]
    cy_complete = cy[np.argmax(radii)]

    radii_incomplete = radii[np.argmin(radii)]  # complete refers to the circle without notch
    cx_incomplete = cx[np.argmin(radii)]
    cy_incomplete = cy[np.argmin(radii)]
    
    rr,cc = disk([cy_complete,cx_complete],radii_complete-1) # Erroded by 1 voxel for removing the notch
    img_complete = np.zeros(imOri.shape)
    img_complete[rr,cc]=1
    
    rr,cc = disk([cy_incomplete,cx_incomplete],radii_incomplete)
    img_incomplete = np.zeros(imOri.shape)
    img_incomplete[rr,cc]=1
    
    
    return img_complete,cy_complete,cx_complete,radii_complete 


# %%
# =================================================================================================
# Finding the center of rotation
# =================================================================================================

def cen_rotation(data_path,slice_num,img_complete,cy_complete,cx_complete,radii_complete,canny_sgm=1):
    
    temp_img= img_complete * (nib.load(data_path).get_data()[:,:,slice_num,0])
    
    cir_mask = np.zeros(temp_img.shape)
    rr,cc = disk([cy_complete,cx_complete],radii_complete-2) # erosion to get rid of boundaries
    cir_mask[rr,cc] = 1
    
    contrast_enh= exposure.equalize_hist(temp_img)
    sobel_edges = filters.sobel(contrast_enh)
    sobel_masked = sobel_edges  *cir_mask
    im = np.power(sobel_masked,5) # increases the contrast such that the quadrant intersection is visible; depends on T2* relaxation, so can vary with the age of the cartridge. 
    

    
    dotp_all = []
    for i in range(len(np.nonzero(cir_mask)[0])):
        possible_angles = np.linspace(0,360,720)
        all_coords = np.nonzero(cir_mask)
   
        test_line = np.zeros(temp_img.shape)
        rr,cc = line(all_coords[0][i]-(radii_complete-1),all_coords[1][i],all_coords[0][i]+(radii_complete-1),all_coords[1][i])
        test_line[rr,cc]=1
        rr,cc = line(all_coords[0][i],all_coords[1][i]-(radii_complete-1),all_coords[0][i],all_coords[1][i]+(radii_complete-1))
        test_line[rr,cc]=1
    
        for j in possible_angles:
            test_line_rt = rt(test_line, angle = j,center=(np.array((all_coords[1][i],all_coords[0][i]))), order=3, preserve_range=True) # the format for center is col, row; not row, col - documentation of skimage is incorrect for 0.13.x
            dotp = np.sum((test_line_rt * im).flatten())
            dotp_all.append(dotp)
   
    row_cor = np.nonzero(cir_mask)[0][int(np.argmax(dotp_all)/len(possible_angles))]
    col_cor = np.nonzero(cir_mask)[1][int(np.argmax(dotp_all)/len(possible_angles))]
    angle_move = possible_angles[np.argmax(dotp_all[(int(np.argmax(dotp_all)/720)*720):(int(np.argmax(dotp_all)/720)*720)+720])]  
    
    ## Just for visualization
    
    vis_line = np.zeros(temp_img.shape)
    rr,cc = line(row_cor-(radii_complete-1),col_cor,row_cor+(radii_complete-1),col_cor)
    vis_line[rr,cc]=1
    rr,cc = line(row_cor,col_cor-(radii_complete-1),row_cor,col_cor+(radii_complete-1))
    vis_line[rr,cc]=1
    vis_line_rt = rt(vis_line, angle =angle_move,center=(np.array((col_cor,row_cor))), order=3, preserve_range=True)

    plt.imshow(temp_img)
    plt.title('Original Image')
    plt.figure()
    plt.imshow(contrast_enh)
    plt.title('Contrast Enhanced Image')
    plt.figure()
    plt.imshow(im)
    plt.title('Sobel Image')
    plt.figure()
    plt.imshow(vis_line_rt )
    plt.title('Estimated Center')
    plt.show()

    print('COR,COC',row_cor,col_cor)
        
    
    return [row_cor,col_cor]

# %%
# =================================================================================================
# Mask calculation for the outer cartridge 
# =================================================================================================

def outer_mask(data,findcartridge_parameters,slice_num,volume_num):
    
    data_best_slices = data
    temp_ind = findcartridge_parameters
    count = 0
    choice = 0
    
    while(choice != 1):
        if count == 0:
            r_cord_ind = np.argmax(temp_ind[4])
            r_cord_neg = np.argmin(temp_ind[4])
            
            for w in [0,1,2]:
                if (w!= r_cord_ind) and (w!=r_cord_neg):
                    mid = w
            
            r_cord = temp_ind[4][r_cord_ind]
            x_cord = temp_ind[3][r_cord_ind]
            y_cord = temp_ind[2][r_cord_ind]
            #w = temp_ind[4][mid]
            w = r_cord-14
        else:
            user_input = [float(p) for p in input('Enter x,y,r,w with a space').split()]
            x_cord = user_input[0] 
            y_cord = user_input[1]
            r_cord = user_input[2]
            w = user_input[3] 
        omask_image = np.zeros(data_best_slices.get_data()[:,:,slice_num,volume_num].shape)
        patch = mpatches.Wedge((x_cord,y_cord),r_cord,0,360,w) # Checked the values for an excellent overlap mask for inner cyl.
        vertices = patch.get_path().vertices
        x=[]
        y=[]
        for k in range(len(vertices)):
            x.append(int(vertices[k][0]))
            y.append(int(vertices[k][1]))
        x.append(x[0])
        y.append(y[0])
        rr,cc = polygon(x,y)
        omask_image[rr, cc] = 1

        
        plt.figure()
        plt.imshow(omask_image*np.mean(data_best_slices.get_data()[:,:,slice_num,volume_num].flatten())*5 + data_best_slices.get_data()[:,:,slice_num,volume_num])
        plt.show()
        print('Currently used x,y,r,w',[x_cord,y_cord,r_cord,w])
        choice_list = [int(x) for x in input('Enter 1 to go to next slice, 0 to change x,y,r,w').split()]
        choice = choice_list[0]
    
        if choice == 1:
            omask = omask_image            
        count +=1
    
    return omask



# %%