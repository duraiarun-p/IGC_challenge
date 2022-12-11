#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 16:48:45 2022

@author: arun
"""
#%% Importing libraries
import sys
import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt
#%% Functions
def rescale(data):
    normalizedData = (data-np.min(data))/(np.max(data)-np.min(data))
    return normalizedData

def immultiply(img1, img2):
    # img1 = np.nan_to_num(img1,nan=0)
    # img2 = np.nan_to_num(img2,nan=0)
    # result_image = cv2.multiply(img1, img2)
    # result_image = np.nan_to_num(result_image)
    result_image = cv2.bitwise_and(img1,img2)
    return result_image

def strain_mask(I):
        
    R=rescale(I[:,:,2])
    th, Rm = cv2.threshold(R, 0.6, 1, cv2.THRESH_BINARY)
    
    G=rescale(I[:,:,1])
    th, Gm = cv2.threshold(G, 0.6, 1, cv2.THRESH_BINARY)
    
    B=rescale(I[:,:,0])
    th, Bm = cv2.threshold(B, 0.6, 1, cv2.THRESH_BINARY)
    
    
    I1 = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    H=rescale(I1[:,:,0])
    th, Hm = cv2.threshold(H, 0.55, 1, cv2.THRESH_BINARY)
    
    I2 = cv2.cvtColor(I, cv2.COLOR_BGR2Lab)
    B1=rescale(I2[:,:,2])
    th, Bm1=cv2.threshold(B1, 0.65, 1, cv2.THRESH_BINARY)
    
    I3=cv2.cvtColor(I, cv2.COLOR_BGR2YCR_CB)
     
    mask2=cv2.bitwise_not(Bm1)
    mask2=np.nan_to_num(mask2,nan=0)
    mask2=rescale(mask2)
    mask1=Hm
    # mask2=Bm1
    mask3=Bm
    mask4=Rm
    mask5=Gm
      
    mask12=immultiply(mask1,mask2)
    mask23=immultiply(mask2,mask3)
    mask45=immultiply(mask4,mask5)
    
    mask_A=immultiply(mask12,mask23)
    mask=immultiply(mask_A,mask45)
    
    mask = np.nan_to_num(mask,nan=0)
    mask = mask - 1
    mask = abs(mask)
    
    mask_1 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    # mask_1 = np.nan_to_num(mask_1)
    
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    erosion = cv2.erode(mask_1,kernel,iterations = 2)
    mask_2A = cv2.dilate(erosion,kernel,iterations = 2)
    
    mask_2_F=np.zeros(np.shape(I))
    mask_2_F[:,:,0]=mask_2A
    mask_2_F[:,:,1]=mask_2A
    mask_2_F[:,:,2]=mask_2A
    
    I_1=cv2.multiply(I,mask_2_F.astype(I.dtype))
    I_2=cv2.multiply(I1,mask_2_F.astype(I.dtype))
    I_3=cv2.multiply(I2,mask_2_F.astype(I.dtype))
    I_4=cv2.multiply(I3,mask_2_F.astype(I.dtype))
    
    return mask_2A, mask_2_F, I_1, I_2, I_3, I_4
#%% Reading image file

I=cv2.imread('/home/arun/Documents/PyWSPrecision/IGC/OS-3-cropped.tif')
siz=I.shape
#%% Brown stain masking by applying thresolding on different color space maps

mask_2, mask_2_F, I_1, I_2, I_3, I_4=strain_mask(I)

#%% Tile-wise process


final_mask=np.zeros_like(mask_2)
final_mask=np.uint8(final_mask)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
tile_step=20
step=siz[0]//tile_step
for i in range(0,siz[0],step):
    for j in range(0,siz[1],step):
        tileM=mask_2[i:i+step,j:j+step]
        tileM=np.uint8(tileM)
        
        tile2=I_2[i:i+step,j:j+step,:]
        tile_V=cv2.equalizeHist(tile2[:,:,2])
        tile_V.astype('uint8')
        tile_V=rescale(tile_V)
        tile_V[tile_V>0.7]=0
        tile_V[tile_V<0.4]=0
        
        ori_type=str(tile2.dtype)
        # tile_V=np.array(tile_V)
        tile_V=tile_V*255
        tile_V=np.uint8(tile_V)

        th, tile_Vm = cv2.threshold(tile_V, 0, 255, cv2.THRESH_OTSU)
        
        tile_Vm_res = cv2.morphologyEx(tile_Vm,cv2.MORPH_OPEN,kernel)
        
        tile_Vm_s = cv2.multiply(tileM,tile_Vm_res)
        
        final_mask[i:i+step,j:j+step]=rescale(tile_Vm_s)
        
mask_F=np.zeros_like(I)
mask_F[:,:,0]=final_mask
mask_F[:,:,1]=final_mask
mask_F[:,:,2]=final_mask

I_1_F=cv2.multiply(I_1,mask_F)
#%% Visualisation
# cv2.namedWindow("output1", cv2.WINDOW_NORMAL) 
# cv2.imshow("output1",mask1)

# cv2.namedWindow("output2", cv2.WINDOW_NORMAL) 
# cv2.imshow("output2",mask2)


# cv2.namedWindow("output3", cv2.WINDOW_NORMAL) 
# cv2.imshow("output3",mask3)

# cv2.namedWindow("output4", cv2.WINDOW_NORMAL) 
# cv2.imshow("output4",mask4)

# cv2.namedWindow("output5", cv2.WINDOW_NORMAL) 
# cv2.imshow("output5",mask5)


# cv2.namedWindow("output10", cv2.WINDOW_NORMAL) 
# cv2.imshow("output10",mask12)

# cv2.namedWindow("output20", cv2.WINDOW_NORMAL) 
# cv2.imshow("output20",mask23)

# cv2.namedWindow("output30", cv2.WINDOW_NORMAL) 
# cv2.imshow("output30",mask45)

cv2.namedWindow("output30", cv2.WINDOW_NORMAL) 
cv2.imshow("output30",I)
cv2.namedWindow("output40", cv2.WINDOW_NORMAL) 
cv2.imshow("output40",mask_2_F*255)
cv2.namedWindow("output60", cv2.WINDOW_NORMAL) 
cv2.imshow("output60",mask_F*255)
cv2.namedWindow("output50", cv2.WINDOW_NORMAL) 
cv2.imshow("output50",I_1)
cv2.namedWindow("output70", cv2.WINDOW_NORMAL) 
cv2.imshow("output70",I_1_F)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
#%%
# plt.figure(1),
# plt.imshow(mask_m)
# plt.show()

# plt.figure(2),
# plt.imshow(mask_2_F[::1],cmap='gray')
# plt.show()