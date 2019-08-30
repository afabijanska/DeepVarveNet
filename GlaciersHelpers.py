# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 12:08:18 2019

@author: an_fab
"""
import h5py
import numpy as np

def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)
    

def extract_ordered_overlap(full_imgs, patch_h, patch_w,stride_h,stride_w):
    
    assert (len(full_imgs.shape)==4)  
    assert (full_imgs.shape[3]==1 or full_imgs.shape[3]==3)  
    img_h = full_imgs.shape[1] 
    img_w = full_imgs.shape[2] 
    assert ((img_h-patch_h)%stride_h==0 and (img_w-patch_w)%stride_w==0)
    N_patches_img = ((img_h-patch_h)//stride_h+1)*((img_w-patch_w)//stride_w+1)
    N_patches_tot = N_patches_img*full_imgs.shape[0]
    
    patches = np.empty((N_patches_tot,patch_h,patch_w,full_imgs.shape[3]), dtype = 'float16')
    iter_tot = 0   
    for i in range(full_imgs.shape[0]):  
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                patch = full_imgs[i,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w,:]
                patches[iter_tot]=patch
                iter_tot +=1   
    assert (iter_tot==N_patches_tot)
    return patches  

def paint_border_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    
    assert (len(full_imgs.shape)==4)  
    assert (full_imgs.shape[3]==1 or full_imgs.shape[3]==3)  
    img_h = full_imgs.shape[1] 
    img_w = full_imgs.shape[2] 
    leftover_h = (img_h-patch_h)%stride_h 
    leftover_w = (img_w-patch_w)%stride_w 
    if (leftover_h != 0):  
        
        tmp_full_imgs = np.zeros((full_imgs.shape[0],img_h+(stride_h-leftover_h),img_w,full_imgs.shape[3]))
        tmp_full_imgs[0:full_imgs.shape[0],0:img_h,0:img_w,0:full_imgs.shape[3]] = full_imgs
        full_imgs = tmp_full_imgs
    if (leftover_w != 0): 
        tmp_full_imgs = np.zeros((full_imgs.shape[0],full_imgs.shape[1],img_w+(stride_w - leftover_w),full_imgs.shape[3]))
        print (tmp_full_imgs.shape)
        tmp_full_imgs[0:full_imgs.shape[0],0:full_imgs.shape[1],0:img_w,0:full_imgs.shape[3]] = full_imgs
        full_imgs = tmp_full_imgs
        
    return full_imgs
#
def recompone_overlap(preds, img_h, img_w, stride_h, stride_w):
    
    assert (len(preds.shape)==4)  
    assert (preds.shape[3]==1 or preds.shape[3]==3)  
    patch_h = preds.shape[1]
    patch_w = preds.shape[2]
    N_patches_h = (img_h-patch_h)//stride_h+1
    N_patches_w = (img_w-patch_w)//stride_w+1
    N_patches_img = N_patches_h * N_patches_w

    assert (preds.shape[0]%N_patches_img==0)
    N_full_imgs = preds.shape[0]//N_patches_img
    
    full_prob = np.zeros((N_full_imgs,img_h,img_w,preds.shape[3])) 
    full_sum = np.zeros((N_full_imgs,img_h,img_w,preds.shape[3]))

    k = 0 
    for i in range(N_full_imgs):
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                full_prob[i,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w,:]+=preds[k]
                full_sum[i,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w,:]+=1
                k+=1
    assert(k==preds.shape[0])
    assert(np.min(full_sum)>=1.0) 
    final_avg = full_prob/full_sum

    return final_avg