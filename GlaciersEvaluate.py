# -*- coding: utf-8 -*-
"""
Created on Sun May 19 17:29:36 2019

@author: an_fab
"""

import os
import configparser
import numpy as np

from skimage import io
from skimage.morphology import disk, skeletonize_3d, dilation, closing
from skimage.filters import median
from skimage.measure import label, regionprops

def PredsPostproc(img):
    
    [Y, X] = img.shape
    #img = median(img, disk(7))
    img = closing(img, disk(7))
    
    bw = np.zeros((Y,X), dtype = 'uint8')
    bw[np.where(img>=127)] = 1
    
    #bw = median(bw, disk(7))
    
    lab = label(bw)
    
    regions = regionprops(lab)
    
    for reg in regions:
        
        if reg.area <= 2000:
        
            indx = np.where(lab == reg.label)
            bw[indx] = 0
        
    bw = skeletonize_3d(bw)
    bw = dilation(bw, disk(2))
    
    return bw

config = configparser.RawConfigParser()
config.read('configuration.txt')

def PredsCompare(img, gt):
    
    [Y, X] = img.shape
    
    err_map = np.zeros((Y,X), dtype = 'uint8')
    
    tp = 0
    tp2 = 0
    fp = 0
    fn = 0
    
    lab_bw = label(gt)
    regions_bw = regionprops(lab_bw)
    
    for reg in regions_bw:
        
        temp = np.zeros((Y,X), dtype = 'uint8')
        temp[np.where(lab_bw==reg.label)] = 1
        
        temp = temp & img
    
        if np.sum(temp) > 0:
            tp = tp + 1
        else:
            fn = fn + 1
            err_map[np.where(lab_bw==reg.label)] = 1    #missed varve
    
    lab_pred = label(img)
    regions_img = regionprops(lab_pred)
    
    for reg in regions_img:
        
        temp = np.zeros((Y,X), dtype = 'uint8')
        temp[np.where(lab_pred==reg.label)] = 1
        
        temp = temp & gt
        
        if np.sum(temp) > 0:
            tp2 = tp2 + 1
        else:
            fp = fp + 1
            err_map[np.where(lab_pred==reg.label)] = 2 # false varve
        
    print(tp, tp2, fn, fp)
    
    return tp, fp, fn, err_map

#========= MAIN SCRIPT

main_dir_test = config.get('data paths','main_dir_test')
predictions_test = main_dir_test + '/preds/'
original_imgs_test = main_dir_test + '/src/'
final_test = main_dir_test + '/final/'
bw_test = main_dir_test + '/bw/'

TP = 0
FP = 0
FN = 0

for path, subdirs, files in os.walk(original_imgs_test):
    
    for i in range(len(files)):
        
        org_path = original_imgs_test + files[i]
        pred_path = predictions_test + files[i]
        fin_path = final_test + files[i]
        bw_path = bw_test + files[i]
        
        org = io.imread(org_path)
        gt = io.imread(bw_path)
        pred = io.imread(pred_path)
        pred = PredsPostproc(pred)

        print(org_path)
        
        tp, fp, fn, err_map = PredsCompare(pred, gt)
        
        TP = TP + tp
        FP = FP + fp
        FN = FN + fn
        
        ind = np.where(err_map == 1) #false varve
        org[:,:,0][ind] = 0
        org[:,:,1][ind] = 255
        org[:,:,2][ind] = 0
       
        ind = np.where(pred > 0)
        org[:,:,0][ind] = 255
        org[:,:,1][ind] = 0
        org[:,:,2][ind] = 0
        
        ind = np.where(err_map == 2) #missed varve
        org[:,:,0][ind] = 0
        org[:,:,1][ind] = 0
        org[:,:,2][ind] = 255
        
        io.imsave(fin_path, org)
        
print('True positives: ' + str(TP))
print('False negatives: ' + str(FN))
print('Varves found [%]: ' + str(TP/(FN+TP)))
print('False positives: ' + str(FP))

