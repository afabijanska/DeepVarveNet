# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 12:00:37 2017

@author: Ania
"""

import os
#import cv2
import numpy as np
import configparser
#import matplotlib.pyplot as plt
#from PIL import Image
from GlaciersHelpers import extract_ordered_overlap, paint_border_overlap
from keras.models import model_from_json
from skimage import io, exposure

##read config file
config = configparser.RawConfigParser()
config.read('configuration.txt')

main_dir_test = config.get('data paths','main_dir_test')
predictions_test = main_dir_test + '/preds/'
original_imgs_test = main_dir_test + '/src/'

json_path = config.get('file names','json_string')
model = model_from_json(open(json_path).read())
model.load_weights('best_weights_org_2.h5')

#data attributes
#get data attributes
patch_height = int(config.get('data attributes','patch_size'))
patch_width = int(config.get('data attributes','patch_size'))
stride_height = int(config.get('prediction settings','stride'))
stride_width = int(config.get('prediction settings','stride'))
assert (stride_height <= patch_height and stride_width <= patch_width)

#========= HELPER FUNCTION: pred_to_imgs

def pred_to_imgs(pred, patch_height, patch_width, mode="original"):
    #assert (len(pred.shape)==3)  #3D array: (Npatches,height*width,2)
    #assert (pred.shape[2]==2 )  #check the classes are 2
    
    pred_images = np.empty((pred.shape[0],patch_height*patch_width))  #(Npatches,height*width)
    
    if mode=="original":
        for i in range(patch_height*patch_width):
            for pix in range(pred.shape[1]):
                pred_images[i,pix]=pred[i,pix,1]
    elif mode=="threshold":
         for i in range(patch_height*patch_width):
            for pix in range(pred.shape[1]):
                if pred[i,pix,1]>=0.5:
                    pred_images[i,pix]=1
                else:
                    pred_images[i,pix]=0
    else:
        print ("mode " +str(mode) +" not recognized, it can be 'original' or 'threshold'")
        exit()
        
    pred_images = np.reshape(pred_images,(pred_images.shape[0], patch_height, patch_width,1))
    
    return pred_images

#========= FUNCTIONS ========================
    
def predict_img_rgb(org_path):
    
   org = io.imread(org_path)
   org = np.asarray(org, dtype='float16')
   org.shape
    
   print ('original image: ' + org_path)
    
   height = org.shape[0]
   width = org.shape[1]
   n_ch = org.shape[2]
    
   print ('image dims: (%d x %d x %d)' % (height, width, n_ch))
    
   org = np.reshape(org,(1, height, width, 3))
    
   org2 = paint_border_overlap(org, patch_height, patch_width, stride_height, stride_width)
   
   new_height = org2.shape[1]
   new_width = org2.shape[2]

   predImg = np.zeros((new_height, new_width))
   howMany = np.zeros((new_height, new_width))
   
   print ('new image dims: (%d x %d)' % (new_height, new_width))
   
   org2 = np.reshape(org2,(1, new_height, new_width, 3))
   assert(org2.shape == (1, new_height, new_width, 3))
   
   patches = extract_ordered_overlap(org2, patch_height, patch_width,stride_height,stride_width)
   print(patches.shape)
   
   predictions = model.predict(patches, batch_size=32, verbose=2)
   
   patchId = 0;
   
   for h in range(int((new_height-patch_height)/stride_height+1)):
       for w in range(int((new_width-patch_width)/stride_width+1)):
           patch = predictions[patchId,:,:,:]
           patch = patch.reshape((patch_height, patch_width))
           
           h1 = h*stride_height
           h2 = (h*stride_height)+patch_height
           w1 = w*stride_width
           w2 = (w*stride_width)+patch_width
           
           a = predImg[h1:h2,w1:w2]
           #predImg[h1:h2,w1:w2] = np.fmax(a,patch)
           #predImg[h1:h2,w1:w2] = np.mean([predImg[h1:h2,w1:w2],patch])
           predImg[h1:h2,w1:w2] = a + patch
           howMany[h1:h2,w1:w2] += 1
           patchId += 1
   
   predImg = predImg/howMany 
   predImg = predImg[0:height,0:width]        
   return predImg

#========= FUNCTIONS ========================

def predict_img_rgb_pix(org_path):
    
   org = io.imread(org_path)
   org = org[:,:,0:3]
   org = np.asarray(org, dtype='float16')
   org.shape
    
   print ('original image: ' + org_path)
    
   height = org.shape[0]
   width = org.shape[1]
   n_ch = org.shape[2]
    
   print ('image dims: (%d x %d x %d)' % (height, width, n_ch))
    
   org = np.reshape(org,(1, height, width, 3))
    
   org2 = paint_border_overlap(org, patch_height, patch_width, stride_height, stride_width)
   
   new_height = org2.shape[1]
   new_width = org2.shape[2]

   predImg = np.zeros((new_height, new_width))
   
   print ('new image dims: (%d x %d)' % (new_height, new_width))
   
   org2 = np.reshape(org2,(1, new_height, new_width, 3))
   assert(org2.shape == (1, new_height, new_width, 3))
   
   patches = extract_ordered_overlap(org2, patch_height, patch_width,stride_height,stride_width)
   patches = patches/255
   print(patches.shape)
   
   predictions = model.predict(patches)
   _max = np.max(predictions)
   _min = np.min(predictions)
   
   print('min: ')
   print(_min)
   print('max: ')
   print(_max)  
    #predictions = np.argmax(predictions,  axis = 1)
   
   #predImg = np.reshape(predictions, (new_height, new_width))
   
   patchId = 0;
   
   for h in range(int((new_height-patch_height)/stride_height+1)):
       for w in range(int((new_width-patch_width)/stride_width+1)):
           
           val = predictions[patchId,1]
           y = h + int(patch_height/2)
           x = w + int(patch_width/2)
           predImg[y,x] = val
           patchId += 1
   
   predImg = predImg[0:height,0:width]
   
   return predImg

#---------------------------------------------------------------------------------------------
#========= MAIN SCRIPT

for path, subdirs, files in os.walk(original_imgs_test):
    
    for i in range(len(files)):
        
        org_path = original_imgs_test + files[i]
        pred_path = predictions_test + files[i]
        
        print(org_path)
        #org = cv2.imread(org_path)
#        org2 = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
#        org2 = np.reshape(org2,(1, 1, org2.shape[0], org2.shape[1]))
#        org2 = paint_border_overlap(org2, patch_height, patch_width, stride_height, stride_width)
        
        #org2 = np.reshape(org2, (org2.shape[2], org2.shape[3]))   
        
        prediction = predict_img_rgb_pix(org_path)
        #prediction = np.reshape(prediction, (prediction.shape[1], prediction.shape[2]))
        scale = np.max(prediction)
        #prediction = exposure.equalize_adapthist(prediction)
        prediction = exposure.rescale_intensity(prediction)
        print('__________________')
        print(scale)
        print('__________________')
        #prediction = (255*(prediction/scale)).astype('uint8')
        prediction = (255*prediction).astype('uint8')
        print('__________________')
        scale = np.max(prediction)
        print(scale)
        print('__________________')
        io.imsave(pred_path,prediction)