# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 12:00:37 2017

@author: Ania
"""

import os
import numpy as np
import configparser
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

weights_file_name = config.get('file names','weights_file_name')
model.load_weights(weights_file_name)

#data attributes
#get data attributes
patch_height = int(config.get('data attributes','patch_size'))
patch_width = int(config.get('data attributes','patch_size'))
stride_height = int(config.get('prediction settings','stride'))
stride_width = int(config.get('prediction settings','stride'))
assert (stride_height <= patch_height and stride_width <= patch_width)

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
        
        prediction = predict_img_rgb_pix(org_path)
        scale = np.max(prediction)

        prediction = exposure.rescale_intensity(prediction)
        print('__________________')
        print(scale)
        print('__________________')

        prediction = (255*prediction).astype('uint8')
        print('__________________')
        scale = np.max(prediction)
        print(scale)
        print('__________________')
        io.imsave(pred_path,prediction)