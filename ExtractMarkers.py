# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 20:46:47 2019

@author: an_fab
"""

import os
import numpy as np
import configparser

from skimage import io 
from skimage.filters import median

config = configparser.RawConfigParser()
config.read('configuration.txt')
main_dir = config.get('data paths','main_dir_all_data') # directory with all images

dirnames = os.listdir (main_dir + ".")

for directory in dirnames:
    
    print(directory)

    dir_path = main_dir + directory
    dir_markers = dir_path + '/gt/'
    dir_bw = dir_path + '/bw/'
    
    print ('Dir processed: ' + dir_path)
    print ('Dir markers: ' + dir_markers)
    print ('Dir b&w: ' + dir_bw)
        
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
    
        f = []
        f.extend(filenames)
     
        for file in f:
        
            filePath = os.path.join(dir_markers,file)
            print(filePath)
        
            img = io.imread(filePath)
            [Y,X,c] = img.shape
        
            R = img[:,:,0]
            G = img[:,:,1]
            B = img[:,:,2]
        
            labels = np.zeros((Y,X), dtype='uint8')
            indx = np.where(R>=252) and np.where(G<=3) and np.where(B<=3)  #varves
            labels[indx] = 255
   
            indx2 = np.where(R<=3) and np.where(G<=3) and np.where(B>=252)  #false varves
            labels[indx2] = 127
             
            labels = median(labels)
            
            filePath2 = os.path.join(dir_bw,file)
            io.imsave(filePath2,labels)