# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 21:55:47 2019

@author: an_fab
"""

import os

from skimage import io

main_dir = 'C:/Users/an_fab/Desktop/glaciers/all'
test_dir = 'C:/Users/an_fab/Desktop/glaciers/test'
train_dir = 'C:/Users/an_fab/Desktop/glaciers/train'

dirs = os.listdir(main_dir)

counter = 0
fraction = 0.5 #fraction training
step = int(1/fraction)

for d in dirs:
    
    print (os.path.join(main_dir, d))
    
    f = []
    
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(main_dir, d, 'src')):
        f.extend(filenames)
        break
    
    for files in f:
        
        print(os.path.join(main_dir, d, 'src',files))
        
        img = io.imread(os.path.join(main_dir, d, 'src',files))
        img = img.astype('uint8')
        bw = io.imread(os.path.join(main_dir, d, 'bw',files))
        bw = bw.astype('uint8')
        
        fileName = d + '_' + files
        #print(fileName)
        
        if counter % step == 0:
            io.imsave(os.path.join(train_dir, 'src',fileName), img)
            io.imsave(os.path.join(train_dir, 'bw',fileName), bw)
        else:
            io.imsave(os.path.join(test_dir, 'src',fileName), img)
            io.imsave(os.path.join(test_dir, 'bw',fileName), bw)
        
        counter = counter + 1