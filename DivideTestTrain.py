# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 21:55:47 2019

@author: an_fab
"""

import os
import configparser

from shutil import copyfile

#read configuration file

config = configparser.RawConfigParser()
config.read('configuration.txt')

main_dir = config.get('data paths','main_dir_all_data') # directory with all images
test_dir = config.get('data paths','main_dir_test')     # directory with train images
train_dir = config.get('data paths','main_dir_train')   # directory with test images

# define test / train split

counter = 0
fraction = 0.5                  #fraction of training data
step = int(1/fraction)

#read all data and divide it into train/test subsets

dirs = os.listdir(main_dir)

for d in dirs:
    
    print (os.path.join(main_dir, d))
    
    f = []
    
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(main_dir, d, 'src')):
        f.extend(filenames)
        break
    
    for files in f:
        
        print(os.path.join(main_dir, d, 'src',files))
        
        
        fileName = d + '_' + files
        
        if counter % step == 0:
            copyfile(os.path.join(main_dir, d, 'src',files), os.path.join(train_dir, 'src', fileName))
            copyfile(os.path.join(main_dir, d, 'bw',files), os.path.join(train_dir, 'bw', fileName))
            
        else:
            
            copyfile(os.path.join(main_dir, d, 'src',files), os.path.join(test_dir, 'src', fileName))
            copyfile(os.path.join(main_dir, d, 'bw',files), os.path.join(test_dir, 'bw', fileName))
                    
        counter = counter + 1