# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:00:43 2019

@author: an_fab
"""
import os
import random
import configparser
import numpy as np
import matplotlib.pyplot as plt

from skimage import io, img_as_ubyte
from skimage.morphology import skeletonize

from GlaciersHelpers import write_hdf5

#read config file
config = configparser.RawConfigParser()
config.read('configuration.txt')
dirTrain = config.get('data paths','main_dir_train')

samplesPerImage = int(config.get('training settings','N_samples'))
patchSize = int(config.get('data attributes','patch_size'))

fileTrainSrc = config.get('file names','train_imgs_original')
fileTrainBw = config.get('file names','train_groundTruth')

dirPath = dirTrain + '\\src\\'
dirPathGT = dirTrain + '\\bw\\'

#Function for generating train data (patch-to-label manner)
def createDataFromDir2(samplesPerImage, patchSize):
    
    f = [] 
     
    for (dirpath, dirnames, filenames) in os.walk(dirPath):
        f.extend(filenames)
        break
     
    X_data = np.zeros((len(f)*samplesPerImage,patchSize,patchSize,3))
    Y_data = np.zeros((len(f)*samplesPerImage))
    
    iter_tot = 0
    
    for file in f:
        
        filePath = os.path.join(dirPath,file)
        filePathGT = os.path.join(dirPathGT,file)
        print(filePath)
        print(filePathGT)
        
        img = io.imread(filePath)
        img = img[:,:,0:3]
        

        Y = img.shape[0]
        X = img.shape[1]
        
        gt = io.imread(filePathGT)
        gt = gt > 0
        gt = img_as_ubyte(255*gt)
        gt[np.where(gt<255)] = 0
        gt[np.where(gt==255)] = 1
        #gt = skeletonize(gt)
        
        indxTrue = np.argwhere(gt==1)     # true varves  - label: 1
        np.random.shuffle(indxTrue)
        numT = indxTrue.shape[0]
                
        indxBkg = np.argwhere(gt<1)        # background - label: 0
        np.random.shuffle(indxBkg)       
        numB = indxBkg.shape[0]
        
        k = 0
        
        mt = 0
        mb = 0
        
        while k < samplesPerImage:
            
            ft = 0
            
            if k%2 == 1 and mt < numT:
                
                x = indxTrue[mt][1]
                y = indxTrue[mt][0]
                val = 1
                mt +=1
                ft = 1
                              
            if (k%2 == 0 and mb < numB) or (ft == 0):
            
                if mb < numB:
                    x = indxBkg[mb][1]
                    y = indxBkg[mb][0]
                    val = 0
                    mb +=1
                else:
                    m = random.randint(0,int(numB/2))
                    x = indxBkg[m][1]
                    y = indxBkg[m][0]
                    val = 0
                        
            x1 = x - patchSize
            x2 = x + patchSize
            y1 = y - patchSize
            y2 = y + patchSize
                 
            if y1 > 0 and y2 < Y and x1 > 0 and x2 < X:
                
                y_center = y
                x_center = x
                   
                patch = img[y_center-int(patchSize/2):y_center+int(patchSize/2),x_center-int(patchSize/2):x_center+int(patchSize/2),:]
                X_data[iter_tot] = patch
                Y_data[iter_tot] = val
                k+=1 
                iter_tot +=1   #total   

    X_data = X_data/255
    
#    s = X_data.mean()
#    m = X_data.std()
#    
#    X_data = (X_data - s)/m
    
    X_data = X_data[1:iter_tot,:,:,:]
    Y_data = Y_data[1:iter_tot]
    
    return X_data, Y_data 

# generate train data and save it    
X_train, Y_train = createDataFromDir2(samplesPerImage, patchSize)

#save data
print ("saving train datasets")
write_hdf5(np.asarray(X_train), fileTrainSrc)
write_hdf5(np.asarray(Y_train), fileTrainBw)


# display sample patches
for i in range (1,21):
    plt.subplot(4,5,i)
    idx = random.randint(0,X_train.shape[0])
    plt.tight_layout()
    plt.imshow(X_train[idx])
    plt.xticks([])
    plt.yticks([])
plt.show()