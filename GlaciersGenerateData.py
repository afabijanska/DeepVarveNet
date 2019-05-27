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

from skimage import io
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

#Function for generating train data (patch-to-patch manner)
def createDataFromDir(samplesPerImage, patchSize):
    
    f = [] 
     
    for (dirpath, dirnames, filenames) in os.walk(dirPath):
        f.extend(filenames)
        break
     
    X_data = np.zeros((len(f)*samplesPerImage,patchSize,patchSize,3))
    Y_data = np.zeros((len(f)*samplesPerImage,patchSize,patchSize))
    
    iter_tot = 0
    
    for file in f:
        
        filePath = os.path.join(dirPath,file)
        filePathGT = os.path.join(dirPathGT,file)
        print(filePath)
        print(filePathGT)

        img = io.imread(filePath)
        gt = io.imread(filePathGT)
        gt = (gt/255).astype('uint8')
        gt = skeletonize(gt)
    
        Y = img.shape[0]
        X = img.shape[1]
        
        indx = np.argwhere(gt>0)
        
        np.random.shuffle(indx)
        np.random.shuffle(indx)
        np.random.shuffle(indx)
        num = indx.shape[0]
        
        k = 0
        
        while k < samplesPerImage:
                   
            for m in range(num):
                
                if k >= samplesPerImage:
                    break
                
                x = indx[m][1]
                y = indx[m][0]
                 
                x1 = x - patchSize
                x2 = x + patchSize
                y1 = y - patchSize
                y2 = y + patchSize
                 
                if y1 > 0 and y2 < Y and x1 > 0 and x2 < X:
                     
                    y_center = y
                    x_center = x
                   
                    patch = img[y_center-int(patchSize/2):y_center+int(patchSize/2),x_center-int(patchSize/2):x_center+int(patchSize/2),:]
                    patch_mask = gt[y_center-int(patchSize/2):y_center+int(patchSize/2),x_center-int(patchSize/2):x_center+int(patchSize/2)]
                    X_data[iter_tot] = patch
                    Y_data[iter_tot] = patch_mask
                    k+=1 
                    iter_tot +=1   #total   
                    
            if k < samplesPerImage:

                x_center = random.randint(0+int(patchSize/2),X-int(patchSize/2))
                y_center = random.randint(0+int(patchSize/2),Y-int(patchSize/2))
                
                patch = img[y_center-int(patchSize/2):y_center+int(patchSize/2),x_center-int(patchSize/2):x_center+int(patchSize/2),:]
                patch_mask = gt[y_center-int(patchSize/2):y_center+int(patchSize/2),x_center-int(patchSize/2):x_center+int(patchSize/2)]
                X_data[iter_tot]=patch
                Y_data[iter_tot]=patch_mask
                k+=1
                iter_tot +=1   #total    

    X_data = X_data/255
    Y_data = Y_data/255
    
    return X_data, Y_data 

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
        gt = io.imread(filePathGT)
        gt = (gt/255).astype('uint8')
        #gt = skeletonize(gt)
      
        Y = img.shape[0]
        X = img.shape[1]
        
        indxTrue = np.argwhere(gt>0.5)  #true varves only
        
        np.random.shuffle(indxTrue)
        np.random.shuffle(indxTrue)
        np.random.shuffle(indxTrue)
        numT = indxTrue.shape[0]
 
        indxFalse = np.argwhere(gt<=0.5) #background and false varves
        
        np.random.shuffle(indxFalse)
        np.random.shuffle(indxFalse)
        np.random.shuffle(indxFalse)
        numF = indxFalse.shape[0]
        
        k = 0
        val = 0
   
       
        while k < samplesPerImage:
            
            mf = 0
            mt = 0
            
            for m in range(numT+numF):
                
                if k >= samplesPerImage:
                    break
                
                if k%2 == 0 and mt < numT:
                    
                    x = indxTrue[mt][1]
                    y = indxTrue[mt][0]
                    val = 1
                    mt +=1
                
                if k%2 == 1 and mf < numF:
                    
                    x = indxFalse[mf][1]
                    y = indxFalse[mf][0]
                    val = 0
                    mf +=1
                
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
    
    return X_data, Y_data 

# generate train data and save it    
X_train, Y_train = createDataFromDir2(samplesPerImage, patchSize)

# display sample patches
for i in range (1,21):
    plt.subplot(4,5,i)
    idx = random.randint(0,25*samplesPerImage)
    plt.tight_layout()
    plt.imshow(X_train[idx])
    plt.xticks([])
    plt.yticks([])
plt.show()

#save data
print ("saving train datasets")
write_hdf5(np.asarray(X_train), fileTrainSrc)
write_hdf5(np.asarray(Y_train), fileTrainBw)
