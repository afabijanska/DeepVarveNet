# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 13:53:49 2019

@author: an_fab
"""

import configparser

from helpers import load_hdf5 
from models import getUnetOrg
from keras.callbacks import ModelCheckpoint, TensorBoard
from losses import ssim, focal_loss, jaccard_coef, jaccard_coef_int


##read config file
config = configparser.RawConfigParser()
config.read('configuration.txt')

main_dir = config.get('data paths','main_dir')
train_imgs_original = config.get('file names','train_imgs_original')
train_groundTruth = config.get('file names','train_groundTruth')
json_path = config.get('file names','json_string')

#data attributes
#get data attributes
rows = int(config.get('data attributes','patch_size'))
cols = int(config.get('data attributes','patch_size'))
n_ch = int(config.get('data attributes','num_channels'))

#load training data
X_train = load_hdf5(main_dir + train_imgs_original)
Y_train = load_hdf5(main_dir + train_groundTruth)

Y_train = Y_train.reshape((Y_train.shape[0], Y_train.shape[1], Y_train.shape[2], 1))

#read training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))

#define callbacks 
checkpointer = ModelCheckpoint(main_dir + 'best_weights_org.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True) 
tbCallback = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

#get model
model = getUnetOrg(rows,cols,n_ch)

json_string = model.to_json()
open(main_dir + json_path, 'w').write(json_string)

model.fit(x=X_train, y=Y_train, epochs=N_epochs, batch_size=batch_size, verbose=1, shuffle=True, validation_split=0.2, callbacks=[checkpointer,tbCallback])

model.save_weights('last_weights_org.h5', overwrite=True)
print('Last weights saved')

#kernel_initializer='he_normal'