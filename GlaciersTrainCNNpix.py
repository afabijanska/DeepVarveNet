# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 13:53:49 2019

@author: an_fab
"""

import configparser

from helpers import load_hdf5 
from models import getSampleModel
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard

##read config file
config = configparser.RawConfigParser()
config.read('configuration.txt')

main_dir = config.get('data paths','main_dir')
train_imgs_original = config.get('file names','train_imgs_original')
train_groundTruth = config.get('file names','train_groundTruth')
json_path = config.get('file names','json_string')
weights_file_name = config.get('file names','weights_file_name')

#data attributes
#get data attributes
rows = int(config.get('data attributes','patch_size'))
cols = int(config.get('data attributes','patch_size'))
n_ch = int(config.get('data attributes','num_channels'))

#load training data
X_train = load_hdf5(main_dir + train_imgs_original)
Y_train = load_hdf5(main_dir + train_groundTruth)

Y_train = to_categorical(Y_train)

#read training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))

#define callbacks 
checkpointer = ModelCheckpoint(main_dir + weights_file_name, verbose=1, monitor='val_loss', mode='auto', save_best_only=True) 
tbCallback = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

#get model
model = getSampleModel(2, (rows,cols,n_ch))

json_string = model.to_json()
open(main_dir + json_path, 'w').write(json_string)

model.fit(x=X_train, y=Y_train, epochs=N_epochs, batch_size=batch_size, verbose=1, shuffle=True, validation_split=0.1, callbacks=[checkpointer,tbCallback])

model.save_weights('last_' + weights_file_name, overwrite=True)
print('Last weights saved')