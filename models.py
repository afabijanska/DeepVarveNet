# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 12:18:14 2019

@author: an_fab
"""

from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Flatten, Dense, BatchNormalization

def getSampleModel(numClasses, shape):
    
    inputs = Input(shape=shape)
    
    conv1 = Conv2D(filters=4, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer = 'he_normal', kernel_regularizer = l2(l = 0.001))(inputs)
    conv1 = Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer = 'he_normal', kernel_regularizer = l2(l = 0.001))(conv1)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(bn1)
    drop1 = Dropout(0.4)(pool1)
   
    conv2 = Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer = 'he_normal', kernel_regularizer = l2(l = 0.001))(drop1)
    conv2 = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer = 'he_normal', kernel_regularizer = l2(l = 0.001))(conv2)
    bn2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(bn2)
    drop2 = Dropout(0.4)(pool2)
   
    conv3 = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer = 'he_normal', kernel_regularizer = l2(l = 0.001))(drop2)
    conv3 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer = 'he_normal', kernel_regularizer = l2(l = 0.001))(conv3)
    bn3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(bn3)
    drop3 = Dropout(0.4)(pool3)

    conv4 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer = 'he_normal', kernel_regularizer = l2(l = 0.001))(drop3)
    conv4 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer = 'he_normal', kernel_regularizer = l2(l = 0.001))(conv4)
    bn4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(3,3))(bn4)
    drop4 = Dropout(0.4)(pool4)
    
    flat1 = Flatten()(drop4)
    dens1 = Dense(256, activation='relu')(flat1)
    dens2 = Dense(numClasses, activation = 'softmax')(dens1)
    
    model = Model(inputs=inputs, outputs=dens2)
    model.compile(optimizer=Adam(decay=0.0001),loss='categorical_crossentropy', metrics = ['accuracy'])
    model.summary()
    
    return model