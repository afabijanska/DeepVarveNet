# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 12:18:14 2019

@author: an_fab
"""

from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose,core, Flatten, Dense, BatchNormalization

from losses import focal_loss, jaccard_coef,jaccard_coef_int

def getSampleModel(numClasses, shape):
    
    inputs = Input(shape=shape)
    
    conv1 = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer = l2(l = 0.001))(inputs)
    #conv1 = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer = l2(l = 0.001))(conv1)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(bn1)
    drop1 = Dropout(0.4)(pool1)
   
    conv2 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer = l2(l = 0.001))(drop1)
    #conv2 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer = l2(l = 0.001))(conv2)
    bn2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(bn2)
    drop2 = Dropout(0.4)(pool2)
   
    conv3 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer = l2(l = 0.001))(drop2)
    #conv3 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer = l2(l = 0.001))(conv3)
    bn3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(bn3)
    drop3 = Dropout(0.4)(pool3)

    conv4 = Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer = l2(l = 0.001))(drop3)
    #conv4 = Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer = l2(l = 0.001))(conv4)
    bn4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2,2))(bn4)
    drop4 = Dropout(0.4)(pool4)
    
    flat1 = Flatten()(drop4)
    dens1 = Dense(256, activation='relu')(flat1)
    dens2 = Dense(numClasses, activation = 'softmax')(dens1)
    
    model = Model(inputs=inputs, outputs=dens2)
    model.compile(optimizer=Adam(decay=0.0001),loss='categorical_crossentropy', metrics = ['accuracy'])
    #model.compile(optimizer='adam',loss='mse', metrics = ['accuracy'])
    model.summary()
    
    return model

def getSampleModel2(numClasses, shape):
    
    inputs = Input(shape=shape)
    
    conv1 = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer = l2(l = 0.001))(inputs)
    #conv1 = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer = l2(l = 0.001))(conv1)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(bn1)
    drop1 = Dropout(0.4)(pool1)
   
    conv2 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer = l2(l = 0.001))(drop1)
    #conv2 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer = l2(l = 0.001))(conv2)
    bn2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(bn2)
    drop2 = Dropout(0.4)(pool2)
   
    conv3 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer = l2(l = 0.001))(drop2)
    #conv3 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer = l2(l = 0.001))(conv3)
    bn3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(bn3)
    drop3 = Dropout(0.4)(pool3)

    conv4 = Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer = l2(l = 0.001))(drop3)
    #conv4 = Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer = l2(l = 0.001))(conv4)
    bn4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2,2))(bn4)
    drop4 = Dropout(0.4)(pool4)
    
    flat1 = Flatten()(drop4)
    dens1 = Dense(256, activation='relu')(flat1)
    dens2 = Dense(numClasses, activation = 'softmax')(dens1)
    
    model = Model(inputs=inputs, outputs=dens2)
    model.compile(optimizer=Adam(decay=0.0001),loss='categorical_crossentropy', metrics = ['accuracy'])
    #model.compile(optimizer='adam',loss='mse', metrics = ['accuracy'])
    model.summary()
    
    return model

#define model
def get_unet(rows,cols, n_ch):
    #
    inputs = Input((rows, cols, n_ch))
    #
    conv1 = Conv2D(4, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(4, (3, 3), activation='relu', padding='same')(conv1)
    #
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #
    conv2 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv2)
    #
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv3)
    #
    up1 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=-1)
    #
    conv4 = Conv2D(8, (3, 3), activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv4)
    #
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=-1)
    #
    conv5 = Conv2D(4, (3, 3), activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(4, (3, 3), activation='relu', padding='same')(conv5)
    #
    #conv6 = Conv2D(2, (3, 3), activation='relu', padding='same')(conv5)
    #conv6 = core.Reshape((n_ch,rows*cols))(conv6)
    #conv6 = core.Permute((2,1))(conv6)
    #
    conv7 = Conv2D(1, (1, 1), activation='sigmoid')(conv5)
    model = Model(inputs=inputs, outputs=conv7)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    #model.compile(optimizer='adam', loss = 'mse', metrics=['accuracy'])
    model.compile(optimizer='adam',loss=focal_loss(gamma=2., alpha=.25), metrics = ['accuracy'])
    model.summary()

    return model

def get_unet2(rows,cols, n_ch):
    #
    inputs = Input((rows, cols, n_ch))
    #
    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal', kernel_regularizer = l2(1e-4))(inputs)
    conv1 = Dropout(0.4)(conv1)
    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal', kernel_regularizer = l2(1e-4))(conv1)
    #
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #
    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal', kernel_regularizer = l2(1e-4))(pool1)
    conv2 = Dropout(0.4)(conv2)
    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal', kernel_regularizer = l2(1e-4))(conv2)
    #
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal', kernel_regularizer = l2(1e-4))(pool2)
    conv3 = Dropout(0.4)(conv3)
    conv3 = Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer = 'he_normal', kernel_regularizer = l2(1e-4))(conv3)
    #
    up1 = concatenate([conv3, conv2], axis=-1)
    #
    conv4 = Conv2DTranspose(16, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal', kernel_regularizer = l2(1e-4))(up1)
    conv4 = Dropout(0.4)(conv4)
    conv4 = Conv2DTranspose(16, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer = 'he_normal', kernel_regularizer = l2(1e-4))(conv4)
    #
    up2 = concatenate([conv4, conv1], axis=-1)
    #
    conv5 = Conv2DTranspose(8, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal', kernel_regularizer = l2(1e-4))(up2)
    conv5 = Dropout(0.4)(conv5)
    conv5 = Conv2DTranspose(8, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal', kernel_regularizer = l2(1e-4))(conv5)
    
#    conv6 = Conv2D(2, (3, 3), activation='relu', padding='same')(conv5)
#    conv6 = core.Reshape((2,rows*cols))(conv6)
#    conv6 = core.Permute((2,1))(conv6)
    #
#    conv7 = core.Activation('softmax')(conv6)
#    model = Model(input=inputs, output=conv7)
    
    conv7 = Conv2D(1, (1, 1), activation='sigmoid')(conv5)
    model = Model(inputs=inputs, outputs=conv7)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='adam', loss='mse', metrics=[jaccard_coef,jaccard_coef_int, 'accuracy'])
    #model.compile(optimizer='adam',loss=focal_loss(gamma=2., alpha=.25), metrics = ['accuracy'])
    model.summary()

    return model

def get_unet3(img_rows,img_cols, n_ch):
    
    inputs = Input((img_rows, img_cols, n_ch))
    
    conv1 = Conv2D(64, kernel_size=(3, 3), activation = 'relu', padding = 'same')(inputs)
    conv1 = Conv2D(64, kernel_size=(3, 3), activation = 'relu', padding = 'same')(conv1)
    print (conv1.shape)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print (pool1.shape)
    conv2 = Conv2D(128, kernel_size=(3, 3), activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv2D(128, kernel_size=(3, 3), activation = 'relu', padding = 'same')(conv2)
    print (conv2.shape)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print (pool2.shape)
    conv3 = Conv2D(256, kernel_size=(3, 3), activation = 'relu', padding = 'same')(pool2)
    conv3 = Conv2D(256, kernel_size=(3, 3), activation = 'relu', padding = 'same')(conv3)
    print (conv3.shape)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    print (pool3.shape)
    conv4 = Conv2D(512, kernel_size=(3, 3), activation = 'relu', padding = 'same')(pool3)
    conv4 = Conv2D(512, kernel_size=(3, 3), activation = 'relu', padding = 'same')(conv4)
    print (conv4.shape)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), padding = 'same')(drop4)
    print (pool4.shape)
    conv5 = Conv2D(1024, kernel_size=(3, 3), activation = 'relu', padding = 'same')(pool4)
    conv5 = Conv2D(1024, kernel_size=(3, 3), activation = 'relu', padding = 'same')(conv5)
    print (conv5.shape)
    drop5 = Dropout(0.5)(conv5)
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(drop5))
    #up6 = Cropping2D(cropping = ((1,0),(0,0)))(up6)
    print (up6.shape)
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, kernel_size=(3, 3), activation = 'relu', padding = 'same')(merge6)
    conv6 = Conv2D(512, kernel_size=(3, 3), activation = 'relu', padding = 'same')(conv6)
    print (conv6.shape)
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv6))
    print (up7.shape)
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, kernel_size=(3, 3), activation = 'relu', padding = 'same')(merge7)
    conv7 = Conv2D(256, kernel_size=(3, 3), activation = 'relu', padding = 'same')(conv7)
    print (conv7.shape)
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv7))
    print (up8.shape)
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, kernel_size=(3, 3), activation = 'relu', padding = 'same')(merge8)
    conv8 = Conv2D(128, kernel_size=(3, 3), activation = 'relu', padding = 'same')(conv8)
    print (conv8.shape)
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv8))
    print (up9.shape)
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, kernel_size=(3, 3), activation = 'relu', padding = 'same')(merge9)
    conv9 = Conv2D(64, kernel_size=(3, 3), activation = 'relu', padding = 'same')(conv9)
    conv9 = Conv2D(2, kernel_size=(3, 3), activation = 'relu', padding = 'same')(conv9)
    print (conv9.shape)
    reshape = core.Reshape((2, img_rows * img_cols), input_shape = (2, img_rows, img_cols))(conv9)
    permute = core.Permute((2,1))(reshape)
    activation = core.Activation('softmax')(permute)
    model = Model(input = inputs, output = activation)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'mse', metrics = ['accuracy'])
    model.summary()

    return model

def getUnetOrg(img_rows,img_cols, n_ch):
    
    inputs = Input((img_rows,img_cols, n_ch))
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    model.summary()

    return model
