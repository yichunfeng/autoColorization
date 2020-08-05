#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 23:06:12 2019

@author: yi-chun
"""

# 數學科學院 1801210118 馮逸群 Yi-Chun, Feng

from keras.layers import Conv2D,  UpSampling2D,BatchNormalization
from keras.layers import Activation, Dense, Dropout, Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt

sp=50000

path = "/Users/yi-chun/Downloads/PyCode/cifar_images/train/images_train_black/"
#path = "/home/notebooks/images_train_black/"
filenames=os.listdir(path)
filenames.sort()
filenames=filenames[0:sp]
data_number=len(filenames)

# color img
path2 = "/Users/yi-chun/Downloads/PyCode/cifar_images/train/images_train_color/"
#path2 = "/home/notebooks/images_train_color/"
filenames2=os.listdir(path2)
filenames2.sort()
filenames2=filenames2[0:sp]
data_number2=len(filenames2)



X = []
for filename in filenames:
    X.append(img_to_array(load_img(path+filename)))
X = np.array(X, dtype=float)
split = int(0.0001*len(X))
Xtrain = X[:split]
Xtrain = 1.0/255*X
#np.savetxt('Xtrain.txt',Xtrain)

Y = []
for filename2 in filenames2:
    Y.append(img_to_array(load_img(path2+filename2)))
Y = np.array(Y, dtype=float)
Ytrain = Y[:split]
Ytrain = 1.0/255*Y
#np.savetxt('Ytrain.txt',Ytrain)


#%%
"""
# Image transformer
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

# Generate training data
batch_size = 2
def image_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
    for batchY in datagen.flow(Ytrain, batch_size=batch_size):
        lab_batchY = rgb2lab(batchY)
        Y_batch = lab_batchY[:,:,:,1:] / 128
        yield (X_batch.reshape(X_batch.shape+(1,)),Y_batch)
"""  
X_batch = rgb2lab(Xtrain)[:,:,:,0]
X_batch=X_batch.reshape(X_batch.shape+(1,))
Y_batch = rgb2lab(Ytrain)[:,:,:,1:] / 128


#%%

model = Sequential()
model.add(InputLayer(input_shape=(32, 32,1)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
#model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
model.add(UpSampling2D((1, 1)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
#model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
model.add(UpSampling2D((1, 1)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
#model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
model.add(UpSampling2D((1, 1)))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
#model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
model.add(UpSampling2D((1, 1)))
model.compile(optimizer='rmsprop', loss='mse')

#%%

history=model.fit(x=X_batch, y=Y_batch, batch_size=2, epochs=1, verbose=1)
plt.figure(num=1, figsize=(8, 5))
plt.plot(history.history['loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
#plt.show()
plt.savefig("/Users/yi-chun/Downloads/PyCode/cifar_images/temp.png")
#plt.figure(num=2, figsize=(8, 5))
#plt.plot(history.history['acc'])
#plt.xlabel('epochs')
#plt.ylabel('accuracy')
        
#%%

#model.save('CNN1_model.h5')      
        
spt=4467

#model = load_model('CNN_model.h5')
path3 = "/Users/yi-chun/Downloads/PyCode/cifar_images/test/images_test_black/"
filenames3=os.listdir(path3)
filenames3.sort()
filenames3=filenames3[0:spt]
#data_number=len(filenames)
#print(filenames3)


C = []
for filename3 in filenames3:
    C.append(img_to_array(load_img(path3+filename3)))
C = np.array(C, dtype=float)
C= rgb2lab(1.0/255*C)[:,:,:,0]
C = C.reshape(C.shape+(1,))
output = model.predict(C)
output = output * 128

for i in range(len(output)):
    cur = np.zeros((32, 32, 3))
    cur[:,:,0] = C[i][:,:,0]
    cur[:,:,1:] = output[i]
    imsave("/Users/yi-chun/Downloads/PyCode/cifar_images/result/"+filenames3[i]+".png", lab2rgb(cur))
    #imsave("result/img_"+str(i)+".png", lab2rgb(cur))        

