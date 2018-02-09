# -*- coding:utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.datasets import *
from keras.losses import *
from keras.optimizers import *
import keras
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train)  # 60000*28*28
# print(y_train)  # 60000
# print(x_test)  # 10000*28*28
# print(y_test)  # 10000

x_train = x_train.reshape(60000, 784).astype('float32')
x_test = x_test.reshape(10000, 784).astype('float32')
x_train = x_train / 255
x_test = x_test / 255
# print(x_train)
# print(x_test)
# print(y_train[55])

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
print(y_train.shape)
print(y_test.shape)

model = Sequential()
model.add(Dense(512, activation='relu', input_dim=784))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=128, epochs=100, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=1)
print(score)

# loss: 0.0106 - acc: 0.9983 - val_loss: 0.1482 - val_acc: 0.9859

"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 512)               401920    
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 512)               262656    
_________________________________________________________________
dropout_2 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                5130      
=================================================================
Total params: 669,706
Trainable params: 669,706
Non-trainable params: 0
_________________________________________________________________
Train on 60000 samples, validate on 10000 samples
"""