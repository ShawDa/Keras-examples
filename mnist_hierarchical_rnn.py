# -*- coding:utf-8 -*-

import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed, LSTM

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')/255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

row, col, pixel = x_train.shape[1:]  # 28,28,1

x = Input(shape=(row, col, pixel))
encoded_rows = TimeDistributed(LSTM(128))(x)  # Encodes a row of pixels using TimeDistributed Wrapper.
encoded_cols = LSTM(128)(encoded_rows)  # Encodes columns of encoded rows.
prediction = Dense(10, activation='softmax')(encoded_cols)
model = Model(x, prediction)
model.summary(line_length=100)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.rmsprop(),
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32,
          epochs=10, verbose=1, validation_data=(x_test, y_test))

scores = model.evaluate(x_test, y_test, verbose=1)
print(scores)

# loss: 0.0287 - acc: 0.9911 - val_loss: 0.0395 - val_acc: 0.9887

"""
____________________________________________________________________________________________________
Layer (type)                                 Output Shape                            Param #
====================================================================================================
input_1 (InputLayer)                         (None, 28, 28, 1)                       0
____________________________________________________________________________________________________
time_distributed_1 (TimeDistributed)         (None, 28, 128)                         66560
____________________________________________________________________________________________________
lstm_2 (LSTM)                                (None, 128)                             131584
____________________________________________________________________________________________________
dense_1 (Dense)                              (None, 10)                              1290
====================================================================================================
Total params: 199,434
Trainable params: 199,434
Non-trainable params: 0
____________________________________________________________________________________________________
Train on 60000 samples, validate on 10000 samples
"""