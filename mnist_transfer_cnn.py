# -*- coding:utf-8 -*-

import datetime
import keras
from keras.datasets import mnist
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense

print(datetime.datetime.now())
input_shape = (28, 28, 1)
now = datetime.datetime.now


def train_model(model, train, test, num_classes):
    """"
    根据train,test训练模型model
    """
    x_train = train[0].reshape((train[0].shape[0],) + input_shape).astype('float32') / 255
    x_test = test[0].reshape((test[0].shape[0],) + input_shape).astype('float32') / 255

    y_train = keras.utils.to_categorical(train[1], num_classes)
    y_test = keras.utils.to_categorical(test[1], num_classes)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.adadelta(), metrics=['accuracy'])
    begin_time = now()
    model.fit(x_train, y_train, batch_size=128, epochs=10,
              verbose=1, validation_data=(x_test, y_test))

    print("Trainning time:", (now() - begin_time))
    score = model.evaluate(x_test, y_test, verbose=1)
    print(score)


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_before5 = x_train[y_train < 5]
y_train_before5 = y_train[y_train < 5]
x_test_before5 = x_test[y_test < 5]
y_test_before5 = y_test[y_test < 5]

x_train_after5 = x_train[y_train > 4]
y_train_after5 = y_train[y_train > 4] - 5
x_test_after5 = x_test[y_test > 4]
y_test_after5 = y_test[y_test > 4] - 5

# 定义两组网络层,卷积层和分类层
filters = 32
pool_size = 2
kernel_size = 3
feature_layers = [
    Conv2D(filters, kernel_size,
           padding='valid',
           input_shape=input_shape),
    Activation('relu'),
    Conv2D(filters, kernel_size),
    Activation('relu'),
    MaxPooling2D(pool_size=pool_size),
    Dropout(0.25),
    Flatten(),
]

classification_layers = [
    Dense(128),
    Activation('relu'),
    Dropout(0.5),
    Dense(5),
    Activation('softmax')
]

model = keras.Sequential(feature_layers + classification_layers)
model.summary()

train_model(model, (x_train_before5, y_train_before5),
            (x_test_before5, y_test_before5), 5)

# freeze feature layers and rebuild model
for l in feature_layers:
    l.trainable = False

model.summary()
train_model(model, (x_train_after5, y_train_after5),
            (x_test_after5, y_test_after5), 5)

# 30596/30596 [===] - 1s 39us/step - loss: 0.0124 - acc: 0.9959 - val_loss: 0.0053 - val_acc: 0.9982
# 29404/29404 [===] - 1s 25us/step - loss: 0.0289 - acc: 0.9910 - val_loss: 0.0215 - val_acc: 0.9926

"""
前一个model Non-trainable params: 0
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 26, 26, 32)        320       
_________________________________________________________________
activation_1 (Activation)    (None, 26, 26, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 24, 24, 32)        9248      
_________________________________________________________________
activation_2 (Activation)    (None, 24, 24, 32)        0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 12, 12, 32)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 12, 12, 32)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 4608)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               589952    
_________________________________________________________________
activation_3 (Activation)    (None, 128)               0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 5)                 645       
_________________________________________________________________
activation_4 (Activation)    (None, 5)                 0         
=================================================================
Total params: 600,165
Trainable params: 590,597
Non-trainable params: 9,568
_________________________________________________________________

"""