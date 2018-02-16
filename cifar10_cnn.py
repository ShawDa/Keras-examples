# -*- coding:utf-8 -*-

from keras.models import Sequential
from keras.layers import *
from keras.datasets import *
import keras
from keras.preprocessing.image import ImageDataGenerator
import os

save_dir = os.path.join(os.getcwd(), 'models')
model_name = 'cifar10_cnn.h5'

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train)  # (50000,32,32,3)
# print(y_train)  # (50000, 1)
# print(x_test)  # (10000,32,32,3)
# print(y_test)  # (10000,1)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
x_train = x_train.astype('float')/255
x_test = x_test.astype('float')/255

model = Sequential()
model.add(Conv2D(32, 3, padding='same', input_shape=x_train.shape[1:], activation='relu'))
model.add(Conv2D(32, 3, activation='relu'))
model.add(MaxPooling2D())  # 默认pool_size=(2,2)
model.add(Dropout(0.25))

model.add(Conv2D(64, 3, padding='same', activation='relu'))
model.add(Conv2D(64, 3, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()
# exit(1)
opt = keras.optimizers.rmsprop(lr=1e-4, decay=1e-6)

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'])

#data_augmentation = True
data_augmentation = False
if not data_augmentation:
    print('Not using data augmentation')
    model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test), shuffle=True)
else:
    print('Using real-time data augmentation')
    datagen = ImageDataGenerator(
        featurewise_center=False,  # 使输入数据集去中心化（均值为0）, 按feature执行
        samplewise_center=False,  # 使输入数据的每个样本均值为0
        featurewise_std_normalization=False,  # 将输入除以数据集的标准差以完成标准化, 按feature执行
        samplewise_std_normalization=False,  # 将输入的每个样本除以其自身的标准差
        zca_whitening=False,  # 对输入数据施加ZCA白化
        rotation_range=0,  # 数据提升时图片随机转动的角度
        width_shift_range=0.1,  # 图片宽度的某个比例，数据提升时图片水平偏移的幅度
        height_shift_range=0.1,  # 图片高度的某个比例，数据提升时图片竖直偏移的幅度
        horizontal_flip=True,  # 进行随机水平翻转
        vertical_flip=False  # 进行随机竖直翻转
    )
    datagen.fit(x_train)
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=32), epochs=100, validation_data=(x_test, y_test), workers=4)  # 之前的keras必须要steps_per_epoch，workers：最大进程数

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)

scores = model.evaluate(x_test, y_test)
print(scores)

# data_augmentation = True  loss: 0.7764 - acc: 0.7456 - val_loss: 0.6924 - val_acc: 0.7673
# data_augmentation = False  loss: 0.6285 - acc: 0.7932 - val_loss: 0.7450 - val_acc: 0.7680

"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 32, 32, 32)        896
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 30, 30, 32)        9248
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 15, 15, 32)        0
_________________________________________________________________
dropout_1 (Dropout)          (None, 15, 15, 32)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 15, 15, 64)        18496
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 13, 13, 64)        36928
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 6, 6, 64)          0
_________________________________________________________________
dropout_2 (Dropout)          (None, 6, 6, 64)          0
_________________________________________________________________
flatten_1 (Flatten)          (None, 2304)              0
_________________________________________________________________
dense_1 (Dense)              (None, 512)               1180160
_________________________________________________________________
activation_1 (Activation)    (None, 512)               0
_________________________________________________________________
dropout_3 (Dropout)          (None, 512)               0
_________________________________________________________________
dense_2 (Dense)              (None, 10)                5130
_________________________________________________________________
activation_2 (Activation)    (None, 10)                0
=================================================================
Total params: 1,250,858
Trainable params: 1,250,858
Non-trainable params: 0
_________________________________________________________________
"""
