# -*- coding:utf-8 -*-

"""Trains a ResNet on the CIFAR10 dataset.
ResNet v1
[a] Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf
cn:http://blog.csdn.net/wspba/article/details/57074389
ResNet v2
[b] Identity Mappings in Deep Residual Networks
https://arxiv.org/pdf/1603.05027.pdf
cn:http://blog.csdn.net/wspba/article/details/60750007
"""

from keras.layers import *
from keras.datasets import *
import keras
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau

# Training parameters
batch_size = 32  # orig paper trained all networks with batch_size=128
epochs = 200
data_augmentation = True
# data_augmentation = False
num_classes = 10

# Subtracting pixel mean improves accuracy 减去像素的平均值可以提高准确性
subtract_pixel_mean = True

# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
n = 3

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
# version = 1
version = 2

# Computed depth from supplied model parameter n
if version == 1:
    depth = n * 6 + 2  # 20
elif version == 2:
    depth = n * 9 + 2  # 29

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)
print(model_type)

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255  # (50000,32,32,3)
x_test = x_test.astype('float32') / 255  # (10000,32,32,3)

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)  # (32,32,3) 所有图片的均值
    # print(x_train_mean)
    x_train -= x_train_mean
    x_test -= x_train_mean

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)  # （50000,10）
y_test = keras.utils.to_categorical(y_test, num_classes)  # （10000，10）
# print(y_train)
# print(y_test)


def lr_schedule(epoch):  # 根据epoch更新学习率
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,  # 设置resnet层，免得重复调用写很多次
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            activation-bn-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=keras.regularizers.l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)  # 3

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)  # 权重初始化 He正态分布初始化方法，参数由0均值，标准差为sqrt(2 / fan_in) 的正态分布产生，其中fan_in权重张量的扇入

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)  # 3

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


if version == 2:
    model = resnet_v2(input_shape=input_shape, depth=depth)
else:
    model = resnet_v1(input_shape=input_shape, depth=depth)

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()
# exit(1)
# print(model_type)

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type  # 用这个名字保存模型
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)  # 该回调函数是学习率调度器

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),  # 当评价指标不在提升时，减少学习率
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

# Run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        validation_data=(x_test, y_test),
                        epochs=epochs, verbose=1, workers=4,
                        callbacks=callbacks)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print(scores)

# V1  loss: 0.1550 - acc: 0.9882 - val_loss: 0.4335 - val_acc: 0.9172  Epoch 00151: val_acc improved from 0.91870 to 0.91880
# V2  loss: 0.1686 - acc: 0.9863 - val_loss: 0.4236 - val_acc: 0.9211  Epoch 00151: val_acc improved from 0.92160 to 0.92210

"""  V1
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, 32, 32, 3)    0
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 32, 32, 16)   448         input_1[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 32, 32, 16)   64          conv2d_1[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 32, 32, 16)   0           batch_normalization_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 32, 32, 16)   2320        activation_1[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 32, 32, 16)   64          conv2d_2[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 32, 32, 16)   0           batch_normalization_2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 32, 32, 16)   2320        activation_2[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 32, 32, 16)   64          conv2d_3[0][0]
__________________________________________________________________________________________________
add_1 (Add)                     (None, 32, 32, 16)   0           activation_1[0][0]
                                                                 batch_normalization_3[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 32, 32, 16)   0           add_1[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 32, 32, 16)   2320        activation_3[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 32, 32, 16)   64          conv2d_4[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 32, 32, 16)   0           batch_normalization_4[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 32, 32, 16)   2320        activation_4[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 32, 32, 16)   64          conv2d_5[0][0]
__________________________________________________________________________________________________
add_2 (Add)                     (None, 32, 32, 16)   0           activation_3[0][0]
                                                                 batch_normalization_5[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 32, 32, 16)   0           add_2[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 32, 32, 16)   2320        activation_5[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 32, 32, 16)   64          conv2d_6[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 32, 32, 16)   0           batch_normalization_6[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 32, 32, 16)   2320        activation_6[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 32, 32, 16)   64          conv2d_7[0][0]
__________________________________________________________________________________________________
add_3 (Add)                     (None, 32, 32, 16)   0           activation_5[0][0]
                                                                 batch_normalization_7[0][0]
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 32, 32, 16)   0           add_3[0][0]
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 16, 16, 32)   4640        activation_7[0][0]
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 16, 16, 32)   128         conv2d_8[0][0]
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 16, 16, 32)   0           batch_normalization_8[0][0]
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 16, 16, 32)   9248        activation_8[0][0]
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 16, 16, 32)   544         activation_7[0][0]
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 16, 16, 32)   128         conv2d_9[0][0]
__________________________________________________________________________________________________
add_4 (Add)                     (None, 16, 16, 32)   0           conv2d_10[0][0]
                                                                 batch_normalization_9[0][0]
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 16, 16, 32)   0           add_4[0][0]
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 16, 16, 32)   9248        activation_9[0][0]
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 16, 16, 32)   128         conv2d_11[0][0]
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 16, 16, 32)   0           batch_normalization_10[0][0]
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 16, 16, 32)   9248        activation_10[0][0]
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 16, 16, 32)   128         conv2d_12[0][0]
__________________________________________________________________________________________________
add_5 (Add)                     (None, 16, 16, 32)   0           activation_9[0][0]
                                                                 batch_normalization_11[0][0]
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 16, 16, 32)   0           add_5[0][0]
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 16, 16, 32)   9248        activation_11[0][0]
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 16, 16, 32)   128         conv2d_13[0][0]
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 16, 16, 32)   0           batch_normalization_12[0][0]
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 16, 16, 32)   9248        activation_12[0][0]
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 16, 16, 32)   128         conv2d_14[0][0]
__________________________________________________________________________________________________
add_6 (Add)                     (None, 16, 16, 32)   0           activation_11[0][0]
                                                                 batch_normalization_13[0][0]
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 16, 16, 32)   0           add_6[0][0]
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 8, 8, 64)     18496       activation_13[0][0]
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 8, 8, 64)     256         conv2d_15[0][0]
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 8, 8, 64)     0           batch_normalization_14[0][0]
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 8, 8, 64)     36928       activation_14[0][0]
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 8, 8, 64)     2112        activation_13[0][0]
__________________________________________________________________________________________________
batch_normalization_15 (BatchNo (None, 8, 8, 64)     256         conv2d_16[0][0]
__________________________________________________________________________________________________
add_7 (Add)                     (None, 8, 8, 64)     0           conv2d_17[0][0]
                                                                 batch_normalization_15[0][0]
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 8, 8, 64)     0           add_7[0][0]
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 8, 8, 64)     36928       activation_15[0][0]
__________________________________________________________________________________________________
batch_normalization_16 (BatchNo (None, 8, 8, 64)     256         conv2d_18[0][0]
__________________________________________________________________________________________________
activation_16 (Activation)      (None, 8, 8, 64)     0           batch_normalization_16[0][0]
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 8, 8, 64)     36928       activation_16[0][0]
__________________________________________________________________________________________________
batch_normalization_17 (BatchNo (None, 8, 8, 64)     256         conv2d_19[0][0]
__________________________________________________________________________________________________
add_8 (Add)                     (None, 8, 8, 64)     0           activation_15[0][0]
                                                                 batch_normalization_17[0][0]
__________________________________________________________________________________________________
activation_17 (Activation)      (None, 8, 8, 64)     0           add_8[0][0]
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 8, 8, 64)     36928       activation_17[0][0]
__________________________________________________________________________________________________
batch_normalization_18 (BatchNo (None, 8, 8, 64)     256         conv2d_20[0][0]
__________________________________________________________________________________________________
activation_18 (Activation)      (None, 8, 8, 64)     0           batch_normalization_18[0][0]
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 8, 8, 64)     36928       activation_18[0][0]
__________________________________________________________________________________________________
batch_normalization_19 (BatchNo (None, 8, 8, 64)     256         conv2d_21[0][0]
__________________________________________________________________________________________________
add_9 (Add)                     (None, 8, 8, 64)     0           activation_17[0][0]
                                                                 batch_normalization_19[0][0]
__________________________________________________________________________________________________
activation_19 (Activation)      (None, 8, 8, 64)     0           add_9[0][0]
__________________________________________________________________________________________________
average_pooling2d_1 (AveragePoo (None, 1, 1, 64)     0           activation_19[0][0]
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 64)           0           average_pooling2d_1[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 10)           650         flatten_1[0][0]
==================================================================================================
Total params: 274,442
Trainable params: 273,066
Non-trainable params: 1,376
__________________________________________________________________________________________________
"""

"""  V2
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, 32, 32, 3)    0
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 32, 32, 16)   448         input_1[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 32, 32, 16)   64          conv2d_1[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 32, 32, 16)   0           batch_normalization_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 32, 32, 16)   272         activation_1[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 32, 32, 16)   64          conv2d_2[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 32, 32, 16)   0           batch_normalization_2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 32, 32, 16)   2320        activation_2[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 32, 32, 16)   64          conv2d_3[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 32, 32, 16)   0           batch_normalization_3[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 32, 32, 64)   1088        activation_1[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 32, 32, 64)   1088        activation_3[0][0]
__________________________________________________________________________________________________
add_1 (Add)                     (None, 32, 32, 64)   0           conv2d_5[0][0]
                                                                 conv2d_4[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 32, 32, 64)   256         add_1[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 32, 32, 64)   0           batch_normalization_4[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 32, 32, 16)   1040        activation_4[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 32, 32, 16)   64          conv2d_6[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 32, 32, 16)   0           batch_normalization_5[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 32, 32, 16)   2320        activation_5[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 32, 32, 16)   64          conv2d_7[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 32, 32, 16)   0           batch_normalization_6[0][0]
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 32, 32, 64)   1088        activation_6[0][0]
__________________________________________________________________________________________________
add_2 (Add)                     (None, 32, 32, 64)   0           add_1[0][0]
                                                                 conv2d_8[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 32, 32, 64)   256         add_2[0][0]
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 32, 32, 64)   0           batch_normalization_7[0][0]
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 32, 32, 16)   1040        activation_7[0][0]
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 32, 32, 16)   64          conv2d_9[0][0]
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 32, 32, 16)   0           batch_normalization_8[0][0]
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 32, 32, 16)   2320        activation_8[0][0]
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 32, 32, 16)   64          conv2d_10[0][0]
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 32, 32, 16)   0           batch_normalization_9[0][0]
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 32, 32, 64)   1088        activation_9[0][0]
__________________________________________________________________________________________________
add_3 (Add)                     (None, 32, 32, 64)   0           add_2[0][0]
                                                                 conv2d_11[0][0]
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 32, 32, 64)   256         add_3[0][0]
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 32, 32, 64)   0           batch_normalization_10[0][0]
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 16, 16, 64)   4160        activation_10[0][0]
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 16, 16, 64)   256         conv2d_12[0][0]
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 16, 16, 64)   0           batch_normalization_11[0][0]
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 16, 16, 64)   36928       activation_11[0][0]
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 16, 16, 64)   256         conv2d_13[0][0]
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 16, 16, 64)   0           batch_normalization_12[0][0]
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 16, 16, 128)  8320        add_3[0][0]
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 16, 16, 128)  8320        activation_12[0][0]
__________________________________________________________________________________________________
add_4 (Add)                     (None, 16, 16, 128)  0           conv2d_15[0][0]
                                                                 conv2d_14[0][0]
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 16, 16, 128)  512         add_4[0][0]
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 16, 16, 128)  0           batch_normalization_13[0][0]
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 16, 16, 64)   8256        activation_13[0][0]
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 16, 16, 64)   256         conv2d_16[0][0]
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 16, 16, 64)   0           batch_normalization_14[0][0]
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 16, 16, 64)   36928       activation_14[0][0]
__________________________________________________________________________________________________
batch_normalization_15 (BatchNo (None, 16, 16, 64)   256         conv2d_17[0][0]
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 16, 16, 64)   0           batch_normalization_15[0][0]
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 16, 16, 128)  8320        activation_15[0][0]
__________________________________________________________________________________________________
add_5 (Add)                     (None, 16, 16, 128)  0           add_4[0][0]
                                                                 conv2d_18[0][0]
__________________________________________________________________________________________________
batch_normalization_16 (BatchNo (None, 16, 16, 128)  512         add_5[0][0]
__________________________________________________________________________________________________
activation_16 (Activation)      (None, 16, 16, 128)  0           batch_normalization_16[0][0]
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 16, 16, 64)   8256        activation_16[0][0]
__________________________________________________________________________________________________
batch_normalization_17 (BatchNo (None, 16, 16, 64)   256         conv2d_19[0][0]
__________________________________________________________________________________________________
activation_17 (Activation)      (None, 16, 16, 64)   0           batch_normalization_17[0][0]
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 16, 16, 64)   36928       activation_17[0][0]
__________________________________________________________________________________________________
batch_normalization_18 (BatchNo (None, 16, 16, 64)   256         conv2d_20[0][0]
__________________________________________________________________________________________________
activation_18 (Activation)      (None, 16, 16, 64)   0           batch_normalization_18[0][0]
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 16, 16, 128)  8320        activation_18[0][0]
__________________________________________________________________________________________________
add_6 (Add)                     (None, 16, 16, 128)  0           add_5[0][0]
                                                                 conv2d_21[0][0]
__________________________________________________________________________________________________
batch_normalization_19 (BatchNo (None, 16, 16, 128)  512         add_6[0][0]
__________________________________________________________________________________________________
activation_19 (Activation)      (None, 16, 16, 128)  0           batch_normalization_19[0][0]
__________________________________________________________________________________________________
conv2d_22 (Conv2D)              (None, 8, 8, 128)    16512       activation_19[0][0]
__________________________________________________________________________________________________
batch_normalization_20 (BatchNo (None, 8, 8, 128)    512         conv2d_22[0][0]
__________________________________________________________________________________________________
activation_20 (Activation)      (None, 8, 8, 128)    0           batch_normalization_20[0][0]
__________________________________________________________________________________________________
conv2d_23 (Conv2D)              (None, 8, 8, 128)    147584      activation_20[0][0]
__________________________________________________________________________________________________
batch_normalization_21 (BatchNo (None, 8, 8, 128)    512         conv2d_23[0][0]
__________________________________________________________________________________________________
activation_21 (Activation)      (None, 8, 8, 128)    0           batch_normalization_21[0][0]
__________________________________________________________________________________________________
conv2d_25 (Conv2D)              (None, 8, 8, 256)    33024       add_6[0][0]
__________________________________________________________________________________________________
conv2d_24 (Conv2D)              (None, 8, 8, 256)    33024       activation_21[0][0]
__________________________________________________________________________________________________
add_7 (Add)                     (None, 8, 8, 256)    0           conv2d_25[0][0]
                                                                 conv2d_24[0][0]
__________________________________________________________________________________________________
batch_normalization_22 (BatchNo (None, 8, 8, 256)    1024        add_7[0][0]
__________________________________________________________________________________________________
activation_22 (Activation)      (None, 8, 8, 256)    0           batch_normalization_22[0][0]
__________________________________________________________________________________________________
conv2d_26 (Conv2D)              (None, 8, 8, 128)    32896       activation_22[0][0]
__________________________________________________________________________________________________
batch_normalization_23 (BatchNo (None, 8, 8, 128)    512         conv2d_26[0][0]
__________________________________________________________________________________________________
activation_23 (Activation)      (None, 8, 8, 128)    0           batch_normalization_23[0][0]
__________________________________________________________________________________________________
conv2d_27 (Conv2D)              (None, 8, 8, 128)    147584      activation_23[0][0]
__________________________________________________________________________________________________
batch_normalization_24 (BatchNo (None, 8, 8, 128)    512         conv2d_27[0][0]
__________________________________________________________________________________________________
activation_24 (Activation)      (None, 8, 8, 128)    0           batch_normalization_24[0][0]
__________________________________________________________________________________________________
conv2d_28 (Conv2D)              (None, 8, 8, 256)    33024       activation_24[0][0]
__________________________________________________________________________________________________
add_8 (Add)                     (None, 8, 8, 256)    0           add_7[0][0]
                                                                 conv2d_28[0][0]
__________________________________________________________________________________________________
batch_normalization_25 (BatchNo (None, 8, 8, 256)    1024        add_8[0][0]
__________________________________________________________________________________________________
activation_25 (Activation)      (None, 8, 8, 256)    0           batch_normalization_25[0][0]
__________________________________________________________________________________________________
conv2d_29 (Conv2D)              (None, 8, 8, 128)    32896       activation_25[0][0]
__________________________________________________________________________________________________
batch_normalization_26 (BatchNo (None, 8, 8, 128)    512         conv2d_29[0][0]
__________________________________________________________________________________________________
activation_26 (Activation)      (None, 8, 8, 128)    0           batch_normalization_26[0][0]
__________________________________________________________________________________________________
conv2d_30 (Conv2D)              (None, 8, 8, 128)    147584      activation_26[0][0]
__________________________________________________________________________________________________
batch_normalization_27 (BatchNo (None, 8, 8, 128)    512         conv2d_30[0][0]
__________________________________________________________________________________________________
activation_27 (Activation)      (None, 8, 8, 128)    0           batch_normalization_27[0][0]
__________________________________________________________________________________________________
conv2d_31 (Conv2D)              (None, 8, 8, 256)    33024       activation_27[0][0]
__________________________________________________________________________________________________
add_9 (Add)                     (None, 8, 8, 256)    0           add_8[0][0]
                                                                 conv2d_31[0][0]
__________________________________________________________________________________________________
batch_normalization_28 (BatchNo (None, 8, 8, 256)    1024        add_9[0][0]
__________________________________________________________________________________________________
activation_28 (Activation)      (None, 8, 8, 256)    0           batch_normalization_28[0][0]
__________________________________________________________________________________________________
average_pooling2d_1 (AveragePoo (None, 1, 1, 256)    0           activation_28[0][0]
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 256)          0           average_pooling2d_1[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 10)           2570        flatten_1[0][0]
==================================================================================================
Total params: 849,002
Trainable params: 843,786
Non-trainable params: 5,216
__________________________________________________________________________________________________
"""