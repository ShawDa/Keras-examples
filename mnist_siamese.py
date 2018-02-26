# -*- coding:utf-8 -*-

from keras.datasets import mnist
import numpy as np
import random
import keras
from keras import backend as K

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train)  # 60000*28*28
# print(y_train)  # 60000
# print(x_test)  # 10000*28*28
# print(y_test)  # 10000
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
input_shape = x_train.shape[1:]


def create_pairs(x, digit_indices):  # pairs举一个正例和反例，labels为1 0 1 0 ...
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    print(n)  # 5420,891
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10)  # 2-9
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


digit_indices = [np.where(y_train == i)[0] for i in range(10)]  # y_train中值为i的下标值
# print(digit_indices)
tr_pairs, tr_y = create_pairs(x_train, digit_indices)
# print(tr_pairs)  # (108400,2,28,28)
# print(tr_y, len(tr_y))  # 108400,1 0 1 0交叉

digit_indices = [np.where(y_test == i)[0] for i in range(10)]
te_pairs, te_y = create_pairs(x_test, digit_indices)
# print(te_pairs.shape)  # (17820, 2, 28, 28)
# print(te_y.shape)  # (17820,)


def create_base_network(input_shape):
    input = keras.Input(shape=input_shape)
    x = keras.layers.Flatten()(input)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    return keras.models.Model(input, x)


base_network = create_base_network(input_shape)
base_network.summary()

input_a = keras.Input(shape=input_shape)
input_b = keras.Input(shape=input_shape)
processed_a = base_network(input_a)
processed_b = base_network(input_b)


def euclidean_distance(vects):  # 欧式距离
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    print(shape1[0])
    return (shape1[0], 1)


distance = keras.layers.Lambda(euclidean_distance,  # 要实现的函数
                               output_shape=eucl_dist_output_shape)([processed_a, processed_b])
model = keras.models.Model([input_a, input_b], distance)
model.summary()


def contrastive_loss(y_true, y_pred):  # 对比损失
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


model.compile(loss=contrastive_loss, optimizer=keras.optimizers.RMSprop(), metrics=[accuracy])
# 拟合distance 和 1 0 1 0...
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=128,
          epochs=100,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))


def compute_accuracy(y_true, y_pred):
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(tr_y, y_pred)
y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(te_y, y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

# 100Epoch:loss: 0.0067 - accuracy: 0.9927 - val_loss: 0.0287 - val_accuracy: 0.9698

"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 28, 28)            0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 784)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               100480    
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 128)               16512     
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 128)               16512     
=================================================================
Total params: 133,504
Trainable params: 133,504
Non-trainable params: 0
_________________________________________________________________
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_2 (InputLayer)            (None, 28, 28)       0                                            
__________________________________________________________________________________________________
input_3 (InputLayer)            (None, 28, 28)       0                                            
__________________________________________________________________________________________________
model_1 (Model)                 (None, 128)          133504      input_2[0][0]                    
                                                                 input_3[0][0]                    
__________________________________________________________________________________________________
lambda_1 (Lambda)               (None, 1)            0           model_1[1][0]                    
                                                                 model_1[2][0]                    
==================================================================================================
Total params: 133,504
Trainable params: 133,504
Non-trainable params: 0
__________________________________________________________________________________________________

"""
