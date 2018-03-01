# Keras examples directory

## Vision models examples

[mnist_mlp.py](mnist_mlp.py)
在MNIST数据集上训练一个简单的多层感知机。

[mnist_cnn.py](mnist_cnn.py)
在MNIST数据集上训练一个简单的卷积网络。

[cifar10_cnn.py](cifar10_cnn.py)
在CIFAR10小图片数据集上训练一个简单的卷积神经网络。

[cifar10_resnet.py](cifar10_resnet.py)
在CIFAR10小图片数据集上训练一个残差网络。

[conv_lstm.py](conv_lstm.py)
Demonstrates the use of a convolutional LSTM network.

[image_ocr.py](image_ocr.py)
Trains a convolutional stack followed by a recurrent stack and a CTC logloss function to perform optical character recognition (OCR).

[mnist_acgan.py](mnist_acgan.py)
Implementation of AC-GAN (Auxiliary Classifier GAN) on the MNIST dataset

[mnist_hierarchical_rnn.py](mnist_hierarchical_rnn.py)
训练一个分层循环网络去给MNIST数据集分类。

[mnist_siamese.py](mnist_siamese.py)
在MNIST数据集上取成对数字训练一个Siamese多层感知器。

[mnist_swwae.py](mnist_swwae.py)
Trains a Stacked What-Where AutoEncoder built on residual blocks on the MNIST dataset.

[mnist_transfer_cnn.py](mnist_transfer_cnn.py)
迁移学习的简单例子。

----

## Text & sequences examples

[addition_rnn.py](addition_rnn.py)
Implementation of sequence to sequence learning for performing addition of two numbers (as strings).

[babi_rnn.py](babi_rnn.py)
Trains a two-branch recurrent network on the bAbI dataset for reading comprehension.

[babi_memnn.py](babi_memnn.py)
Trains a memory network on the bAbI dataset for reading comprehension.

[imdb_bidirectional_lstm.py](imdb_bidirectional_lstm.py)
Trains a Bidirectional LSTM on the IMDB sentiment classification task.

[imdb_cnn.py](imdb_cnn.py)
Demonstrates the use of Convolution1D for text classification.

[imdb_cnn_lstm.py](imdb_cnn_lstm.py)
Trains a convolutional stack followed by a recurrent stack network on the IMDB sentiment classification task.

[imdb_fasttext.py](imdb_fasttext.py)
Trains a FastText model on the IMDB sentiment classification task.

[imdb_lstm.py](imdb_lstm.py)
Trains an LSTM model on the IMDB sentiment classification task.

[lstm_stateful.py](lstm_stateful.py)
Demonstrates how to use stateful RNNs to model long sequences efficiently.

[pretrained_word_embeddings.py](pretrained_word_embeddings.py)
Loads pre-trained word embeddings (GloVe embeddings) into a frozen Keras Embedding layer, and uses it to train a text classification model on the 20 Newsgroup dataset.

[reuters_mlp.py](reuters_mlp.py)
Trains and evaluate a simple MLP on the Reuters newswire topic classification task.

----

## Generative models examples

[lstm_text_generation.py](lstm_text_generation.py)
Generates text from Nietzsche's writings.

[conv_filter_visualization.py](conv_filter_visualization.py)
Visualization of the filters of VGG16, via gradient ascent in input space.

[deep_dream.py](deep_dream.py)
Deep Dreams in Keras.

[neural_doodle.py](neural_doodle.py)
Neural doodle.

[neural_style_transfer.py](neural_style_transfer.py)
Neural style transfer.

[variational_autoencoder.py](variational_autoencoder.py)
Demonstrates how to build a variational autoencoder.

[variational_autoencoder_deconv.py](variational_autoencoder_deconv.py)
Demonstrates how to build a variational autoencoder with Keras using deconvolution layers.

----

## Examples demonstrating specific Keras functionality

[antirectifier.py](antirectifier.py)
Demonstrates how to write custom layers for Keras.

[mnist_sklearn_wrapper.py](mnist_sklearn_wrapper.py)
Demonstrates how to use the sklearn wrapper.

[mnist_irnn.py](mnist_irnn.py)
Reproduction of the IRNN experiment with pixel-by-pixel sequential MNIST in "A Simple Way to Initialize Recurrent Networks of Rectified Linear Units" by Le et al.

[mnist_net2net.py](mnist_net2net.py)
Reproduction of the Net2Net experiment with MNIST in "Net2Net: Accelerating Learning via Knowledge Transfer".

[reuters_mlp_relu_vs_selu.py](reuters_mlp_relu_vs_selu.py)
Compares self-normalizing MLPs with regular MLPs.

[mnist_tfrecord.py](mnist_tfrecord.py)
MNIST dataset with TFRecords, the standard TensorFlow data format.

