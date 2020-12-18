#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:45:05 2020

@author: laura
"""

import numpy as np
from time import time
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Conv2DTranspose
import keras
from keras.layers import AveragePooling2D, BatchNormalization, Dropout
from keras.layers import ELU
from keras.models import Model
from keras.callbacks import Callback

class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
         Measure the similarity between embedded point z_i and centroid µ_j.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def cnn(input_shape=(15, 15, 14), classes = 4):
    
    elu_alpha = 0.1
    """
    Conv2D model.
    Parameters:
    ----------
    img_shape: e.g. (28, 28, 1) for MNIST
    classes: number of classes
    
    Returns
    -------
        CNN Model
    """
    input_img = Input(shape=input_shape)
    print(input_img.shape)
    ######### CB 1 ###########
    x = Conv2D(32, (3, 3), activation='linear', padding='same', strides=(1, 1))(input_img)
    x = BatchNormalization()(x)
    x = ELU(alpha=elu_alpha)(x)
    x = Conv2D(32, (3, 3), activation='linear', padding='same', strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = ELU(alpha=elu_alpha)(x)
    # x = Conv2D(20, (3, 3), activation='linear', padding='same', strides=(1, 1))(x)
    # x = BatchNormalization()(x)
    # x = ELU(alpha=elu_alpha)(x)
    x = AveragePooling2D()(x)
    ######### CB 2 ###########
    x = Conv2D(64, (3, 3), activation='linear', padding='same', strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = ELU(alpha=elu_alpha)(x)
    x = Conv2D(64, (3, 3), activation='linear', padding='same', strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = ELU(alpha=elu_alpha)(x)
    # x = Conv2D(20, (3, 3), activation='linear', padding='same', strides=(1, 1))(x)
    # x = BatchNormalization()(x)
    # x = ELU(alpha=elu_alpha)(x)
    x = AveragePooling2D()(x)
    ######### Calssification Layer ###########
    x = Flatten()(x)
    x = Dropout(0.35)(x)
    x = Dense(128, activation='linear')(x)
    x = BatchNormalization()(x)
    x = ELU(alpha=elu_alpha)(x)
    x = Dropout(0.35)(x)
    
    encoded = Dense(units=10, name='encoder_last')(x)
    
    x = Dense(units=128*int(input_shape[0]/8)*int(input_shape[0]/8), activation='relu')(encoded)

    x = Reshape((int(input_shape[0]/8), int(input_shape[0]/8), 128))(x)
    
    x = Conv2DTranspose(64, 3, strides=2, padding= 'valid', activation='relu', name='deconv3')(x)

    x = Conv2DTranspose(64, 3, strides=2, padding= 'valid', activation='relu', name='deconv2')(x)
    
    x = Conv2DTranspose(32, 3, strides=2, padding= 'valid', activation='relu', name='deconv1')(x)

    decoded = Conv2D(input_shape[2], 1, padding='same', name='conv1x1')(x)
    
    autoencoder = Model(inputs=input_img, outputs=decoded, name='AE') 
    
    encoder = Model(inputs=input_img, outputs=encoded, name='encoder')
#    x = Dense(classes, activation='softmax')(x)

    return autoencoder, encoder


def autoencoderConv2D_1(input_shape=(15, 15, 14), filters=[32, 64, 128, 10]):
    
    input_img = Input(shape=input_shape)

    x = Conv2D(filters[0], 3, strides=2, padding='same', activation='relu', name='encoder_1', input_shape=input_shape)(input_img)

    x = Conv2D(filters[1], 3, strides=2, padding='same', activation='relu', name='encoder_2')(x)

    x = Conv2D(filters[2], 3, strides=2, padding= 'same', activation='relu', name='encoder_3')(x)

    x = Flatten()(x)
    
    encoded = Dense(units=filters[3], name='encoder_%d' % (len(filters)))(x)
    
    x = Dense(units=filters[2]*int(input_shape[0]/8)*int(input_shape[0]/8), activation='relu')(encoded)

    x = Reshape((int(input_shape[0]/8), int(input_shape[0]/8), filters[2]))(x)
    
    x = Conv2DTranspose(filters[1], 3, strides=2, padding= 'valid', activation='relu', name='deconv3')(x)

    x = Conv2DTranspose(filters[1], 3, strides=2, padding= 'valid', activation='relu', name='deconv2')(x)
    
    x = Conv2DTranspose(filters[0], 3, strides=2, padding= 'valid', activation='relu', name='deconv1')(x)

    decoded = Conv2D(input_shape[2], 1, padding='same', name='AE')(x)
    
    autoencoder = Model(inputs=input_img, outputs=decoded, name='AE') 
    
    encoder = Model(inputs=input_img, outputs=encoded, name='encoder')
       
    return autoencoder, encoder

def autoencoderConv2D_2(input_shape=(15, 15, 14), filters=[32, 64, 128, 256, 10]):

    input_img = Input(shape=input_shape)

    x = Conv2D(filters[0], 3, strides=2, padding='same', activation='relu', name='encoder_1', input_shape=input_shape)(input_img)

    x = Conv2D(filters[1], 3, strides=2, padding='same', activation='relu', name='encoder_2')(x)

    x = Conv2D(filters[2], 3, strides=2, padding= 'same', activation='relu', name='encoder_3')(x)
    
    x = Conv2D(filters[3], 3, strides=2, padding= 'same', activation='relu', name='encoder_4')(x)
    
    encoded = Conv2D(filters[4], 3, strides=1, padding= 'same', activation='relu', name='encoder_5')(x)

    x = Conv2DTranspose(filters[3], 3, strides=1, padding= 'same', activation='relu', name='deconv4')(encoded)
       
    x = Conv2DTranspose(filters[2], 3, strides=2, padding= 'valid', activation='relu', name='deconv3')(x)

    x = Conv2DTranspose(filters[1], 3, strides=2, padding= 'valid', activation='relu', name='deconv2')(x)
    
    x = Conv2DTranspose(filters[0], 3, strides=2, padding= 'valid', activation='relu', name='deconv1')(x)

    decoded = Conv2D(input_shape[2], 1, padding='same', name='conv1x1')(x)
    
    autoencoder = Model(inputs=input_img, outputs=decoded, name='AE') 
    
    encoder = Model(inputs=input_img, outputs=encoded, name='encoder')

    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')

def autoencoder(dims, act='relu'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        Model of autoencoder
    """
    n_stacks = len(dims) - 1
    # input
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, name='encoder_%d' % i)(h)

    # hidden layer
    h = Dense(dims[-1], name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        h = Dense(dims[i], activation=act, name='decoder_%d' % i)(h)

    # output
    h = Dense(dims[0], name='decoder_0')(h)

    return Model(inputs=x, outputs=h)

class Metrics(Callback):
    def __init__(self, validation, patience):   
        super(Metrics, self).__init__()
        self.validation = validation[0] 
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
            
        print('validation shape', len(self.validation[0]))
        
    def on_train_begin(self, logs={}):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = 0

        
    def on_epoch_begin(self, epoch, logs={}):        
        self.true_positives = []
        self.predicted_positives = []
        self.possible_positives = []
        
    def on_batch_end(self, batch, logs={}):
        val_targ = self.validation[1]   
        val_predict = keras.utils.to_categorical(np.argmax(self.model.predict(self.validation[0]),axis=-1), num_classes=val_targ.shape[-1])
        self.true_positives.append(np.sum(np.round(np.clip(val_targ * val_predict, 0, 1)),axis=0))
        self.possible_positives.append(np.sum(np.round(np.clip(val_targ, 0, 1)),axis=0))
        self.predicted_positives.append(np.sum(np.round(np.clip(val_predict, 0, 1)),axis=0))
     
    def on_epoch_end(self, epoch, logs={}):
        recall = np.sum(self.true_positives, axis=0)/(np.sum(self.possible_positives, axis=0) + K.epsilon())
        precision =np.sum(self.true_positives, axis=0)/(np.sum(self.predicted_positives, axis=0) + K.epsilon())
        f1 = 2*((precision*recall)/(precision+recall+K.epsilon()))
        mean_f1 = np.mean(f1)
        
        #update the logs dictionary:
        logs["mean_f1"]=mean_f1

        print(f' — val_f1: {f1}\n — val_precision: {precision}\n — val_recall: {recall}')
        print(f' — mean_f1: {mean_f1}')
        
        current = logs.get("mean_f1")
        if np.less(self.best, current):
            self.best = current
            self.wait = 0
            print("Found best weights at epoch {}".format(epoch + 1))
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)
                
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping".format(self.stopped_epoch + 1))