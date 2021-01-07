#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 11:44:19 2020

@author: daliana
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 18:14:10 2020

@author: daliana
"""
import json
import logging
import os
import numpy as np
import sys
import errno
from osgeo import gdal
import glob
import multiprocessing
import subprocess, signal
from sklearn import preprocessing as pp
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from keras.callbacks import Callback
import csv
from sklearn import metrics
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE #T-Distributed Stochastic Neighbor Embedding
import time
import warnings
import tensorflow as tf
from keras import backend as K
from keras.layers import Input
from keras.engine.topology import Layer, InputSpec


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def load_image(patch):
    # Read Image
    print (patch)
    gdal_header = gdal.Open(patch)
    # get array
    img = gdal_header.ReadAsArray()
    
    return img

def kill_child_processes(signum, frame):
    parent_id = os.getpid()
    ps_command = subprocess.Popen("ps -o pid --ppid %d --noheaders" % parent_id, shell=True, stdout=subprocess.PIPE)
    ps_output = ps_command.stdout.read()
    retcode = ps_command.wait()
    for pid_str in ps_output.strip().split("\n")[:-1]:
        os.kill(int(pid_str), signal.SIGTERM)
    sys.exit()
    
def load_image_SAR(list_dir):
    # Read Image
    files_tif = glob.glob(os.path.join(list_dir) + '/*.tif')
    for path in files_tif:
        print (path)
        gdal_header = gdal.Open(path)
        # get array
        img = gdal_header.ReadAsArray()
        img[np.isnan(img)] = 0
        img = 10.0**(img/10.0)     # from db to intensity
        img[img>1] = 1   # for SAR 
        img = img.reshape((img.shape[0],img.shape[1],1))
        try:
            stack_img = np.concatenate((stack_img,img), axis=-1)
        except:
            stack_img = img.copy()
            
    return stack_img


def create_stack_SAR(raster_path,start,end):
    list_dir = os.listdir(raster_path)
    list_dir.sort(key=lambda f: int((f.split('_')[0])))
    list_dir = [os.path.join(raster_path,f) for f in list_dir[start:end]]
    num_cores = multiprocessing.cpu_count()    
    
    pool = multiprocessing.Pool(num_cores)
    img_stack = pool.map(load_image_SAR, list_dir)    
    signal.signal(signal.SIGTERM, kill_child_processes)
    pool.close()
    pool.join()
    
    img_stack = np.array(img_stack)
    img_stack = np.rollaxis(img_stack,0,3) 
    img_stack = img_stack.reshape((img_stack.shape[0],img_stack.shape[1],
                                   img_stack.shape[2]*img_stack.shape[3]))
    return img_stack

def load_norm(img,start,end,coords = None, scaler_filename = None):
    # load images, create stack and normalize
    image = create_stack_SAR(img,start,end)
    row,col,depth = image.shape
    if not os.path.isfile(scaler_filename):
        img_tmp = image[coords]
        scaler = pp.StandardScaler().fit(img_tmp)
        img_tmp = []                
        joblib.dump(scaler, scaler_filename) 
    else:
        print('Loading scaler')
        scaler = joblib.load(scaler_filename) 

    image = image.reshape((row*col,depth))
    image = scaler.transform(image)
    image = image.reshape((row,col,depth))    
    return image

def check_folder(folder_dir):
    '''Create folder if not available
    '''
    if not os.path.exists(folder_dir):
        try:
            os.makedirs(folder_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise              
                
def listFiles(top_dir='.', exten='.tif'):
    list_dir = os.listdir(top_dir)
    list_dir.sort(key=lambda f: int(filter(str.isdigit, f.split('_')[0])))

    filesPathList = list()
    for dirpath in list_dir:
        files_tif = glob.glob(os.path.join(top_dir,dirpath) + '/*.tif')
        filesPathList.extend(files_tif)

    return filesPathList

    
    
def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
        
def split_train_val(coords_sq1, coords_sq2 ):
    """Split the data in train and validation."""                
    
    # Shuflle
    #index_seq1 = np.random.choice(len(coords_sq1[0]), len(coords_sq1[0]), replace=False)
    #index_seq2 = np.random.choice(len(coords_sq2[0]), len(coords_sq2[0]), replace=False)
    
    index_seq1 = np.array(range(len(coords_sq1[0])))
    index_seq2 = np.array(range(len(coords_sq2[0])))
    
    # Split into train and validation set
    split_seq1 = int(0.9 * len(index_seq1))
    split_seq2 = int(0.9 * len(index_seq2))
    
    # get indices for training and validation set and stack both sequences 
    # to form final train and validation sets
    index_tr_seq1 = index_seq1[:split_seq1]
    index_tr_seq1 = index_tr_seq1.reshape((len(index_tr_seq1),1))
    # create a second column to specify each seq, 0 for seq 1 and 1 for seq 2
    index_tr_seq1 = np.hstack((index_tr_seq1,np.zeros((len(index_tr_seq1),1))))
    index_val_seq1 = index_seq1[split_seq1:] 
    index_val_seq1 = index_val_seq1.reshape((len(index_val_seq1),1))
    index_val_seq1 = np.hstack((index_val_seq1,np.zeros((len(index_val_seq1),1))))
    
    index_tr_seq2 = index_seq2[:split_seq2]
    index_tr_seq2 = index_tr_seq2.reshape((len(index_tr_seq2),1))
    # create a second colums to specify each seq, 0 for seq 1 and 1 for seq 2
    index_tr_seq2 = np.hstack((index_tr_seq2,np.ones((len(index_tr_seq2),1))))
    index_val_seq2 = index_seq2[split_seq2:] 
    index_val_seq2 = index_val_seq2.reshape((len(index_val_seq2),1))
    index_val_seq2 = np.hstack((index_val_seq2,np.ones((len(index_val_seq2),1))))
    
    # concatenate index, images and coords
    index_tr = np.int64(np.vstack((index_tr_seq1,index_tr_seq2)))
    np.random.shuffle(index_tr)
    index_val = np.int64(np.vstack((index_val_seq1,index_val_seq2)))
    np.random.shuffle(index_val)
    
    coords = np.hstack((coords_sq1,coords_sq2))  
            
    return  coords, index_tr, index_val

def split_test(coords_sq1, coords_sq2):
        
        index_seq1 = np.array(range(len(coords_sq1[0])))
        index_seq2 = np.array(range(len(coords_sq2[0])))
    
        index_seq1 = index_seq1.reshape((len(index_seq1),1))
        index_seq1 = np.hstack((index_seq1,np.zeros((len(index_seq1),1))))
        index_seq2 = index_seq2.reshape((len(index_seq2),1))
        index_seq2 = np.hstack((index_seq2,np.ones((len(index_seq2),1))))
               
        # concatenate index and coords
        index_test = np.int64(np.vstack((index_seq1,index_seq2)))
        coords = np.hstack((coords_sq1,coords_sq2))
       
        return index_test, coords

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    
    assert y_pred.size == y_true.size
    
    D = max(y_pred.max(), y_true.max()) + 1
    
    w = np.zeros((D, D), dtype=np.int64)
    
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
        
    #from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment
    
    #ind = linear_assignment(w.max() - w)
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

         
def ratio_per_class(model_dir, train, labels_all):
    
    if train:
        ratio = np.array([len(labels_all[labels_all==1]),len(labels_all[labels_all==2]),
                 len(labels_all[labels_all==3]),len(labels_all[labels_all==4])])        
        ratio = np.max(ratio)/ratio
        np.save(os.path.join(model_dir,'ratio'),ratio)
    else:
        ratio = np.load(os.path.join(model_dir,'ratio.npy'))
    
    return ratio
        
def mapping_labels(labels_all, classes):
    # Mapping labels   
    lbl_tmp = labels_all.copy()
    labels2new_labels = dict((c, i) for i, c in enumerate(classes))
    new_labels2labels = dict((i, c) for i, c in enumerate(classes))
    for j in range(len(classes)):
        labels_all[lbl_tmp == classes[j]] = labels2new_labels[classes[j]]
    
    return labels_all

def check_overwriting_models(model_k, file_output, restore_from):
    
    # Set the logger
    set_logger(os.path.join(model_k, 'train.log'))
    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwritting
    model_dir_has_best_weights = os.path.isfile(file_output)
    overwritting = model_dir_has_best_weights and restore_from is None
    assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"    

def cf(save_dir, y_true, y_pred, pred_name, acc):
    
    sns.set(font_scale=3)
    #cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    #cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    acc = round(acc*100,2)
    
    uniqueLabels = list(set(y_true))
    clusters = list(set(y_pred))
    cm = [[0 for i in range(len(clusters))] for i in range(len(uniqueLabels))]
    for i, act_label in enumerate(uniqueLabels):
        for j, pred_label in enumerate(y_pred):
            if y_true[j] == act_label:
                cm[i][pred_label] = cm[i][pred_label] + 1

    cm = np.asarray(cm) 
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(16, 14))
    sns.heatmap(cmn, annot=True, cbar=0, cmap="YlGnBu", linewidths=.5, linecolor= 'black')
    plt.title('Average class accuracy: ' + str(acc), fontsize=30)
    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Clustering label', fontsize=25)
    plt.savefig(os.path.join(save_dir,'conf_mat_{}.png'.format(pred_name)), dpi = 300, format='png', bbox_inches = 'tight')
    plt.clf()
    plt.close()   
    

def write_metrics_k_means(save_dir, y_true, y_pred, index_test, flag, pred_name):
    
        if flag:           
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            logfile = open(save_dir + '/{}.csv'.format(pred_name), 'w')
            logwriter = csv.DictWriter(logfile, fieldnames=[ 'acc', 'nmi', 'ari'])
            logwriter.writeheader()
        else:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            logfile = open(save_dir + '/{}.csv'.format(pred_name), 'w')
            logwriter = csv.DictWriter(logfile, fieldnames=[ 'acc', 'nmi', 'ari'])
            logwriter.writeheader()
        
        #Accuracy metrics
        acc = np.round(cluster_acc(y_true, y_pred), 5)
        nmi = np.round(metrics.normalized_mutual_info_score(y_true, y_pred), 5)
        ari = np.round(metrics.adjusted_rand_score(y_true, y_pred), 5)
        print('acc = %.5f, nmi = %.5f, ari = %.5f' % ( acc, nmi, ari))
        logwriter.writerow(dict(acc=acc, nmi=nmi, ari=ari))
            
#        index_0 = index_test[:,1]==0
#        index_1 = index_test[:,1]==1
#        
#        y_true_seq1 = y_true[index_0]
#        y_true_seq2 = y_true[index_1]
#        
#        y_pred_seq1 = y_pred[index_0]
#        y_pred_seq2 = y_pred[index_1]
#        
#        #Accuracy metrics
#        acc_seq1 = np.round(cluster_acc(y_true_seq1, y_pred_seq1), 5)
#        nmi_seq1 = np.round(metrics.normalized_mutual_info_score(y_true_seq1, y_pred_seq1), 5)
#        ari_seq1 = np.round(metrics.adjusted_rand_score(y_true_seq1, y_pred_seq1), 5)
#        #print('acc_seq1 = %.5f, nmi_seq1 = %.5f, ari_seq1 = %.5f' % ( acc_seq1, nmi_seq1, ari_seq1))
#        logwriter.writerow(dict(acc=acc_seq1, nmi=nmi_seq1, ari=ari_seq1))
#        
#        #Accuracy metrics
#        acc_seq2 = np.round(cluster_acc(y_true_seq2, y_pred_seq2), 5)
#        nmi_seq2 = np.round(metrics.normalized_mutual_info_score(y_true_seq2, y_pred_seq2), 5)
#        ari_seq2 = np.round(metrics.adjusted_rand_score(y_true_seq2, y_pred_seq2), 5)
#        #print('acc_seq2 = %.5f, nmi_seq2 = %.5f, ari_seq2 = %.5f' % ( acc_seq2, nmi_seq2, ari_seq2))
#        logwriter.writerow(dict(acc=acc_seq2, nmi=nmi_seq2, ari=ari_seq2))
        logfile.close()
        
#        cf(save_dir, y_true, y_pred, pred_name, acc)

    
#@staticmethod
def target_distribution(q):  # target distribution P which enhances the discrimination of soft label Q
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T    


class EarlyStoppingByLossVal(Callback):
    """Stop training when a monitored quantity has stopped improving or min ~zeros.
    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
            value = min value desired
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
    """

    def __init__(self, monitor='val_loss',
                 min_delta=0, value = 0, patience=0, verbose=0, mode='auto'):
        super(Callback, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.value = value
        self.wait = 0
        self.stopped_epoch = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
            return
        
        if self.best <= self.value:
            self.stopped_epoch = epoch
            self.model.stop_training = True

        elif self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))


def t_SNE_visualization(model_dir, extract_features, y_pred):

    import seaborn as sns
    time_start = time.time()
    
    extract_features = extract_features[0:20000]
    y_pred = y_pred[0:20000]
        
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(extract_features)
    
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    
    plt.figure(figsize = (10,10))
#    sns.scatterplot(tsne_results[:,0], tsne_results[:,1], 
#                    hue=y_pred, 
#                    palette='Set1', s=100, alpha=0.2).set_title('t_SNE Visualization', fontsize=15)
    plt.legend()
    plt.ylabel('t_sne2')
    plt.xlabel('t_sne1')
  
    plt.savefig(os.path.join(model_dir,'t_SNE.png'), dpi = 300, format='png', bbox_inches = 'tight')
    plt.clf()
    plt.close()
    sys.exit()

#    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
#    tsne_results = tsne.fit_transform(y_true)

    sys.exit()


def f_func(x, y):
   return tf.reduce_sum(tf.square(x - y), axis=1)

def g_func(x, y):
   return tf.reduce_sum(tf.square(x - y), axis=1)



class ClusteringLayer(Layer):
    """

    """

    def __init__(self, n_clusters, weights=None, alpha = 1.0, **kwargs):
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

        list_dist = []
        for i in range(0, self.n_clusters):
            dist = f_func(inputs, self.clusters[i:i+1, :])
            list_dist.append(dist)
        self.stack_dist = tf.stack(list_dist)

        return tf.transpose(self.stack_dist)
    

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
     

class KmeansLoss(Layer):
    """

    """
    def __init__(self, n_clusters, weights=None, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(KmeansLoss, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)


    def build(self, input_shape):
        assert len(input_shape) == 2
        #input_dim = input_shape[1]
        #self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.alpha = self.add_weight(shape=(1,), initializer='glorot_uniform', name='alpha')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True


    def call(self, dists, **kwargs):
        
        dists = tf.transpose(dists)
        ## Second, find the minimum squared distance for softmax normalization
        min_dist = tf.reduce_min(dists, axis=0)
        
        ## Third, compute exponentials shifted with min_dist to avoid underflow (0/0) issues in softmaxes
        list_exp = []
        for i in range(self.n_clusters):
            exp = tf.exp(-self.alpha * (dists[i] - min_dist))
            list_exp.append(exp)
        stack_exp = tf.stack(list_exp)
        sum_exponentials = tf.reduce_sum(stack_exp, axis=0)
                
        ## Fourth, compute softmaxes and the embedding/representative distances weighted by softmax
        list_softmax = []
        list_weighted_dist = []
        for j in range(self.n_clusters):
            softmax = stack_exp[j] / sum_exponentials
            weighted_dist = dists[j] * softmax
            list_softmax.append(softmax)
            list_weighted_dist.append(weighted_dist)
        stack_weighted_dist = tf.stack(list_weighted_dist)

        # Compute the error and k-means term
        kmeans_loss = tf.reduce_sum(stack_weighted_dist, axis=0)

        return kmeans_loss
    
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0],

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(KmeansLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#%%
def Loss_AE():
    """
    """
    def loss(y_true, y_pred):
        rec_error = g_func(y_true, y_pred)
        ae_loss = tf.reduce_mean(rec_error)
        return ae_loss
    return loss


def Loss_KM(lambda_):
    """
    """
    def loss(y_true, y_pred):
        return tf.reduce_mean(y_pred) * lambda_
    
    return loss












    
