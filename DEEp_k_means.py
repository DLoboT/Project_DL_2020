#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:59:58 2020

@author: daliana
"""

from time import time
import argparse
import logging
import os
from utils import assigment_cluster, write_metrics_k_means
from function_sar_deepkmeans_ import Params, load_image, load_norm, check_folder, cluster_acc, EarlyStoppingByLossVal
from function_sar_deepkmeans_ import  split_train_val, target_distribution, split_test, ratio_per_class, t_SNE_visualization, KmeansLoss, Loss_KM, Loss_AE
from function_sar_deepkmeans_ import mapping_labels, check_overwriting_models, f_func, g_func, ClusteringLayer
from models import autoencoderConv2D_1, Metrics, cnn
from keras.callbacks import ModelCheckpoint
import glob
import numpy as np
from generator_DKM import DataGenerator
from keras.utils import plot_model
from keras.optimizers import  Adam, SGD
from keras.models import Model
from keras.layers import Input
from sklearn.cluster import KMeans
import gc
from sklearn import metrics
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='./experiments/v1/',
                    help="Experiment directory containing params.json")
parser.add_argument('--restore_from', default=None,
                    help="Optional, file containing weights to reload before training")

parser.add_argument('--img_sar', default='/mnt/Datos/Materias/2020.1/DL/Campo_Verde/Rasters/', help="Directory with theimages dataset")
parser.add_argument('--gt_dir', default='/mnt/Datos/Materias/2020.1/DL/Campo_Verde/Labels_uint8/', help="Directory with the gt dataset")
parser.add_argument('--mask_dir', default='/mnt/Datos/Materias/2020.1/DL/Campo_Verde/TrainTestMask_50_50.tif', help="Directory with the mask")

parser.add_argument('--img_seq1', default=[1,7], help="SAR images for sequence 1")
parser.add_argument('--img_seq2', default=[8,14], help="Optical image for sequence 2")
parser.add_argument('--date_label1', default=4, help="Label image for sequence 1")
parser.add_argument('--date_label2', default=11, help="Label image for sequence 2")
parser.add_argument('--cluster_number', default=4, help="4-predefined cluster, 5- representative classes only, 11-all")
parser.add_argument('--samples_per_epoch', default=30000, help="Samples per class")
parser.add_argument('--num_epochs', default=1)

parser.add_argument('--num_epochs_per_alpha', default=1, 
                    help="Number of epochs to train the DKM with each alpha")

parser.add_argument('--save_dir', default='./results/dec')
parser.add_argument('--pre_train', default=False, help="True for pre-train the autoencoder and the cluster representatives")
parser.add_argument('--DKM', default=True, help="Fine Tuning with the DKM method")

# DKM parameters
parser.add_argument("--annealing", default=True, 
                    help="Use an annealing scheme for the values of alpha (otherwise a constant is used) for DKM ")
parser.add_argument("--lambda", type=float, default=0.01, dest="lambda_",
                    help="Value of the hyperparameter weighing the clustering loss against the reconstruction loss")

if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    assert os.path.isdir(args.img_sar), "Couldn't find the dataset at {}".format(args.img_sar)
    assert os.path.isdir(args.gt_dir), "Couldn't find the dataset at {}".format(args.gt_dir)

    #   List of the folders
    labels_list = glob.glob(args.gt_dir + '/*.tif')
    labels_list.sort()    
    #    Loading the labels
    label_seq1 = load_image(labels_list[args.date_label1-1])
    label_seq2 = load_image(labels_list[args.date_label2-1])
    
    mask_img = load_image(args.mask_dir)
    
    # Loading the mask of train-test
    # get coordinates of training=1 or test=2 samples 
#    idx = 1 if args.train else 2
    idx = 2
    coords = np.where(mask_img==idx)
                
    # Load data and normalize
    scaler_seq1 = os.path.join(args.model_dir,"scaler_seq1.save")
    data_seq1 = load_norm(args.img_sar,args.img_seq1[0]-1, args.img_seq1[1], coords, scaler_seq1)
    params.channels = data_seq1.shape[-1]
    
    scaler_seq2 = os.path.join(args.model_dir,"scaler_seq2.save")
    data_seq2 = load_norm(args.img_sar,args.img_seq2[0]-1, args.img_seq2[1], coords, scaler_seq2)
    
    # Mask labels and get classes of interest (1,2,3)
    # Class 4 as others
    label_seq1[mask_img!=idx] = 0
    label_seq2[mask_img!=idx] = 0
     
    if args.cluster_number == 4:
        
        label_seq1[(label_seq1>3)] = 4
        label_seq2[(label_seq2>3)] = 4
    
    ##  Setting clustering of only relevant classes
    elif args.cluster_number == 5:
        index_1 = (label_seq1==1) + (label_seq1==2) + (label_seq1==3) + (label_seq1==9) 
        index_2 = (label_seq2==2) + (label_seq2==3) + (label_seq2==9) 
        label_seq1[index_1==False]=0
        label_seq2[index_2==False]=0

#    # Get coordinates for each seq
    coords_sq1 = np.where(label_seq1!=0)
    coords_sq2 = np.where(label_seq2!=0)

#%%    
    # Concatenate data and labels
    data_all = np.concatenate((data_seq1,data_seq2), axis=-1)
    label_seq1 = label_seq1.reshape((label_seq1.shape[0],label_seq1.shape[1],1))
    label_seq2 = label_seq2.reshape((label_seq2.shape[0],label_seq2.shape[1],1))
    labels_all = np.concatenate((label_seq1,label_seq2), axis=-1)

    print("Hyperparameters...")
    print("lambda =", args.lambda_)
    
    if args.annealing and not args.pre_train:
        constant_value = 1  # specs.embedding_size # Used to modify the range of the alpha scheme
        max_n = 40  # Number of alpha values to consider
        alphas = np.zeros(max_n, dtype=float)
        alphas[0] = 0.1
        for i in range(1, max_n):
            alphas[i] = (2 ** (1 / (np.log(i + 1)) ** 2)) * alphas[i - 1]
        alphas = alphas / constant_value
    elif not args.annealing and args.pre_train:
        constant_value = 1  # specs.embedding_size # Used to modify the range of the alpha scheme
        max_n = 20  # Number of alpha values to consider (constant values are used here)
        alphas = 8*np.ones(max_n, dtype=float) # alpha is constant
        alphas = alphas / constant_value
    else:
        parser.error("Run with either annealing or pretraining, but not both for DKM.")
        exit()

    # Clean vars
    data_seq1 = []
    data_seq2 = []
    label_seq1 = []
    label_seq2 = []
    gc.collect()

    # get ratio for each class
    ratio = ratio_per_class(args.model_dir, args.pre_train, labels_all)
       
    # Order labels
    classes = np.unique(labels_all[labels_all!=0])
    params.classes = len(classes)
    print(params.classes)

    # Mapping labels to a 1,2,3 configuration
    labels_all = mapping_labels(labels_all, classes)
    
    for i in range(10):
        dims_ae =[32, 64, 128, 4]
        
        tf.set_random_seed(123)
        np.random.seed(123)
        
        # Define generators            
        dim = (params.patch_size, params.patch_size, params.channels)
        
        #autoencoder, encoder = cnn(input_shape = dim, classes = params.classes)
        autoencoder, encoder = autoencoderConv2D_1(input_shape=dim, filters = dims_ae)
    
        #print(type(autoencoder.get_layer(name='encoder_%d' % (len(dims_ae))).get_weights()[1]))    
        #print(autoencoder.get_layer(name='encoder_%d' % (len(dims_ae))).get_weights()[0])    
        print(autoencoder.get_layer(name='encoder_%d' % (len(dims_ae))).get_weights()[0].reshape((-1))[0])
    
        optimizer= Adam(lr=1e-4)  
        
        autoencoder.compile(optimizer=optimizer, loss='mse')
        
        n_clusters = params.classes
 
        model_k = os.path.join(args.model_dir, str(i))
        check_folder(model_k)
        file_output = os.path.join(model_k,'bestmodel_{}.hdf5'.format(i))
        
        #This is for pre-train the autoencoder
#        index, coords = split_test(coords_sq1, coords_sq2)
        
#        training_generator = DataGenerator(data_all, None, coords, index, params.channels, 
#                                               params.patch_size, params.batch_size, dim, params.classes, 
#                                               samp_per_epoch= args.samples_per_epoch, shuffle=True) 
       
#        if args.pre_train:
#
#            # Function to check model overwriting            
#            check_overwriting_models(model_k, file_output, args.restore_from)
#
#            plot_model(autoencoder, to_file=os.path.join(args.model_dir,'autoencoder.png'), show_shapes=True)
#                  
##%%  Step 1: pretrain the autoencoder             
#
#            # Train model on dataset
#            history = autoencoder.fit_generator(generator=training_generator,
#                                epochs = args.num_epochs)
#            
#            autoencoder.save(file_output)
#            print('best_weights is loaded successfully.')

#%% Build Clustering model

        if args.pre_train:
        
            autoencoder.load_weights(file_output)
            print('Load Model from: {}'.format(file_output))
                
            #%%  Step 2: initialize cluster centers using k-means    
         
            print('Initializing cluster centers with k-means.')      
            # n_init : Number of time the k-means algorithm will be run with different centroid seeds.
            kmeans = KMeans(n_clusters= n_clusters, n_init=20)
            
            #Returns labels : array, shape [n_samples,] Index of the cluster each sample belongs to.
            index_test, coords = split_test(coords_sq1, coords_sq2)
            
            AE_generator = DataGenerator(data_all, None, coords, index_test, params.channels, params.patch_size, 
                                           params.batch_size, dim, params.classes) 
                       
            print('Extract Features')
            extract_features = encoder.predict_generator(AE_generator, verbose=1)
                        
            y_pred = kmeans.fit_predict(extract_features)
                       
            y_true = labels_all[coords[0][index_test[:,0]], coords[1][index_test[:,0]], index_test[:,1]]
    
            y_pred = assigment_cluster(y_true, y_pred)         
            write_metrics_k_means(args.save_dir, y_true, y_pred, index_test, 'AE+Kmeans_{}'.format(i) )
        
    #%% Step 3: deep k means (DKM)
        if args.DKM:
            #  Choosing only the encoder layers of the autoencoder for my model. 
            hidden = autoencoder.get_layer(name='encoder_%d' % (len(dims_ae))).output
                        
            # The cluster centers are used to initialize the cluster representatives in DKM
            clustering_layer = ClusteringLayer(n_clusters= n_clusters, name='clustering')(hidden)
            
            kmeans_loss = KmeansLoss(n_clusters, trainable=False, name='kmeans')(clustering_layer)        
    
            Deep_km = Model(inputs=encoder.input, outputs=[autoencoder.output, kmeans_loss]) 
            Deep_km.summary()
     
            Deep_km.compile(optimizer=Adam(lr=1e-5), loss=[Loss_AE(), Loss_KM(args.lambda_)])
            
            if args.pre_train:
                Deep_km.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
                                
            #Returns labels : array, shape [n_samples,] Index of the cluster each sample belongs to.
            index, coords = split_test(coords_sq1, coords_sq2)            
            
            deep_kmeans_gen = DataGenerator(data_all, None, coords, index, params.channels, 
                                            params.patch_size, params.batch_size, dim, params.classes, 
                                            samp_per_epoch= args.samples_per_epoch, shuffle=True, 
                                            use_augm = params.use_augm, flag_AE = False)
            
            y_true = labels_all[coords[0][index[:,0]], coords[1][index[:,0]], index[:,1]]
         
            #Training
            for k in range(len(alphas)):
                
                print("Training step: alpha[{}]: {}".format(k, alphas[k]))                
                
                Deep_km.get_layer(name='kmeans').set_weights([alphas[k:k+1]])
                
                print(Deep_km.get_layer(name='kmeans').get_weights()[0].reshape((-1))[0])
                              
                history = Deep_km.fit_generator(generator=deep_kmeans_gen, epochs = args.num_epochs_per_alpha)
    
            
            Deep_km_predictor = Model(inputs=Deep_km.input, outputs=clustering_layer) 
            
            AE_generator = DataGenerator(data_all, None, coords, index, params.channels, params.patch_size, 
                                               params.batch_size, dim, params.classes) 
            
            extract_feat = Deep_km_predictor.predict_generator(AE_generator, verbose=1)
            y_pred = np.argmin(extract_feat, axis = 1)
            
            y_pred = assigment_cluster(y_true, y_pred)
            write_metrics_k_means(args.save_dir, y_true, y_pred, index, 'DKM_{}'.format(i))
            
#            write_metrics_k_means(args.save_dir, y_true, y_pred, index, 0 , 'DKM')
            
            Deep_km.save(file_output)
            print('best_weights DKM is loaded successfully.') 
    
    
