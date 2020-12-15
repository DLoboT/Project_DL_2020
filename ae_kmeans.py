#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 17:55:46 2020

@author: daliana
"""


from time import time
import argparse
import logging
import os
from functions_sar import Params, load_image, load_norm, check_folder, cluster_acc, EarlyStoppingByLossVal
from functions_sar import  split_train_val, target_distribution, split_test, ratio_per_class, t_SNE_visualization  
from functions_sar import mapping_labels, check_overwriting_models, write_metrics_k_means
from models import ClusteringLayer, autoencoderConv2D_1, Metrics, cnn
from keras.callbacks import ModelCheckpoint
import glob
import numpy as np
from generator import DataGenerator
from keras.utils import plot_model
from keras.optimizers import  Adam, SGD
from keras.models import Model
from sklearn.cluster import KMeans, SpectralClustering
import gc
from sklearn import metrics
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
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
parser.add_argument('--num_epochs', default=50, help="Samples per class")

parser.add_argument('--train', default=False, help="True for pre-train the autoencoder")
parser.add_argument('--kmeans', default=True, help="True for pre-train the autoencoder")
parser.add_argument('--DEC', default=True, help="DEC method")

# DEC parameters
parser.add_argument('--save_dir', default='./results/dec')
parser.add_argument('--maxiter', default=8e3, type=int)
parser.add_argument('--update_interval', default=200, type=int)
parser.add_argument('--tol', default=0.001, type=float)

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
   
    # Concatenate data and labels
    data_all = np.concatenate((data_seq1,data_seq2), axis=-1)
    label_seq1 = label_seq1.reshape((label_seq1.shape[0],label_seq1.shape[1],1))
    label_seq2 = label_seq2.reshape((label_seq2.shape[0],label_seq2.shape[1],1))
    labels_all = np.concatenate((label_seq1,label_seq2), axis=-1)

    
    # Clean vars
    data_seq1 = []
    data_seq2 = []
    label_seq1 = []
    label_seq2 = []
    gc.collect()

    # get ratio for each class
    ratio = ratio_per_class(args.model_dir, args.train, labels_all)
       
    # Order labels
    classes = np.unique(labels_all[labels_all!=0])
    params.classes = len(classes)
    print(params.classes)

    # Mapping labels to a 1,2,3 configuration
    labels_all = mapping_labels(labels_all, classes)
    
    dims_ae =[32, 64, 128, 4]
    # Define generators            
    dim = (params.patch_size, params.patch_size, params.channels)
    
    tf.reset_default_graph()
    tf.set_random_seed(123)
    np.random.seed(123)
    
    autoencoder, encoder = autoencoderConv2D_1(input_shape=dim, filters = dims_ae)
    optimizer= Adam(lr=1e-4)  
    print(autoencoder.get_layer(name='encoder_%d' % (len(dims_ae))).get_weights()[0].reshape((-1))[0])
    
    autoencoder.compile(optimizer=optimizer, loss='mse')

    autoencoder.save(os.path.join(args.model_dir,'default.hdf5'))
         
    for i in range(20):
        
        model_k = os.path.join(args.model_dir, str(i))
        check_folder(model_k)
        file_output = os.path.join(model_k,'bestmodel_{}.hdf5'.format(i))

        if args.train:
            
            index, coords = split_test(coords_sq1, coords_sq2)
             
            # Function to check model overwriting            
            check_overwriting_models(model_k, file_output, args.restore_from)
                                      
            training_generator = DataGenerator(data_all, None, coords, index, params.channels, 
                                               params.patch_size, params.batch_size, dim, params.classes, 
                                               samp_per_epoch= args.samples_per_epoch, shuffle=True, 
                                               use_augm = params.use_augm) 
     
            plot_model(autoencoder, to_file=os.path.join(args.model_dir,'autoencoder.png'), show_shapes=True)
                  
#%%  Step 1: pretrain the autoencoder             
            checkpointer = ModelCheckpoint(file_output)

            # Train model on dataset
            autoencoder.load_weights(os.path.join(args.model_dir,'default.hdf5'))
            history = autoencoder.fit_generator(generator=training_generator,
                                epochs = args.num_epochs, callbacks = [checkpointer])

            autoencoder.save(file_output)

#%% Step 2: Build Clustering model
        if args.kmeans:
            
            autoencoder.load_weights(file_output)
            print('best_weights is loaded successfully.')
                        
            n_clusters = params.classes
        
            #  Choosing only the encoder layers of the autoencoder for my model. 
            hidden = autoencoder.get_layer(name='encoder_%d' % (len(dims_ae))).output
                       
            # prepare DEC model
            clustering_layer = ClusteringLayer(n_clusters= n_clusters, alpha = 1, name='clustering')(hidden)
            
            model = Model(inputs=encoder.input, outputs=clustering_layer) 
            model.summary()
                
            plot_model(model, to_file=os.path.join(args.model_dir,'cluster.png'), show_shapes=True)    
                
            print('Initializing cluster centers with k-means.')      
            # n_init : Number of time the k-means algorithm will be run with different centroid seeds.
            kmeans = KMeans(n_clusters= n_clusters, n_init=20)
                
            #Returns labels : array, shape [n_samples,] Index of the cluster each sample belongs to.
            index_test, coords = split_test(coords_sq1, coords_sq2)
               
            # Here I create a new index to match the data with the target p.
            index_p = np.array(range(len(index_test)))[:, np.newaxis]
            index_test = np.concatenate((index_test,index_p), axis = 1)
          
            AE_generator = DataGenerator(data_all, None, coords, index_test, params.channels, params.patch_size, 
                                           params.batch_size, dim, params.classes) 
                       
            print('Extract Features')
            extract_features = encoder.predict_generator(AE_generator, verbose=1)
                        
            y_pred = kmeans.fit_predict(extract_features)
    
            y_pred_last = np.copy(y_pred)
            
            model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
        
            y_true = labels_all[coords[0][index_test[:,0]], coords[1][index_test[:,0]], index_test[:,1]]
          
            write_metrics_k_means(args.save_dir, y_true, y_pred, index_test, 1 ,'AE+Kmeans_{}'.format(i) )
    
#%% Step 3: deep clustering (DEC)
        if args.DEC:
                
            model.compile(optimizer=SGD(lr=1e-4), loss='kld')
    
            print('Update interval', args.update_interval)
            nb_samples = len(index_test)
            save_interval = int(len(index_test)/params.batch_size) * 2 # 5 epochs
            print('Save interval', save_interval)
            
            # logging file
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            logfile = open(args.save_dir + '/dec_log_{}.csv'.format(i), 'w')
            logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'L'])
            logwriter.writeheader()
    
            loss = 0
            idx = 0
            t0 = time()
            for ite in range(int(args.maxiter)):
                if ite % args.update_interval == 0:
                    q = model.predict_generator(AE_generator, verbose=1)
                    p = target_distribution(q)  # update the auxiliary target distribution p
                    print(p.shape)
                   # evaluate the clustering performance
                    y_pred = q.argmax(1)
                    if y_true is not None:
                        acc = np.round(cluster_acc(y_true, y_pred), 5)
                        nmi = np.round(metrics.normalized_mutual_info_score(y_true, y_pred), 5)
                        ari = np.round(metrics.adjusted_rand_score(y_true, y_pred), 5)
                        loss = np.round(loss, 5)
                        logwriter.writerow(dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss))
                        print('Iter-%d: ACC= %.4f, NMI= %.4f, ARI= %.4f;  L= %.5f' % (ite, acc, nmi, ari, loss))
    
                    # check stop criterion
                    # When delta_label ==0 is because y_pred = y_pred_last. (It was no improvement)               
                    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                    y_pred_last = np.copy(y_pred)
                    # If I put a high args.tol means that the improvement must to be significative.
                    if ite > 0 and delta_label < args.tol:
                        print('delta_label ', delta_label, '< tol ', args.tol)
                        print('Reached tolerance threshold. Stopping training.')
                        logfile.close()
                        break
    
                # train on batch
                if (idx + 1) * params.batch_size > len(index_test):
                    temp_index_test = index_test[idx * params.batch_size:]
                    idx = 0
                else:
                    temp_index_test = index_test[idx * params.batch_size:(idx + 1) * params.batch_size]
                    idx += 1
                
                test_generator_DEC = DataGenerator(data_all, p, coords, temp_index_test, params.channels, params.patch_size, 
                                           params.batch_size, dim, params.classes) 
    
                loss = model.fit_generator(generator = test_generator_DEC, epochs=1).history['loss'][0]
                
                # save intermediate model
                if ite % save_interval == 0:
                    # save DEC model checkpoints
                    print('saving model to: ' + args.save_dir + '/DEC_model_' + str(ite) + '.h5')
                    model.save_weights(args.save_dir + '/DEC_model_' + str(ite) + '.h5')
    
                ite += 1
    
            # save the trained model
            logfile.close()
            print('saving model to: ' + args.save_dir + '/DEC_model_last_{}.h5'.format(i))
            model.save_weights(args.save_dir + '/DEC_model_last_{}.h5'.format(i))
                   
            acc = cluster_acc(y_true, y_pred)
            # Show the final results
            print('acc_DEC:', acc)
            write_metrics_k_means(args.save_dir, y_true, y_pred, index_test, 0 , 'DEC')
            print('clustering time: %d seconds.' % int(time() - t0))
    
    
