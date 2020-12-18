#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:56:07 2020

@author: laura
"""

import numpy as np
import keras
import sys

# adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, data, labels, coords, idx_coord, channels = 14, 
                 patch_size = 15, batch_size=32, dim=(15,15,14), 
                 n_classes=4, samp_per_epoch = None, shuffle=False, 
                 use_augm=False, flag_AE = True):
        '''
        Parameters
        ----------
        data : image stack with shape (row, col, channels*seq)
        labels : labels stack with shape (row, col, seq)
        coords : x,y coordinate for each pixel of interest
        idx_coord : index of coordinates, shape (len(coords),2)
        channels : channels of imput data for each seq. The default is 14.
        patch_size : patch size. The default is 15.
        batch_size : The default is 32.
        dim : input dimension for the CNN model. The default is (15,15,14).
        n_classes : number of classes. The default is 4.
        samp_per_epoch : (optional) # of samples for each epoch. The default is None.
        shuffle : (optional) shuffle after each epoch. The default is False.
        use_augm : (optional) data augmenattion. The default is False.

        Returns
        -------
            Datagenerator

        '''

        self.data = data
        self.label = labels
        self.dim = dim
        self.batch_size = batch_size
        self.list_coords = idx_coord
        self.coords = coords
        self.shuffle = shuffle
        self.patch_size = patch_size
        self.channels = channels
        self.n_classes = n_classes
        self.use_augm = use_augm
        self.samp_per_epoch = samp_per_epoch
        self.flag_AE = flag_AE
        self.on_epoch_end()
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.samp_per_epoch:  
            # train over #samp_per_epoch random samples at each epoch
            return int(np.ceil(self.samp_per_epoch / self.batch_size))
        else:
            # use all avaliable samples at each epoch
            return int(np.ceil(len(self.list_coords) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if (index+1)*self.batch_size > len(self.indexes):
            indexes = self.indexes[index*self.batch_size:]
        else:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        idx_tmp = [self.list_coords[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(idx_tmp)

        return X, Y 

    def on_epoch_end(self):
        'Updates indexes and list coords after each epoch'
        if self.samp_per_epoch:
            self.indexes = np.arange(self.samp_per_epoch)
        else:
            self.indexes = np.arange(len(self.list_coords))
            
        if self.shuffle == True:
            # shuffle indexes we use to iterate on
            np.random.shuffle(self.indexes)
            # shuffle the coordiantes index 
            np.random.shuffle(self.list_coords)

    def __data_generation(self, idx_tmp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        X = np.empty((len(idx_tmp), *self.dim))

        # Generate data
        for i in range(len(idx_tmp)):
            # get patch
            # idx_tmp --> (index for x y coord, index for the seq (0 for seq 1 and 1 for seq 2) )
            patch_tmp = self.data[self.coords[0][idx_tmp[i][0]]-self.patch_size//2:self.coords[0][idx_tmp[i][0]]+self.patch_size//2+self.patch_size%2,
                                  self.coords[1][idx_tmp[i][0]]-self.patch_size//2:self.coords[1][idx_tmp[i][0]]+self.patch_size//2+self.patch_size%2,
                                  idx_tmp[i][1]*self.channels:idx_tmp[i][1]*self.channels+self.channels]

            idx_tmp = np.array(idx_tmp)
            # Random flips and rotations 
            if self.use_augm:
                transf = np.random.randint(0,5,1)
                if transf == 0:
                    # rot 90
                    patch_tmp = np.rot90(patch_tmp,1,(0,1))
                    
                elif transf == 1:
                    # rot 180
                    patch_tmp = np.rot90(patch_tmp,2,(0,1))
                  
                elif transf == 2:
                    # flip horizontal
                    patch_tmp = np.flip(patch_tmp,0)
                  
                elif transf == 3:
                    # flip vertical
                    patch_tmp = np.flip(patch_tmp,0)
                  
                elif transf == 4:
                    # rot 270
                    patch_tmp = np.rot90(patch_tmp,3,(0,1))
                 
                
            X[i,] = patch_tmp
        
        if self.label is not None:
            Y = np.empty((len(idx_tmp),self.label.shape[-1]), dtype=np.float64)
            
            for i in range(len(idx_tmp)):            
                
                Y[i,] = self.label[idx_tmp[i][2]]

            return X, Y
        
        if self.flag_AE:
            return X, X
        
        return X, [X, np.ones((len(idx_tmp)))]


    
    
