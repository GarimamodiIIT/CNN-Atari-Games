#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 10:46:00 2019

@author: AshishModi
"""

import os
import numpy as np
from glob import glob
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from itertools import combinations as iter_combinations
import pickle
import random

max_episodes=50

output_dataset_directory='test_grayscale'

pca=PCA(n_components=50, random_state=0)

def loadGrayscaledFrames(episode_directory):
    #returns all the r frames in an episode with the size of 210*160 i.e, r*210*160
    frames=np.array([cv2.imread(frame, 0) for frame in sorted(glob(episode_directory+'/*.png'))])
    frames=(frames/255).astype('float32')
    frames=frames[:, 35:195, :]
    #print(np.shape(frames))
    #pick alternate pixels
    frames=frames[:, ::2, ::2]
    #print(np.shape(frames))
    #print(frames[0][np.where(frames[0]>0.0)])
    return frames

def applyPCA(frames, step):
    #first step will be to apply the standard scaling to the frames 
    #std_scaler=StandardScaler()
    #frames=std_scaler.fit_transform(frames)
    print('PCA')
    #apply pca on each frame individually
    #pca=PCA(n_components=50, random_state=0)
    if step==0:
        
        frames=pca.fit_transform(frames)
    else:
        frames=pca.transform(frames)
    #print(frames)
    return frames

def returnCombinations(i, no_of_combinations):
    index=np.linspace(i-6, i-1, 6).astype(int) #keeping the last frame as constant
    #print(index)
    comb_set=list(iter_combinations(index, 4))
    random_combinations=np.random.randint(0, len(comb_set), no_of_combinations).flatten()
    combinations=[]
    for step in random_combinations:
        combinations.append(comb_set[step])
    #print(combinations)
    return combinations

def preprocessTest1(dataset_directory):
    
    #load the list of all the episodes
    
    list_of_episodes=sorted(os.listdir(dataset_directory))
    #print(len(list_of_episodes))
    
    #create the output dataset directory
    if not os.path.exists(output_dataset_directory):
        os.mkdir(output_dataset_directory)
    
    #read images from each episode
    episode_count=0
    
    #frames=loadGrayscaledFrames(train_directory+'/00000001')
    #frames=applyPCA(frames, 0)
    
    #rewards=pd.read_csv(dataset_directory+'/rewards.csv', header=None).values.astype(int)
    #print(np.shape(rewards))
    features=[]
    save_count=0
    for episode in list_of_episodes:
        path_to_episode=dataset_directory+'/'+episode
        if not os.path.isdir(path_to_episode):
            continue
        #first step will be to load all the images in the directory
        frames=loadGrayscaledFrames(path_to_episode)
        #apply PCA to every frame in an episode
        #frames=applyPCA(frames, 1)
        #print(np.shape(frames))
        new_frame=[]
        for i in range(5):
            
            if i==0:
                new_frame=frames[i]
            else:
                new_frame=np.hstack([new_frame, frames[i]])
        new_frame=new_frame.reshape(new_frame.shape[0],  new_frame.shape[1], 1)
        #print(np.shape(new_frame))
        
        '''if len(features)==0:
            features=new_frame
        else:
            features=np.vstack([features, new_frame])'''
        features.append(new_frame)
        #print(np.shape(features))
        #load the rewards
        #print(np.shape(frames))
        
        #print(rewards.shape[0])
        
        
        
        #dump the features
        episode_count+=1
        if episode_count%500 ==0:
        
            print(np.shape(features))
            features=np.asarray(features)
            if not os.path.exists(output_dataset_directory+'/'+str(save_count)):
                os.mkdir(output_dataset_directory+'/'+str(save_count))
            outfile=open(output_dataset_directory+'/'+str(save_count)+'/features', 'wb')
            pickle.dump(features, outfile)
            outfile.close()
            features=[]
            save_count+=1
            
    #outfile=open(output_dataset_directory+'/val_labels', 'wb')
    #pickle.dump(rewards, outfile)
    #outfile.close()
    
    print(np.shape(features))
    if save_count==61:
        print(np.shape(features))
        features=np.asarray(features)
        if not os.path.exists(output_dataset_directory+'/'+str(save_count)):
            os.mkdir(output_dataset_directory+'/'+str(save_count))
        outfile=open(output_dataset_directory+'/'+str(save_count)+'/features', 'wb')
        pickle.dump(features, outfile)
        outfile.close()
    #if episode_count>=max_episodes:
    #break
        
        
        #break'''
            

def loadFeaturesTest1():
    
    list_of_features=([feature for feature in sorted(glob(output_dataset_directory+'/*_features'))])
    print(list_of_features)
    features=[]
    for feature_file in list_of_features:
        infile=open(feature_file, 'rb')
        new_features=pickle.load(infile)
        infile.close()
        if len(features)==0:
            features=new_features
        else:
            features=np.vstack([features, new_features])
    print(np.shape(features))
    
    '''list_of_labels=([label for label in sorted(glob(output_dataset_directory+'/*_labels'))])
    print(list_of_labels)
    labels=[]
    for labels_file in list_of_labels:
        infile=open(labels_file, 'rb')
        new_labels=pickle.load(infile)
        infile.close()
        
        if len(labels)==0:
            labels=new_labels
        else:
            labels=np.append(labels, new_labels)
        print(np.shape(labels))
    print(np.shape(labels))
    
    labels=labels[:, 1]
    print(labels)'''
    
    
    
    return features

def randomizeFeatures(features, labels):
    index=np.random.permutation(len(features))
    #print(index[np.where(index==len(features)-1)])
    features=features[index]
    labels=labels[index]
    #print(np.shape(labels))
    return features, labels
            
if __name__=='__main__':
    preprocessTest1('../SVM/test_dataset')
        
    '''features=loadSVMFeaturesTest1()'''
