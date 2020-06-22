#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 21:36:11 2019

@author: AshishModi
"""

import os
import numpy as np
from glob import glob
import cv2
#from sklearn.decomposition import PCA
#xfrom sklearn.preprocessing import StandardScaler
import pandas as pd
from itertools import combinations as iter_combinations
import pickle


max_episodes=500
max_count_pick=150
output_dataset_directory='preprocess_grayscale_cnn'



def loadGrayscaledFrames(episode_directory):
    #returns all the r frames in an episode with the size of 210*160 i.e, r*210*160
    frames=np.array([cv2.imread(frame, 0) for frame in sorted(glob(episode_directory+'/*.png'))])
    frames=(frames/255).astype('float32')
    print(np.shape(frames))
    #remove boundary values
    frames=frames[:, 35:195, :]
    #print(np.shape(frames))
    #pick alternate pixels
    frames=frames[:, ::2, ::2]
    #print(np.shape(frames))
    #print(frames[0][np.where(frames[0]>0.0)])
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

def preprocess(dataset_directory):
    
    #load the list of all the episodes
    
    list_of_episodes=sorted(os.listdir(dataset_directory))
    #print(len(list_of_episodes))
    
    #create the output dataset directory
    if not os.path.exists(output_dataset_directory):
        os.mkdir(output_dataset_directory)
    
    #read images from each episode
    episode_count=0
    for episode in list_of_episodes:
        print(episode)
        path_to_episode=dataset_directory+'/'+episode
        if not os.path.isdir(path_to_episode):
            continue
        #first step will be to load all the images in the directory
        frames=loadGrayscaledFrames(path_to_episode)
        #apply PCA to every frame in an episode
        
        #load the rewards
        #print(np.shape(frames))
        rewards=pd.read_csv(path_to_episode+'/rew.csv', header=None).values.astype(int)
        #print(rewards.shape[0])
        
        flag=0
        zero_features=[]
        zero_labels=[]
        features=[]
        labels=[]
        for i in range(6, len(frames)-1):
            
            #combinations=returnCombinations(i, 2)
            current_reward=rewards[i]
            if not flag==0 and current_reward==0:
                flag=flag-1
                continue
            
            
            if current_reward==0:
                combinations=returnCombinations(i, 1)
                for comb in combinations:
                    new_features=[]
                    for j in comb:
                        if len(new_features)==0:
                            new_features=frames[j]
                        else:
                            new_features=np.hstack([new_features, frames[j]])
                    new_features=np.hstack([new_features, frames[i]])
                    new_features=new_features.reshape(new_features.shape[0],  new_features.shape[1], 1)
                    #print('Stacked_image:', np.shape(new_features))
                    #features.append(new_features)
                    #labels.append(current_reward)
                    '''if len(zero_features)==0:
                        zero_features=new_features
                        zero_labels=current_reward
                    else:
                        zero_features=np.vstack([zero_features, new_features]) 
                        zero_labels=np.vstack([zero_labels, current_reward])'''
                    zero_features.append(new_features)
                    zero_labels.append(current_reward)
                    #features.append(new_features)
                    #labels.append(current_reward)
                #print('0', np.shape(zero_features))
                flag=1
                    
            elif current_reward==1:
                combinations=returnCombinations(i, 1)
                for comb in combinations:
                    new_features=[]
                    for j in comb:
                        if len(new_features)==0:
                            new_features=frames[j]
                        else:
                            new_features=np.hstack([new_features, frames[j]])
                    new_features=np.hstack([new_features, frames[i]])
                    new_features=new_features.reshape(new_features.shape[0],  new_features.shape[1], 1)
                    #print(np.shape(new_features))
                    '''if len(features)==0:
                        features=new_features
                        labels=current_reward
                    else:
                        features=np.vstack([features, new_features]) 
                        labels=np.vstack([labels, current_reward])'''
                    features.append(new_features)
                    labels.append(current_reward)
                    #print('1', np.shape(features))
                    flag=0
                    
            #print(current_reward)
            #print(i)
                
        #print(rewards[len(rewards)-1])    
        count=0
        for i in labels:
            if i==1:
                count+=1
        #print('1 rewards count:', count)
        
        
        
        zero_features=np.asarray(zero_features)
        features=np.asarray(features)
        zero_labels=np.asarray(zero_labels)
        labels=np.asarray(labels)
        #print(np.shape(labels), np.shape(features))
        
        pick_count=max_count_pick-count
        #if count>pick_count:
            #pick_count=count
        
        #random_pick=np.random.randint(0, len(zero_features), len(features))
        random_pick=np.random.randint(0, len(zero_features), pick_count)
        picked_samples=zero_features[random_pick, :, :]
        #print(np.shape(picked_samples))
        picked_sample_labels=zero_labels[random_pick]
        features=np.vstack([features, picked_samples])
        labels=np.vstack([labels, picked_sample_labels])
        #print(np.shape(features), np.shape(labels))
            
        idx=np.random.permutation(len(features))
        features=features[idx]
        labels=labels[idx]
        print(np.shape(features), np.shape(labels))
        
        #dump the features
        #first create the directory for the episode to load its data, as and when required
        if not os.path.exists(output_dataset_directory+'/'+episode):
            os.mkdir(output_dataset_directory+'/'+episode)
        outfile=open(output_dataset_directory+'/'+episode+'/features', 'wb')
        pickle.dump(features, outfile)
        outfile.close()
        outfile=open(output_dataset_directory+'/'+episode+'/labels', 'wb')
        pickle.dump(labels, outfile)
        outfile.close()
        episode_count+=1
        
        #print(np.shape(features))
        if episode_count>=max_episodes:
            break
        
        
        #break
            

'''def loadFeatures():
    
    list_of_features=([feature for feature in sorted(glob(output_dataset_directory+'/*_features'))])
    #print(list_of_features)
    features=[]
    for feature_file in list_of_features:
        infile=open(feature_file, 'rb')
        new_features=pickle.load(infile)
        infile.close()
        if len(features)==0:
            features=new_features
        else:
            features=np.vstack([features, new_features])
        #features.append(new_features)
    #features=np.asarray(featuresa)
    print(np.shape(features))
    
    list_of_labels=([label for label in sorted(glob(output_dataset_directory+'/*_labels'))])
    #print(list_of_labels)
    labels=[]
    for labels_file in list_of_labels:
        infile=open(labels_file, 'rb')
        new_labels=pickle.load(infile)
        infile.close()
        
        if len(labels)==0:
            labels=new_labels
        else:
            labels=np.append(labels, new_labels)
        #print(np.shape(labels))
    print(np.shape(labels))
    
    
    
    return features, labels'''



def randomizeFeatures(features, labels):
    index=np.random.permutation(len(features))
    #print(index[np.where(index==len(features)-1)])
    features=features[index]
    labels=labels[index]
    #print(np.shape(labels))
    return features, labels
            
if __name__=='__main__':
    
    preprocess('../../SVM/trainDataset')
        
    #features, labels=loadFeatures()
        
 