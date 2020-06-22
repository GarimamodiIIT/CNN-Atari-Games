#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 21:34:05 2019

@author: AshishModi
"""


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
import keras.metrics as km

import pickle
import numpy as np
import time
import pandas as pd
import os
import sys

from preprocess_cnn_keras import *
from sklearn.metrics import f1_score
from preprocess_final_test import *
from preprocess_test import *

input_shape=(80, 400, 1)

output_train_dataset_directory='preprocess_grayscale_cnn'
max_epochs=10
batch_size=150
output_validation_dataset_directory='validation_grayscale'
output_test_dataset_directory='test_grayscale'

def createModel():
    model=Sequential()
    
    model.add(Conv2D(32, kernel_size=(3,3), strides=(2, 2), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(64, kernel_size=(3,3), strides=(2, 2), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    return model



def predictRewards(dataset_type, model):
    predictions=[]
    true_rewards=[]
    if dataset_type=='train':
        #read features
        list_of_episodes=sorted(os.listdir(output_train_dataset_directory))
        for episode in list_of_episodes:
            print(episode)
            path_to_episode=output_train_dataset_directory+'/'+episode
            if not os.path.isdir(path_to_episode):
                continue
            
            infile=open(path_to_episode+'/features', 'rb')
            features=pickle.load(infile)
            infile.close()
            #print(np.shape(features))
            infile=open(path_to_episode+'/labels', 'rb')
            labels=pickle.load(infile)
            infile.close()
            labels=labels.flatten()
            #true_rewards.append(labels)
            if len(true_rewards)==0:
                true_rewards=labels
            else:
                true_rewards=np.append(true_rewards, labels)
            #print(true_rewards, np.shape(true_rewards))
            
            reward_predict=model.predict(features, batch_size=1)
            #print(reward_predict)
            reward_predict=np.array((np.squeeze(reward_predict)>0.5).astype('int')).ravel()
            
            #print(reward_predict)
            if len(predictions)==0:
                predictions=reward_predict
            else:
                predictions=np.append(predictions, reward_predict)
                
        #print(predictions, true_rewards)
        print(np.shape(predictions), np.shape(true_rewards))
        print('Training Set')
        score=f1_score(true_rewards, predictions, average=None)
        print('Class f1: ', score)
        score=f1_score(true_rewards, predictions, average='binary')
        print('Binary f1: ', score)
    elif dataset_type=='validation':
        #read features
        true_rewards=[]
        predictions=[]
        #list_of_episodes=sorted(os.listdir(output_validation_dataset_directory))
        list_of_episodes=np.linspace(0, 23, 24).astype(int)
        print(list_of_episodes)
        for episode in list_of_episodes:
            print(episode)
            path_to_episode=output_validation_dataset_directory+'/'+str(episode)
            if not os.path.isdir(path_to_episode):
                continue
            
            infile=open(path_to_episode+'/features', 'rb')
            features=pickle.load(infile)
            infile.close()
            #print(np.shape(features))
            infile=open(path_to_episode+'/labels', 'rb')
            labels=pickle.load(infile)
            infile.close()
            labels=np.array(labels)
            #print(labels)
            #labels=labels[:, 1]
            
            labels=labels.flatten()
            #true_rewards.append(labels)
            if len(true_rewards)==0:
                true_rewards=labels
            else:
                true_rewards=np.append(true_rewards, labels)
            #print(true_rewards, np.shape(true_rewards))
            
            reward_predict=model.predict(features, batch_size=1)
            #print(reward_predict)
            reward_predict=np.array((np.squeeze(reward_predict)>0.5).astype('int')).ravel()
            
            #print(reward_predict)
            if len(predictions)==0:
                predictions=reward_predict
            else:
                predictions=np.append(predictions, reward_predict)
                
        #print(predictions, true_rewards)
        print(np.shape(predictions), np.shape(true_rewards))
        print('Validation Set')
        score=f1_score(true_rewards, predictions, average=None)
        print('Class f1: ', score)
        score=f1_score(true_rewards, predictions, average='binary')
        print('Binary f1: ', score)
        
    elif dataset_type=='test':
        #read features
        true_rewards=[]
        predictions=[]
        #list_of_episodes=sorted(os.listdir(output_test_dataset_directory))
        list_of_episodes=np.linspace(0, 61, 62).astype(int)
        print(list_of_episodes)
        for episode in list_of_episodes:
            print(episode)
            path_to_episode=output_test_dataset_directory+'/'+str(episode)
            if not os.path.isdir(path_to_episode):
                continue
            
            infile=open(path_to_episode+'/features', 'rb')
            features=pickle.load(infile)
            infile.close()
            #print(np.shape(features))
            
            
            #print(true_rewards, np.shape(true_rewards))
            
            reward_predict=model.predict(features, batch_size=1)
            #print(reward_predict)
            reward_predict=np.array((np.squeeze(reward_predict)>0.5).astype('int')).ravel()
            
            #print(reward_predict)
            if len(predictions)==0:
                predictions=reward_predict
            else:
                predictions=np.append(predictions, reward_predict)
                
        #print(predictions, true_rewards)
        print(np.shape(predictions))
        
        predictions=np.asarray(predictions)
        print('Test Predictions')
        print(predictions)
        df=pd.DataFrame(predictions)
        #print(df)
        df.to_csv('cnn_test_predictionstry1.csv')
        
        #predictions.to_csv('cnn_test_predictions.csv')
        
    
    

def cnn():
    
    #max epochs have to be run on each episode read
    list_of_episodes=sorted(os.listdir(output_train_dataset_directory))
    #print(list_of_episodes)
    
    model=createModel()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', km.binary_accuracy])
    for epoch in range(max_epochs):
        
        step=1
        print('-----------------------------------------------------------------------------')
        print('Epoch ', epoch , 'Completed')
        print('-----------------------------------------------------------------------------')
        
        for episode in list_of_episodes:
            path_to_episode=output_train_dataset_directory+'/'+episode
            if not os.path.isdir(path_to_episode):
                continue
            
            infile=open(path_to_episode+'/features', 'rb')
            features=pickle.load(infile)
            infile.close()
            #print(np.shape(features))
            infile=open(path_to_episode+'/labels', 'rb')
            labels=pickle.load(infile)
            infile.close()
            #print(np.shape(labels))
            
            model.fit(features, labels, batch_size=batch_size)
            
            
            
            
            step+=1
            
    #store the model
    outfile=open('cnn_model', 'wb')
    pickle.dump(model, outfile)
    outfile.close()
    
    #load model
    '''infile=open('cnn_model', 'rb')
    model=pickle.load(infile)
    infile.close()'''
    
    print('Prediction of Train Set')
    predictRewards('train', model)
    
    
        
        
        
    
    
    print('Prediction of Validation Set')
    predictRewards('validation', model)
    
    print('Prediction of Test Set')
    predictRewards('test', model)
    
















if __name__=='__main__':
    if len(sys.argv)!=4:
        print('Invalid list of arguments')
        sys.exit()
    
    train_dataset=sys.argv[1]
    
    validation_dataset=sys.argv[2]
    test_dataset=sys.argv[3]
    
    
    #enable these feature extracting thing
    preprocess(train_dataset)
    preprocessTest(validation_dataset)
    preprocessTest1(test_dataset)
    
    
    
    cnn()
    
