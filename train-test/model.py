import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d ,avg_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected ,flatten
from tflearn.layers.estimator import regression
import numpy as np
import pandas as pd
from random import shuffle
import matplotlib.pyplot as plt
import os

def label_img(word_label):
    ##(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
    if word_label==6:
        return[0,0,0,0,0,0,1]
    elif word_label==0:
        return[1,0,0,0,0,0,0]
    elif word_label==1:
        return[0,1,0,0,0,0,0]
    elif word_label==2:
        return[0,0,1,0,0,0,0]
    elif word_label==3:
        return[0,0,0,1,0,0,0]
    elif word_label==4:
        return[0,0,0,0,1,0,0]
    elif word_label==5:
        return[0,0,0,0,0,1,0]

def label_test2(model_out):
    #(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
    if np.argmax(model_out)==0:
        str_label='anger'
    elif np.argmax(model_out)==1:
        str_label='disgust'
    elif np.argmax(model_out)==2:
        str_label='fear'
    elif np.argmax(model_out)==3:
        str_label='happiness'
    elif np.argmax(model_out)==4:
        str_label='sadness'
    elif np.argmax(model_out)==5:
        str_label='surprise'
    elif np.argmax(model_out)==6:
        str_label='neutral'
    return str_label


def neural_network_model4(IMG_SIZE,LR):
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')


    convnet = conv_2d(convnet, 64, (5, 5), activation='relu')
    convnet = max_pool_2d(convnet, (5, 5),strides=(2,2))


    convnet = conv_2d(convnet, 64, (3, 3), activation='relu')
    convnet = conv_2d(convnet, 64, (3, 3), activation='relu')
    convnet = avg_pool_2d(convnet, (3, 3),strides=(2,2))
    convnet = dropout(convnet, 0.7)

    convnet = conv_2d(convnet, 128, (3, 3), activation='relu')
    convnet = conv_2d(convnet, 128, (3, 3), activation='relu')
    convnet = avg_pool_2d(convnet, (3, 3),strides=(2,2))
    convnet = dropout(convnet, 0.7)

    convnet = flatten(convnet)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.7)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.7)

    convnet = fully_connected(convnet, 7, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    convnet = dropout(convnet, 0.7)

    model = tflearn.DNN(convnet,tensorboard_dir='log')
    return model

def train_model4(IMG_SIZE,LR,MODEL_NAME,model=False):
    train=np.load('train_data.npy')
    test=np.load('testing_data.npy')

    #classic ml code (train data)
    X=np.array([i[0]for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    Y=[i[1] for i in train]


    #naming the model
    if not model:
        model=neural_network_model4(IMG_SIZE,LR)

    model.load(MODEL_NAME)

    #print "yo"

    #test accuracy (same as classic ml code for train)
    test_x=np.array([i[0]for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    test_y=[i[1]for i in test]

    #print "bae"
    #fitting the model into the network
    model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}),
        snapshot_step=256, show_metric=True, run_id=MODEL_NAME)

    #saving the model
    model.save(MODEL_NAME)
