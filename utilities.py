from tqdm import tqdm
import os
from random import shuffle
import cv2
import numpy as np

def label_img(img):
    #print img
    #[neutral,anger,contempt,disgust,fear,happiness,sadness,surprise]
    word_label=img.split('.')[-3]
    if word_label=='neutral':
        return[1,0,0,0,0,0,0,0]
    elif word_label=='anger':
        return[0,1,0,0,0,0,0,0]
    elif word_label=='contempt':
        return[0,0,1,0,0,0,0,0]
    elif word_label=='disgust':
        return[0,0,0,1,0,0,0,0]
    elif word_label=='fear':
        return[0,0,0,0,1,0,0,0]
    elif word_label=='happiness':
        return[0,0,0,0,0,1,0,0]
    elif word_label=='sadness':
        return[0,0,0,0,0,0,1,0]
    elif word_label=='surprise':
        return[0,0,0,0,0,0,0,1]

def create_train_data(TRAIN_DIR,IMG_SIZE):
    training_data=[]
    for img in tqdm(os.listdir(TRAIN_DIR)):
        #print img
        if img !='.DS_Store':
            label=label_img(img)
            #print label
            path=os.path.join(TRAIN_DIR,img)
            img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
            training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    #print training_data
    np.save('train_data.npy',training_data)
    return training_data

def process_test_data(TEST_DIR,IMG_SIZE):
    testing_data=[]
    for img in tqdm(os.listdir(TEST_DIR)):
        path=os.path.join(TEST_DIR,img)
        img_num=img.split('.')[0]
        img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img),img_num])
    np.save('test_data.npy',testing_data)
    return testing_data
