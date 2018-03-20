import pandas as pd
import cv2
import numpy as np
from random import shuffle

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

data_file1='./train.csv'
data_file2='./test.csv'

data = pd.read_csv(data_file1)
data2= pd.read_csv(data_file2)
width, height = 48, 48
faces = []
label=[]
training_data=[]
testing_data=[]
for index,row in data.iterrows():
    face = [int(pixel) for pixel in row['pixels'].split(' ')]
    face = np.asarray(face).reshape(width, height)
    lab=label_img(row['emotion'])
    training_data.append([np.array(face),np.array(lab)])
shuffle(training_data)
#print training_data
np.save('train_data.npy',training_data)

faces = []
label=[]
for index,row in data2.iterrows():
    face = [int(pixel) for pixel in row['pixels'].split(' ')]
    face = np.asarray(face).reshape(width, height)
    lab=label_img(row['emotion'])
    testing_data.append([np.array(face),np.array(lab)])
shuffle(testing_data)
#print training_data
np.save('testing_data.npy',testing_data)
