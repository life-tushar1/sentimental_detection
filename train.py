import cv2
import numpy as np
import os
import utilities
import model
import testing
import tensorflow as tf


tf.reset_default_graph()

TRAIN_DIR='/users/tusharsharma/desktop/project/sentimental_data/data'
TEST_DIR='/users/tusharsharma/desktop/project/sentimental_data/test'
IMG_SIZE=50
LR=1e-4

MODEL_NAME ='sentimentaldetection-{}-{}'.format(LR,'test-5-conv-basic')

#remove comment incase we dont have the train data else put comment below
#train_data=utilities.create_train_data(TRAIN_DIR,IMG_SIZE)

#if we have array of train data else put comment below
#train_data=np.load('train_data.npy')

#model creation and training  head to model.py for more details

#model=model.train_model(train_data,IMG_SIZE,LR,MODEL_NAME)

#testing of model
#testing.test(MODEL_NAME,TEST_DIR,IMG_SIZE,LR)
