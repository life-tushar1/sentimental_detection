import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import numpy as np


def neural_network_model(IMG_SIZE,LR):
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 128, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 128, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 8, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    convnet = dropout(convnet, 0.8)

    model = tflearn.DNN(convnet,tensorboard_dir='log')
    return model



def train_model(train_data,IMG_SIZE,LR,MODEL_NAME,model=False):
    #fit
    #basically im reducing 100 imgs from below
    train=train_data[:-100]
    #the reduced 100 imgs are used as testing
    test=train_data[-100:]

    #classic ml code (train data)
    X=np.array([i[0]for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    Y=[i[1] for i in train]


    #naming the model
    if not model:
        model=neural_network_model(IMG_SIZE,LR)

    #test accuracy (same as classic ml code for train)
    test_x=np.array([i[0]for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    test_y=[i[1]for i in test]


    #fitting the model into the network
    model.fit({'input': X}, {'targets': Y}, n_epoch=8, validation_set=({'input': test_x}, {'targets': test_y}),
        snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

    #saving the model
    model.save(MODEL_NAME)
