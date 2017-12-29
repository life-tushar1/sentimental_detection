import cv2
import numpy as np
import sys
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow
#test=face_cascade.load('haarcascade_frontalface_default.xml')
#print(test)
font=cv2.FONT_HERSHEY_SIMPLEX
LR=1e-3
IMG_SIZE=50
fc=cv2.CascadeClassifier('/Users/tusharsharma/opencv-3.2.0/data/haarcascades/haarcascade_upperbody.xml')
MODEL_NAME ='sentimentaldetection-{}-{}'.format(LR,'test-5-conv-basic')

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)






convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.5)

convnet = fully_connected(convnet, 8, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
convnet = dropout(convnet, 0.5)

model = tflearn.DNN(convnet,tensorboard_dir='log')

model.load(MODEL_NAME)
cam=cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    frame2=cv2.resize(frame,(100,100))
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    face=fc.detectMultiScale(gray,1.01,5)

    for (a,b,c,d) in face:#x,y,width,height
        #cv2.rectangle(frame2,(a,b),(a+c,b+d),(0,0,255),2)
        img2=frame2[b:b+d,a:a+c]
        img2=cv2.resize(cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY),(IMG_SIZE,IMG_SIZE))
        cv2.imshow('test',img2)
        cvt=np.array(img2)
        data=cvt.reshape(IMG_SIZE,IMG_SIZE,1)
        model_out=model.predict([data])[0]
        if np.argmax(model_out)==1:
            str_label='anger'
        elif np.argmax(model_out)==2:
            str_label='contempt'
        elif np.argmax(model_out)==3:
            str_label='disgust'
        elif np.argmax(model_out)==4:
            str_label='fear'
        elif np.argmax(model_out)==5:
            str_label='happiness'
        elif np.argmax(model_out)==6:
            str_label='sadness'
        elif np.argmax(model_out)==7:
            str_label='surprise'
        elif np.argmax(model_out)==0:
            str_label='neutral'
        print str_label
        cv2.putText(frame,str_label,(0,50),font,0.8,(0,255,0),2,cv2.LINE_AA)
    '''for (a,b,c,d) in lips:
        #if b<400:
            cv2.rectangle(frame,(a,b),(a+c,b+d),(0,255,0),2)'''

    cv2.imshow('detect',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
