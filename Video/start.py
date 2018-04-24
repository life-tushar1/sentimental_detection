import cv2
import os
import numpy as np
import sys
import tflearn
import tensorflow
import model
IMG_SIZE=48
LR=1e-4
MODEL_NAME ='sentimentaldetection-{}-{}'.format(LR,'sentimental4')
font=cv2.FONT_HERSHEY_SIMPLEX

fc=cv2.CascadeClassifier('/Users/tusharsharma/opencv-3.2.0/data/haarcascades/haarcascade_frontalface_default.xml')
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
fourcc=cv2.VideoWriter_fourcc('m','p','4','v')
out=cv2.VideoWriter()
success=out.open("out2.mov",fourcc,20.0,(1280,720),True)
cap = cv2.VideoCapture('out.mov')
ret=True
c=1

model1=model.neural_network_model4(IMG_SIZE,LR)
model1.load(MODEL_NAME)
t=0
str_label=""


while(ret):
    ret, frame = cap.read()
    if(ret):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face=fc.detectMultiScale(gray,1.3,5)
        if(len(face)>0):
            for (a,b,c,d) in face:#x,y,width,height
                cv2.rectangle(frame,(a,b),(a+c,b+d),(0,0,255),2)
                img2=frame[b:b+d,a:a+c]
                img2=cv2.resize(cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY),(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_CUBIC)
                #cv2.imshow('test',img2)
                cvt=np.array(img2)
                data=cvt.reshape(IMG_SIZE,IMG_SIZE,1)
                model_out=model1.predict([data])[0]
                #print t
                #print model_out[2]

                for index, emotion in enumerate(EMOTIONS):
                    cv2.putText(frame, emotion, (10, index * 20 + 20), font, 0.5, (0, 255, 255), 1)
                    cv2.rectangle(frame, (130, index * 20 + 10), (130 + int(model_out[index] * 100), (index + 1) * 20 + 4), (255, 255, 0), -1)

                if t==5:
                    str_label=model.label_test2(model_out)
                    #print str_label
                    t=0
                cv2.putText(frame,str_label,(a,b),font,0.8,(0,255,0),2,cv2.LINE_AA)
        if(len(face)<1):
            for index, emotion in enumerate(EMOTIONS):
                cv2.putText(frame, emotion, (10, index * 20 + 20), font, 0.5, (0, 0, 255), 1)
                cv2.rectangle(frame, (130, index * 20 + 10), (130 + int(0* 100), (index + 1) * 20 + 4), (0, 0, 255), -1)
        t+=1
        if t==6:
            t=0


        out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
