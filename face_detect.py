import cv2
import numpy as np
import sys
import tflearn
import tensorflow
import model

#test=face_cascade.load('haarcascade_frontalface_default.xml')
#print(test)
font=cv2.FONT_HERSHEY_SIMPLEX
LR=1e-4
IMG_SIZE=50
fc=cv2.CascadeClassifier('/Users/tusharsharma/opencv-3.2.0/data/haarcascades/haarcascade_upperbody.xml')
MODEL_NAME ='sentimentaldetection-{}-{}'.format(LR,'test-5-conv-basic')

model1=model.neural_network_model(IMG_SIZE,LR)
model1.load(MODEL_NAME)


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
        model_out=model1.predict([data])[0]
        
        str_label=model.label_test(model_out)
        print str_label
        cv2.putText(frame,str_label,(0,50),font,0.8,(0,255,0),2,cv2.LINE_AA)

    cv2.imshow('detect',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
