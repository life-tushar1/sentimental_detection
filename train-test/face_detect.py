import cv2
import numpy as np
import sys
import tflearn
import tensorflow
import model

#test=face_cascade.load('haarcascade_frontalface_default.xml')
#print(test)
def predict(LR,IMG_SIZE,MODEL_NAME):
    font=cv2.FONT_HERSHEY_SIMPLEX
    fc=cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

    model1=model.neural_network_model4(IMG_SIZE,LR)
    model1.load(MODEL_NAME)

    t=0
    cam=cv2.VideoCapture(0)
    str_label=""
    while True:
        ret, frame = cam.read()
        #frame2=cv2.resize(frame,(300,300))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face=fc.detectMultiScale(gray,1.3,5)

        for (a,b,c,d) in face:#x,y,width,height
            cv2.rectangle(frame,(a,b),(a+c,b+d),(0,0,255),2)
            img2=frame[b:b+d,a:a+c]
            img2=cv2.resize(cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY),(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_CUBIC)
            cv2.imshow('test',img2)
            cvt=np.array(img2)
            data=cvt.reshape(IMG_SIZE,IMG_SIZE,1)
            model_out=model1.predict([data])[0]
            #print t
            if t==5:
                str_label=model.label_test2(model_out)
                print str_label
                t=0
            cv2.putText(frame,str_label,(a,b),font,0.8,(0,255,0),2,cv2.LINE_AA)
        t+=1
        if t==6:
            t=0
        cv2.imshow('detect',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
