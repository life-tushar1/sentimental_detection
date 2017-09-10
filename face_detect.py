import cv2
import numpy as np
#test=face_cascade.load('haarcascade_frontalface_default.xml')
#print(test)
fc=cv2.CascadeClassifier('/Users/tusharsharma/Downloads/opencv-3.2.0/data/haarcascades/haarcascade_frontalface_default.xml')
mouth=cv2.CascadeClassifier('/Users/tusharsharma/Downloads/opencv-3.2.0/data/haarcascades/haarcascade_Smile.xml')
cam=cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face=fc.detectMultiScale(gray,1.3,5)
    lips=mouth.detectMultiScale(gray,6,20)#minSize=(23,25))
    #lips=mouth.detectMultiScale(gray)
    print lips
    for (a,b,c,d) in face:
        cv2.rectangle(frame,(a,b),(a+c,b+d),(0,0,255),2)

    for (a,b,c,d) in lips:
        #if b<400:
            cv2.rectangle(frame,(a,b),(a+c,b+d),(0,255,0),2)

    cv2.imshow('detect',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
