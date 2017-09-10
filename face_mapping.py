import cv2
import dlib
vid=cv2.VideoCapture(0)
dectector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("/Users/tusharsharma/Downloads/shape_predictor_68_face_landmarks.dat")
while True:
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detections = dectector(gray)


    for k,d in enumerate(detections):
        #print k
        #print d
        shape = predictor(gray, d)
        #print shape.part(49).x
        #print "new"
        #print shape.part(49).x
        #print shape.part(55).x
        #print ((shape.part(49).x)/(shape.part(55).x))
        xl=[]
        yl=[]
        for i in range(1,68):
            xl.append(float(shape.part(i).x))
            yl.append(float(shape.part(i).y))
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,255,0), thickness=2)
        #print xl[48]
        #print yl[48]
        #print "new"
        #print xl[3]
        #print yl[3]
        #print "next"
        #s=xl[48]-xl[3]
        #k=yl[48]-yl[3]
        #print s
        #print k
        #m=s/k

        #print m
        #if m>2.8:
            #print "smilly "

        llcX=(xl[54]-xl[14])
        llcY=(yl[54]-yl[14])

        rclX=(xl[2]-xl[48])
        rclY=(yl[48]-yl[2])

        print "left "+str(llcX)+" "+str(llcY)
        print "right "+str(rclX)+" "+str(rclY)

        mx=llcX/rclX
        my=llcY/rclY
        print "mx="+str(mx)+" my="+str(my)
        print "ratio of mx/my "+str(mx/my)
        print "ratio of my/mx "+str(my/mx)
        if my>9.5:
            print "smile"
            cv2.rectangle(frame,(int(xl[2]),int(yl[2])),(int(xl[11]),int(yl[11])),(0,0,255),2)
    cv2.imshow("image", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
