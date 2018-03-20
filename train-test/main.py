import train
import testing
import face_detect
IMG_SIZE=48
LR=1e-4

MODEL_NAME ='sentimentaldetection-{}-{}'.format(LR,'sentimental4')


print "do u want to train, test or run"
x=raw_input()
if x=="train":
    train.train(IMG_SIZE,LR,MODEL_NAME)
if x=="test":
    testing.test(MODEL_NAME,IMG_SIZE,LR)
if x=="run":
    face_detect.predict(LR,IMG_SIZE,MODEL_NAME)
