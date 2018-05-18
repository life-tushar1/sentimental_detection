
IMG SIZE=48
LR=1e-4
for model-4

Requirements.

1- tensorflow CPU -python2 <br>
2- tflearn -python2 <br>
3- openCv2 -python2 <br>
4- Numpy -python2 <br>
5- pandas -python2 <br>
6- matplotlib -python2 <br>
7- python2 <br>
<br>
to create model 4 or test on a FER2013 dataset go to <b>data_creation directory</b>download FER2013 dataset from kraggle and store in this Directory.<br>
first run the <b>cvsCreate.py</b> then run <b>createData.py</b> <br>

copy <b>train_data.npy</b> and <b>test_data.npy</b> from this current directory to <b>train-test</b> directory<br><br>


step 1. Go to <b>train-test</b><br><br>

step 2. run main.py<br><br>

step 3. if you are opting for training please follow the above instructions for data creation. Recommended to increase the n_epoc from 3 to 24 in <b>model.py</b> for higher accuracy, staying in train-test dir type the following command on terminal
<b>tensorboard --logdir="log"</b> for graphical view how the model is performing. Type this command once log dir is visble in <b>train-test</b> directory.<br><br>

step 4. IF YOU WANT TO USE PRE-TRAINED MODEL:
go to <b>train-test</b> directory then into <b>model</b> directory copy all content and save in <b>train-test</b> directory.<br><br>

step 5. For testing we need a model, first if u want to use the pre-trained model follow step 4 else if you want to create your own model follow step 3. After that select test when you execute <b>main.py</b>.<br><br>

step 6. If you to run a prediction on live-data :
go to <b>train-test</b> directory but we need a model, first if u want to use the pre-trained model follow step 4 else if you want to create your own model follow step 3. After that select test when you execute <b>main.py</b>.After that select run when you execute <b>main.py</b>.<br><br>

step 7. If you want to run a analysis on a video file :
go to <b>Video</b> directory, remember to rename the video file into out.mov (right now it only supports mov format).The code is file name sensitive. To run this we need a model, first if u want to use the pre-trained model follow step 4 else if you want to create your own model follow step 3, copy all the new content generated from train-test if u have followed step 3 else simply copy all the contents on <b>model</b> directory to <b>Video</b> directory. Execute <b>start.py</b>. After a while a new file will be generated named out2.mov. In here is your result stored.<br><br>
