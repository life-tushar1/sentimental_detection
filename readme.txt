IMG SIZE=48
LR=1e-4
for model-4

to create model 4 go to data_creation dir-> download FER2013 dataset from kraggle and store in this Directory.
run the cvsCreate.py then run createData.py

copy train_data.npy and test_data.npy from this dir to train-test

run main.py.

after you select train a log dir will be created

staying in train-test dir type the following command on terminal

tensorboard --logdir="log"

here you will be able to see the graph of model.
