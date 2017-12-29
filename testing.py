import matplotlib.pyplot as plt
import tflearn
import numpy as np
import utilities
import model
import os



def test(MODEL_NAME,TEST_DIR,IMG_SIZE,LR):
    model1=model.neural_network_model(IMG_SIZE,LR)
    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model1.load(MODEL_NAME)
        print "model loaded"
    #test_data=utilities.process_test_data(TEST_DIR,IMG_SIZE)
    test_data=np.load('test_data.npy')
    fig=plt.figure()
    #[neutral,anger,contempt,disgust,fear,happiness,sadness,surprise]
    for num,data in enumerate(test_data[:12]):
        img_num=data[1]
        img_data=data[0]
        y=fig.add_subplot(3,4,num+1)
        orig=img_data
        data=img_data.reshape(IMG_SIZE,IMG_SIZE,1)
        model_out=model1.predict([data])[0]
        str_label=model.label_test(model_out)
        y.imshow(orig,cmap='gray')
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()
