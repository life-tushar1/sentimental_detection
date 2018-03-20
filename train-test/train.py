import cv2
import numpy as np
import os
import model
import testing
import tensorflow as tf

def train(IMG_SIZE,LR,MODEL_NAME):
    tf.reset_default_graph()
    model.train_model4(IMG_SIZE,LR,MODEL_NAME)
