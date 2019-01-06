import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time

from keras.preprocessing import image
from resnet50 import ResNet50
from mobilenet import MobileNet
from os import listdir
from imagenet_utils import preprocess_input


def image_to_encoding(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    return x

A = tf.Variable(tf.random_normal(shape=[50176, 6]))
b = tf.Variable(tf.random_normal(shape=[6]))

saver = tf.train.Saver()
model = MobileNet(weights='imagenet', include_top=False)

test_X = [ ]
ans = [ ]
for x in range(1, 40001, 1000):
    test_X = [ ]
    for y in range(x, x+1000):
        test_X.append(image_to_encoding('./testing/' + str(y) + '.png'))

    start = time.time()
    test_X = np.array(test_X)
    test_X = test_X.reshape((-1, 224, 224, 3))
    test_X = preprocess_input(test_X)
    test_X = model.predict(test_X)
    test_X = test_X.reshape((1000,-1))
    end = time.time()
    print("Completed iteration -> " + str(x / 1000) + " Time taken -> " + str(end-start))

    with tf.Session() as sess:
        saver.restore(sess, './model.ckpt')
        pred = tf.nn.softmax(tf.matmul(test_X, A) + b)
        pred = tf.argmax(pred, axis=1)
        pred = pred + 1
        res = pred.eval()
        ans = ans + res.tolist()

ind = np.arange(1,40001,1)
d = {'id': ind, 'category': ans}
df = pd.DataFrame(data=d, dtype=np.int8)
df.to_csv('submission.csv', index=False)

    

