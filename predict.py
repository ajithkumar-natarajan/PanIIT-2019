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
for x in range(1, 40001):
    test_X.append(image_to_encoding('./testing/' + str(x) + '.png'))

X = np.array([[]])
test_X = np.array(test_X)
test_X = test_X.reshape((-1,224,224,3))
test_X = preprocess_input(test_X)

for x in range(0, 40000, 1000):
    start = time.time()
    temp = model.predict(test_X[x:x+1000,:])
    temp = temp.reshape((1000,-1))
    if X.shape[1] == 0:
        X = temp
    else:
        X = np.concatenate((X,temp), axis=0)
    end = time.time()
    print("Completed iteration -> " + str(x / 1000) + " Time taken -> " + str(end-start))

data = tf.placeholder(dtype=tf.float32, shape=[None, 50176])
a = tf.nn.softmax(tf.matmul(data, A) + b)
b = tf.argmax(a, axis=1)
pred = b + 1

with tf.Session() as sess:
    saver.restore(sess, './model.ckpt')

    for x in range(0, 40000, 1000):
        test_X = X[x:x+10,:]
        test_X = np.array(test_X, dtype=np.float32)
        prediction = sess.run(pred, feed_dict={data: test_X})
        ans = ans + prediction.tolist()
        print("Processed ->" + str(x))


ind = np.arange(1,40001,1)
d = {'id': ind, 'category': ans}
df = pd.DataFrame(data=d, dtype=np.int8)
df.to_csv('submission.csv', index=False)

    

