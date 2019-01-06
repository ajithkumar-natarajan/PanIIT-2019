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

labels = pd.read_csv("./training/solution.csv")
category = labels['category'] - 1
m = labels.shape[0]
num_labels = 6

model = MobileNet(weights='imagenet', include_top=False)

def image_to_encoding(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    return x

# utility function to calculate accuracy 
def accuracy(predictions, labels): 
    correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) 
    accu = (100.0 * correctly_predicted) / predictions.shape[0] 
    return accu 

train_Y = np.zeros((m, num_labels), dtype=int)
train_Y[np.arange(m), category] = 1
train_Y = train_Y[:500,:]

train_X = [ ]
test_X = [ ]


print("Processing Training data")
for cnt in range(1, 501):
    train_X.append(image_to_encoding('./training/training/' + str(cnt) + '.png'))

print("Processing Testing Data")
for x in range(1, 401):
    if x % 50 == 0:
        print("Processed -> " + str(x))
    test_X.append(image_to_encoding('./testing/' + str(x) + '.png'))

print("Generating Encoding -> Training Data")
start = time.time()
train_X = np.array(train_X)
train_X = train_X.reshape((-1, 224, 224, 3))
train_X = preprocess_input(train_X)
train_X = model.predict(train_X)
train_X = train_X.reshape((500,-1))
end = time.time()
print("Time taken -> " + str(end-start))

print("Generating Encoding -> Testing Data")
start = time.time()
test_X = np.array(test_X)
test_X = test_X.reshape((-1, 224, 224, 3))
test_X = preprocess_input(test_X)
test_X = model.predict(test_X)
test_X = test_X.reshape((400,-1))
end = time.time()
print("Time taken -> " + str(end-start))

train_X = np.array(train_X, dtype=np.float32)
test_X = np.array(test_X, dtype=np.float32)

print("Encoding Generated")
graph = tf.Graph()

with graph.as_default():
    A = tf.Variable(tf.random_normal(shape=[50176, num_labels]))
    b = tf.Variable(tf.random_normal(shape=[num_labels]))

    data = tf.placeholder(dtype=tf.float32, shape=[None, 50176])
    target = tf.placeholder(dtype=tf.float32, shape=[None, num_labels])

    logits = tf.matmul(data, A) + b
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=logits))

    # Define the learning rate, batch_size etc.
    learning_rate = 0.003
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    goal = opt.minimize(loss)

    train_prediction = tf.nn.softmax(logits)

batch_size = 50
num_steps = 2000
loss_trace = [ ]

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    for step in range(num_steps):
        # pick a randomized offset 
        offset = np.random.randint(0, train_Y.shape[0] - batch_size - 1)

        batch_data = train_X[offset:(offset + batch_size), :]
        batch_labels = train_Y[offset:(offset + batch_size), :]

        # Prepare the feed dict 
        feed_dict = {data : batch_data, target : batch_labels}

        # run one step of computation 
        _, l, predictions = session.run([goal, loss, train_prediction], feed_dict=feed_dict)
        loss_trace.append(l)

        if (step % 500 == 0):
            print("Minibatch loss at step {0}: {1}".format(step, l)) 
            print("Minibatch accuracy: {:.1f}%".format(accuracy(predictions, batch_labels)))
    
    print("Generating Predictions")
    pred = tf.nn.softmax(tf.matmul(test_X, A) + b)
    pred = tf.argmax(pred, axis=1)
    pred = pred + 1
    res = pred.eval()
    ind = np.arange(1,401,1)
    d = {'id': ind, 'category': res}
    df = pd.DataFrame(data=d, dtype=np.int8)
    df.to_csv('submission.csv', index=False)

# Visualization of the results
# loss function
plt.plot(loss_trace)
plt.title('Cross Entropy Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()