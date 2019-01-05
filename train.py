import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.preprocessing import image
from resnet50 import ResNet50
from os import listdir
from imagenet_utils import preprocess_input


labels = pd.read_csv("./training/solution.csv")
category = labels['category'] - 1
m = labels.shape[0]
num_labels = 6

model = MobileNet(weights='imagenet', include_top=False)
#model = ResNet50(weights='imagenet', include_top=False)

def image_to_encoding(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)
    encoding = model.predict(x)
    encoding = np.squeeze(encoding)
    return encoding

# utility function to calculate accuracy 
def accuracy(predictions, labels): 
    correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) 
    accu = (100.0 * correctly_predicted) / predictions.shape[0] 
    return accu 

train_X = np.array([ ])
train_Y = np.zeros((m, num_labels), dtype=int)
train_Y[np.arange(m), category] = 1
#train_Y = train_Y[:499,:]
#train_Y = np.transpose(train_Y)


for cnt in range(1, 5000):
    print("Processing -> " + str(cnt))
    train_X = np.append(train_X, image_to_encoding('./training/training/' + str(cnt) + '.png'))

train_X = train_X.reshape(-1,2048)
#train_X = np.transpose(train_X)

graph = tf.Graph()

with graph.as_default():
    A = tf.Variable(tf.random_normal(shape=[2048, num_labels]))
    b = tf.Variable(tf.random_normal(shape=[num_labels]))

    data = tf.placeholder(dtype=tf.float32, shape=[None, 2048])
    target = tf.placeholder(dtype=tf.float32, shape=[None, num_labels])

    logits = tf.matmul(data, A) + b
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=logits))

    # Define the learning rate, batch_size etc.
    learning_rate = 0.003
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    goal = opt.minimize(loss)

    train_prediction = tf.nn.softmax(logits)

batch_size = 50
num_steps = 5000

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

        if (step % 100 == 0):
            print("Minibatch loss at step {0}: {1}".format(step, l)) 
            print("Minibatch accuracy: {:.1f}%".format(accuracy(predictions, batch_labels)))