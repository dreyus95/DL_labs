import tensorflow as tf
import numpy as np
import math
import os
import skimage as ski
import skimage.io

from tensorflow.contrib.layers import xavier_initializer_conv2d as xavier_conv2d
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def draw_conv_filters(epoch, step, weights, save_dir):
  # kxkxCxn_filters
  k, k, C, num_filters = weights.shape

  w = weights.copy().swapaxes(0, 3).swapaxes(1,2)
  w = w.reshape(num_filters, C, k, k)
  w -= w.min()
  w /= w.max()

  border = 1
  cols = 8
  rows = math.ceil(num_filters / cols)
  width = cols * k + (cols-1) * border
  height = rows * k + (rows-1) * border

  for i in range(1):
    img = np.zeros([height, width])
    for j in range(num_filters):
      r = int(j / cols) * (k + border)
      c = int(j % cols) * (k + border)
      img[r:r+k,c:c+k] = w[j,i]
    filename = 'epoch_%02d_step_%06d_input_%03d.png' % (epoch, step, i)
    ski.io.imsave(os.path.join(save_dir, filename), img)


def conv_2d(tensor, filters, biases, strides=1, activation=tf.nn.relu):
    h1 = tf.nn.conv2d(tensor, filters, strides=[1, strides, strides, 1],
                          padding='SAME')
    h1 = tf.nn.bias_add(h1, biases)
    return activation(h1)


def max_pool_2d(tensor, k_size=None, padding='SAME'):
    return tf.nn.max_pool(tensor, ksize=[1, k_size, k_size, 1], strides=[1, k_size, k_size, 1], padding=padding)


def dropout(tensor, use_dropout, rate=0.5):
    return tf.layers.dropout(tensor, rate=rate, training=use_dropout)


# FC layer
def dense(tensor, filters, biases, activation=None):
    tensor = tf.reshape(tensor, [-1, filters.get_shape().as_list()[0]])
    res = tf.matmul(tensor, filters) + biases
    if activation:
        return activation(res)
    return res


OUTPUT_SHAPE = 10
weight_decay = 1e-4

x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SHAPE])
weights = {
    'conv1': tf.get_variable('w_conv1', [5, 5, 1, 16], initializer=xavier_conv2d()),
    'conv2': tf.get_variable('w_conv2', [5, 5, 16, 32], initializer=xavier_conv2d()),

    'fc3': tf.get_variable('w_fc3', [7 * 7 * 32, 512], initializer=xavier_conv2d()),
    'fc4': tf.get_variable('w_fc4', [512, OUTPUT_SHAPE], initializer=xavier_conv2d())
}

biases = {
    'conv1': tf.Variable(tf.zeros([16]), name='b_conv1'),
    'conv2': tf.Variable(tf.zeros([32]), name='b_conv2'),
    'fc3': tf.Variable(tf.zeros([512]), name='b_fc3'),
    'fc4': tf.Variable(tf.zeros([OUTPUT_SHAPE]), name='b_fc4')
}

use_dropout = tf.placeholder(tf.bool, name='use_dropout')

#################### 1ST LAYER ####################
input_x = tf.reshape(x, [-1, 28, 28, 1])

# h1 is [batch_size, 28, 28, 16]
h1 = conv_2d(input_x, weights["conv1"], biases["conv1"])
print("H1:", h1.get_shape())

# max pool convolved layer [batch_size, 14, 14, 16]
h1_pooled = max_pool_2d(h1, k_size=2)
print("H1 pooled:", h1_pooled.get_shape())

#################### 2ND LAYER ####################
# h2 is [batch_size, 14, 14, 32]
h2 = conv_2d(h1_pooled, weights["conv2"], biases["conv2"])
print("H2:", h2.get_shape())

# max pool convolved layer [batch_size, 7, 7, 32]
h2_pooled = max_pool_2d(h2, k_size=2)
print("H2 pooled:", h2_pooled.get_shape())

#################### FC LAYER ####################
dropout = dropout(h2_pooled, use_dropout)  # adding dropout to reduce overfitting
print("Dropout:", dropout.get_shape())

fc1 = dense(dropout, weights['fc3'],  biases['fc3'], activation=tf.nn.relu)
print("FC1:", fc1.get_shape())

#################### FC LAYER ####################
# logits is [batch_size, 10]
logits = dense(fc1, weights['fc4'],  biases['fc4'], activation=None)
print("Logits:", logits.get_shape())


#################### LOSS & TRAIN STEP ####################
regularizers = 0
for w in [weights['conv1'], weights['conv2'], weights['fc3']]:
    regularizers += tf.nn.l2_loss(w)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

loss = loss + weight_decay*regularizers

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

lr = tf.placeholder(tf.float32)
train_step = tf.train.AdamOptimizer(lr).minimize(loss)


#################### MAIN CODE ####################
err_train = [[] for _ in range(2)]
acc_train = [[] for _ in range(2)]
err_test = [[] for _ in range(2)]
acc_test = [[] for _ in range(2)]


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    max_epochs = 8
    batch_size = 250
    lr_policy = 1e-4
    num_examples = mnist.train.images.shape[0]
    num_batches = num_examples // batch_size

    train_x = mnist.train.images
    train_x = train_x.reshape([-1, 28, 28, 1])
    train_y = mnist.train.labels

    valid_x = mnist.validation.images
    valid_x = valid_x.reshape([-1, 28, 28, 1])
    valid_y = mnist.validation.labels

    test_x = mnist.test.images
    test_x = test_x.reshape([-1, 28, 28, 1])
    test_y = mnist.test.labels

    train_mean = train_x.mean()
    train_x -= train_mean
    valid_x -= train_mean
    test_x -= train_mean

    for epoch in range(1, max_epochs + 1):
        cnt_correct = 0

        permutation_idx = np.random.permutation(num_examples)
        train_x = train_x[permutation_idx]
        train_y = train_y[permutation_idx]

        for i in range(num_batches):
            # store mini-batch to ndarray
            batch_x = train_x[i * batch_size:(i + 1) * batch_size, :]
            batch_y = train_y[i * batch_size:(i + 1) * batch_size, :]

            data_dict = {x: batch_x, y: batch_y, lr: lr_policy, use_dropout: True}
            loss_val, acc, _ = sess.run([loss, accuracy, train_step], feed_dict=data_dict)

            if i % 5 == 0:
                print("epoch %d, step %d/%d, batch loss = %.4f" % (epoch, i * batch_size, num_examples, loss_val))
            if i % 100 == 0:
                w = sess.run(weights['conv1'])
                draw_conv_filters(epoch, i * batch_size, w, "output_mnist_conv/")
            if i > 0 and i % 50 == 0:
                print("Train accuracy = %.4f" % acc)

        print("Train accuracy = %.4f" % acc)
        # evaluate(sess, "Validation", valid_x, valid_y, config)