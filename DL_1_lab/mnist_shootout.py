import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

class TFDeep:
    """
        Class that represents a deep neural network implementation in tensorflow.
    """

    def __init__(self, shapes, param_delta=0.1, param_lambda=0.01, optimizer="gds",
                 activation=tf.nn.relu):
        """Arguments:
           - shapes: shape of neural network
           - C: number of classes
           - param_delta: training step
        """
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, shapes[0]])
        self.Yoh_ = tf.placeholder(dtype=tf.float32, shape=[None, shapes[-1]])
        self.is_training = tf.placeholder(tf.bool)

        self.weights = []
        self.biases = []
        self.hs = []

        for index, shape in enumerate(shapes[1:]):
            self.weights.append(tf.Variable(initial_value=tf.random_normal([shapes[index], shape])))
            self.biases.append(tf.Variable(initial_value=tf.random_normal([1, shape])))

        # NN input
        self.hs.append(activation(self.batch_norm(tf.matmul(self.X, self.weights[0]) + self.biases[0],
                                  self.is_training)))

        # NN inner connections
        for i in range(1, len(shapes[1:-1])):
            self.hs.append(activation(self.batch_norm(tf.matmul(self.hs[-1], self.weights[i]) + self.biases[i],
                                      self.is_training)))

        # NN output
        if len(shapes[1:-1]) == 0:
            # regular logistic regression
            output = tf.matmul(self.X, self.weights[-1]) + self.biases[-1]
        else:
            # neural network last layer output
            output = tf.matmul(self.hs[-1], self.weights[-1]) + self.biases[-1]

        self.probs = tf.nn.softmax(output)

        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.Yoh_ * tf.log(self.probs + 1e-8), axis=1))
        self.regularization = [param_lambda * tf.nn.l2_loss(weights) for weights in self.weights]
        self.loss = self.cross_entropy + tf.add_n(self.regularization)

        if optimizer == "gds":
            self.train_step = tf.train.GradientDescentOptimizer(param_delta).minimize(self.loss)
        elif optimizer == "adam":
            step = tf.Variable(0, trainable=False)
            rate = tf.train.exponential_decay(param_delta, step, 100, 0.98, staircase=True)
            self.train_step = tf.train.AdamOptimizer(rate).minimize(self.loss, global_step=step)

        self.session = tf.InteractiveSession()

    def batch_norm(self, inputs, is_training, decay=0.999, epsilon=1e-3):

        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

        def if_true():
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
            train_mean = tf.assign(pop_mean,
                                   pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                                  pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,
                                                 batch_mean, batch_var, beta, scale, epsilon)

        def if_false():
            return tf.nn.batch_normalization(inputs,
                                             pop_mean, pop_var, beta, scale, epsilon)

        result = tf.cond(is_training, if_true, if_false)
        return result

    def train(self, X, Yoh_, param_niter):
        """Arguments:
           - X: actual datapoints [NxD]
           - Yoh_: one-hot encoded labels [NxC]
           - param_niter: number of iterations
        """
        self.session.run(tf.initialize_all_variables())

        for i in range(param_niter+1):
            tr = self.session.run([self.train_step], feed_dict={self.X: X, self.Yoh_: Yoh_})
            if i % 1000 == 0:
                loss = self.session.run(self.loss, feed_dict={self.X: X, self.Yoh_: Yoh_})
                print("{0:4}. Loss: {1:.8f}".format(i, loss))

    def _shuffle(self, X, Yoh_):
        perm = np.random.permutation(len(X))
        return X[perm], Yoh_[perm]

    def _split_dataset(self, X, Yoh_, ratio=0.8):
        X, Yoh_ = self._shuffle(X, Yoh_)
        split = int(ratio * len(X))
        return X[:split], X[split:], Yoh_[:split], Yoh_[split:]

    def train_mb(self, X, Yoh_, n_epochs=1000, batch_size=50, train_ratio=1., print_step=10):
        self.session.run(tf.initialize_all_variables())
        prev_loss = window_loss = float('inf')

        X_train, X_val, Y_train, Y_val = self._split_dataset(X, Yoh_, ratio=train_ratio)
        n_samples = len(X_train)
        n_batches = int(n_samples / batch_size)

        for epoch in range(n_epochs):
            X_train, Y_train = self._shuffle(X_train, Y_train)
            i = 0
            avg_loss = 0

            while i < n_samples:
                batch_X, batch_Yoh_ = X_train[i:i + batch_size], Y_train[i:i + batch_size]
                data_dict = {self.X: batch_X, self.Yoh_: batch_Yoh_, self.is_training: True}
                val_loss, _ = self.session.run([self.loss, self.train_step], feed_dict=data_dict)

                avg_loss += val_loss / n_batches
                i += batch_size

            # validation
            data_dict = {self.X: X_val, self.Yoh_: Y_val, self.is_training: False}
            val_loss, _ = self.session.run([self.loss, self.train_step], feed_dict=data_dict)
            window_loss = min(window_loss, val_loss)
            if epoch % 50 == 0:
                if window_loss > prev_loss:
                    print("Early stopping: epoch", epoch)
                    # break
                prev_loss = window_loss
                window_loss = float('inf')

            if epoch % print_step == 0:
                print("Epoch: {:4d}; avg_train_loss {:.9f}; validation_loss {:.9f}".format(epoch, avg_loss, val_loss))

        print("Optimization Finished!")
        print("Validation loss {:.9f}".format(val_loss))

    def eval(self, X):
        """Arguments:
           - X: actual datapoints [NxD]
           Returns: predicted class probabilites [NxC]
        """
        probs = self.session.run(self.probs, feed_dict={self.X: X})
        return probs

    def classify(self, X):
        return np.argmax(self.eval(X), axis=1)

    def eval_perf(self, Y, Y_):
        # needed to compute scores of our model
        # 'weighted' takes into consideration labels imbalance
        if max(int(max(Y_) + 1), int(max(Y) + 1)) == 2:
            average = 'binary'
        else:
            average = 'weighted'

        accuracy = accuracy_score(Y_, Y)
        precision = precision_score(Y_, Y, average=average)
        recall = recall_score(Y_, Y, average=average)
        f1 = f1_score(Y_, Y, average=average)

        print("Accuracy: {0:.3f}\n"
              "Precision: {1:.3f}\n"
              "Recall: {2:.3f}\n"
              "F1: {3:.3f} ".format(accuracy, precision, recall, f1))


if __name__ == "__main__":
    train_images = []
    train_images_labels = []
    test_images = []
    test_images_labels = []

    for i, image in enumerate(mnist.train.images):
        train_images.append(np.array(image, dtype='float32'))

    for label in mnist.train.labels:
        train_images_labels.append(label)

    for i, image in enumerate(mnist.test.images):
        test_images.append(np.array(image, dtype='float32'))

    for label in mnist.test.labels:
        test_images_labels.append(label)

    shape2 = [784, 10]

    tfdeep2 = TFDeep(shape2, param_delta=0.015, param_lambda=0.01, optimizer="gds")
    tfdeep2.train_mb(np.array(train_images), np.array(train_images_labels), train_ratio=0.8)
    probs2 = tfdeep2.eval(np.array(test_images))
    tfdeep2.eval_perf(np.argmax(probs2, axis=1), np.argmax(test_images_labels, axis=1))