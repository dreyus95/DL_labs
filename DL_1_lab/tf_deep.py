import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import  ListedColormap
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import data

NUM_EXAMPLES = 10
CLASSES = 2
DISTRIBUTIONS = 6
SEED = 106


class TFDeep:
    """
        Class that represents a deep neural network implementation in tensorflow.
    """

    def __init__(self, shapes, param_delta=0.1, param_lambda=0.01):
        """Arguments:
           - D: dimensions of each datapoint
           - C: number of classes
           - param_delta: training step
        """
        # definicija podataka i parametara:
        # definirati self.X, self.Yoh_, self.W, self.b
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, shapes[0]])
        self.Yoh_ = tf.placeholder(dtype=tf.float32, shape=[None, shapes[-1]])

        self.weights = []
        self.biases = []
        self.hs = []

        # example: 2 5 3
        for index, shape in enumerate(shapes[1:]):
            self.weights.append(tf.Variable(initial_value=tf.random_normal([shapes[index], shape])))
            self.biases.append(tf.Variable(initial_value=tf.random_normal([1, shape])))

        # NN input
        # self.hs.append(tf.nn.sigmoid(tf.matmul(self.X, self.weights[0]) + self.biases[0]))
        self.hs.append(self.relu(tf.matmul(self.X, self.weights[0]) + self.biases[0]))

        # NN inner connections
        for i in range(1, len(shapes[1:-1])):
            self.hs.append(self.relu(tf.matmul(self.hs[-1], self.weights[i]) + self.biases[i]))
            # self.hs.append(tf.nn.sigmoid(tf.matmul(self.hs[-1], self.weights[i]) + self.biases[i]))

        # NN output
        if len(shapes[1:-1]) == 0:
            # regular logistic regression
            output = tf.matmul(self.X, self.weights[-1]) + self.biases[-1]
        else:
            # neural network last layer output
            output = tf.matmul(self.hs[-1], self.weights[-1]) + self.biases[-1]

        # formulacija modela: izračunati self.probs
        #   koristiti: tf.matmul, tf.nn.softmax
        self.probs = tf.nn.softmax(output)

        # formulacija gubitka: self.loss
        #   koristiti: tf.log, tf.reduce_sum, tf.reduce_mean
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.Yoh_ * tf.log(self.probs + 1e-8), axis=1))
        self.regularization = [param_lambda * tf.nn.l2_loss(weights) for weights in self.weights]
        self.loss = self.cross_entropy + tf.add_n(self.regularization)

        # formulacija operacije učenja: self.train_step
        #   koristiti: tf.train.GradientDescentOptimizer,
        #              tf.train.GradientDescentOptimizer.minimize
        self.train_step = tf.train.GradientDescentOptimizer(param_delta).minimize(self.loss)

        # instanciranje izvedbenog konteksta: self.session
        #   koristiti: tf.Session
        # better than tf.Session(), installs itself as default, we can use obj.eval()
        self.session = tf.InteractiveSession()

    def train(self, X, Yoh_, param_niter):
        """Arguments:
           - X: actual datapoints [NxD]
           - Yoh_: one-hot encoded labels [NxC]
           - param_niter: number of iterations
        """
        # incijalizacija parametara
        #   koristiti: tf.initialize_all_variables
        self.session.run(tf.initialize_all_variables())

        # optimizacijska petlja
        #   koristiti: tf.Session.run
        for i in range(param_niter+1):
            tr = self.session.run([self.train_step], feed_dict={self.X: X, self.Yoh_: Yoh_})
            if i % 1000 == 0:
                loss = self.session.run(self.loss, feed_dict={self.X: X, self.Yoh_: Yoh_})
                print("{0:4}. Loss: {1:.8f}".format(i, loss))

    def eval(self, X):
        """Arguments:
           - X: actual datapoints [NxD]
           Returns: predicted class probabilites [NxC]
        """
        #   koristiti: tf.Session.run
        # ispiši performansu (preciznost i odziv po razredima)
        probs = self.session.run(self.probs, feed_dict={self.X: X})
        return probs

    def classify(self, X):
        return np.argmax(self.eval(X), axis=1)

    def relu(self, input):
        output = tf.nn.relu(input)
        return output

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
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(SEED)
    tf.set_random_seed(SEED)

    # instanciraj podatke X i labele Yoh_
    X, Y_ = data.sample_gmm_2d(DISTRIBUTIONS, CLASSES, NUM_EXAMPLES)

    # use this for debugging
    # colors = ['red', 'green', 'blue']
    # plt.scatter(X[:, 0], X[:, 1], c=Y_.flatten(), cmap=ListedColormap(colors))

    Yoh_ = Y_.reshape(-1)
    Yoh_ = np.eye(CLASSES)[Yoh_]

    shape = [2, 10, CLASSES]

    # izgradi graf:
    tfdeep = TFDeep(shape, param_delta=0.01, param_lambda=0.01)

    # nauči parametre:
    tfdeep.train(X, Yoh_, 10000)

    # dohvati vjerojatnosti na skupu za učenje
    probs = tfdeep.eval(X)

    tfdeep.eval_perf(probs, Yoh_)

    data.plot_decision_boundary(X, lambda x: tfdeep.classify(x))
    # graph the data points
    data.graph_data(X, Y_, np.argmax(probs, axis=1))
