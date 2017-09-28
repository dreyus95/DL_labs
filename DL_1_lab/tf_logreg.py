import tensorflow as tf
import numpy as np

import data

NUM_EXAMPLES = 50
CLASSES = 2
DISTRIBUTIONS = 6


class TFLogreg:

    def __init__(self, D, C, param_delta=0.1):
        """Arguments:
           - D: dimensions of each datapoint
           - C: number of classes
           - param_delta: training step
        """
        # definicija podataka i parametara:
        # definirati self.X, self.Yoh_, self.W, self.b
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, D])
        self.Yoh_ = tf.placeholder(dtype=tf.float32, shape=[None, C])
        self.W = tf.Variable(initial_value=tf.zeros([D, C]))
        self.b = tf.Variable(initial_value=tf.zeros([1, C]))

        # formulacija modela: izračunati self.probs
        #   koristiti: tf.matmul, tf.nn.softmax
        self.probs = tf.nn.softmax(tf.matmul(self.X, self.W) + self.b)

        # formulacija gubitka: self.loss
        #   koristiti: tf.log, tf.reduce_sum, tf.reduce_mean
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.Yoh_ * tf.log(self.probs), reduction_indices=[1]))

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
        for _ in range(param_niter):
            self.session.run(self.train_step, feed_dict={self.X: X, self.Yoh_: Yoh_})

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

    def eval_perf(self, Y, Y_):
        predicted = tf.argmax(Y, axis=1)
        correct = tf.argmax(Y_, axis=1)

        TP = tf.count_nonzero(predicted * correct, dtype=tf.float32)
        TN = tf.count_nonzero((predicted - 1) * (correct - 1), dtype=tf.float32)
        FP = tf.count_nonzero(predicted * (correct - 1), dtype=tf.float32)
        FN = tf.count_nonzero((predicted - 1) * correct, dtype=tf.float32)

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)

        acc, prec, rec, f_1 = self.session.run([accuracy, precision, recall, f1], feed_dict={self.X: X})

        print("Accuracy: {0:.3f}\n"\
              "Precision: {1:.3f}\n"\
              "Recall: {2:.3f}\n"\
              "F1: {3:.3f} ".format(acc, prec, rec, f_1))


if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(103)
    tf.set_random_seed(103)

    # instanciraj podatke X i labele Yoh_
    X, Y_ = data.sample_gmm_2d(DISTRIBUTIONS, CLASSES, NUM_EXAMPLES)

    Yoh_ = Y_.reshape(-1)
    Yoh_ = np.eye(CLASSES)[Yoh_]

    # izgradi graf:
    tflr = TFLogreg(X.shape[1], Yoh_.shape[1], 0.15)

    # nauči parametre:
    tflr.train(X, Yoh_, 10000)

    # dohvati vjerojatnosti na skupu za učenje
    probs = tflr.eval(X)

    tflr.eval_perf(probs, Yoh_)

    # iscrtaj rezultate, decizijsku plohu
    data.plot_decision_boundary(X, lambda x: tflr.classify(x))
    # graph the data points
    data.graph_data(X, Y_, np.argmax(probs, axis=1))