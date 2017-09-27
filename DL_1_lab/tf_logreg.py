import tensorflow as tf
import numpy as np

import data

NUM_EXAMPLES = 50
CLASSES = 2
DISTRIBUTIONS = 2


class TFLogreg:
  def __init__(self, D, C, param_delta=0.5):
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
    probs = self.session.run(self.probs, feed_dict={self.X: X})
    return probs


if __name__ == "__main__":
  # inicijaliziraj generatore slučajnih brojeva
  np.random.seed(101)
  tf.set_random_seed(101)

  # instanciraj podatke X i labele Yoh_
  X, Y_ = data.sample_gmm_2d(DISTRIBUTIONS, CLASSES, NUM_EXAMPLES)

  classes = int(max(Y_) + 1)

  Yoh_ = Y_.reshape(-1)
  Yoh_ = np.eye(classes)[Yoh_]

  # izgradi graf:
  tflr = TFLogreg(X.shape[1], Yoh_.shape[1], 0.5)

  # nauči parametre:
  tflr.train(X, Yoh_, 1000)

  # dohvati vjerojatnosti na skupu za učenje
  probs = tflr.eval(X)

  # ispiši performansu (preciznost i odziv po razredima)
  print(np.count_nonzero(np.argmax(probs, axis=1) == Y_.flatten()))

  # iscrtaj rezultate, decizijsku plohu