import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

import data

HIDDEN_LAYER_DIM = 5
SEED = 118
NUM_EXAMPLES = 10
CLASSES = 3
DISTRIBUTIONS = 6
REG_LAMBDA = 1e-3


def relu(input):
    output = np.maximum(0, input)
    return output


def softmax(z, sum):
    res = np.divide(z, sum)
    return res


def fcann2_train(X, Y_, param_niter=1e5, param_delta=0.07):
    """
        Method that performs training part of a Fully Connected Artificial Neural Network
    :param X: dataset
    :param Y_: true classes
    :param param_niter: number of iterations
    :param param_delta: strength of update on each iteration
    :return:
    """
    D = X.shape[1]
    C = Y_.shape[1]
    N = X.shape[0]

    W1 = np.random.randn(D, HIDDEN_LAYER_DIM)
    b1 = np.random.randn(1, HIDDEN_LAYER_DIM)

    W2 = np.random.randn(HIDDEN_LAYER_DIM, C)
    b2 = np.random.randn(1, C)

    model = {}

    prev_loss = 9999

    for i in range(int(param_niter)+1):

        ###### HIDDEN LAYER PASS ######
        S1 = np.dot(X, W1) + b1
        H1 = relu(S1)
        S2 = np.dot(H1, W2) + b2

        ##### SOFTMAX PART #####
        exp_scores = np.exp(S2)
        sumexp = np.sum(exp_scores, axis=1, keepdims=True)
        probs = softmax(exp_scores, sumexp)

        ##### UPDATES WEIGHTS #####
        dS2 = (probs - Y_) / N

        dS1 = np.dot(dS2, W2.T)
        dS1[H1 <= 0] = 0  # reLU iz max(0, x), derivative is 0 if values are < 0

        dW2 = np.dot(H1.T, dS2)
        db2 = np.sum(dS2, axis=0, keepdims=True)

        dW1 = np.dot(X.T, dS1)
        db1 = np.sum(dS1, axis=0, keepdims=True)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += REG_LAMBDA * W2
        dW1 += REG_LAMBDA * W1

        # Gradient descent parameter update
        W1 += -param_delta * dW1
        b1 += -param_delta * db1
        W2 += -param_delta * dW2
        b2 += -param_delta * db2

        correct_class_prob = probs[range(len(X)), np.argmax(Y_, axis=1)]
        correct_class_logprobs = -np.log(correct_class_prob)  # N x 1
        loss = correct_class_logprobs.sum()

        if prev_loss > loss:
            # Assign new parameters to the model
            model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'best_iter': i}
            prev_loss = loss

        # dijagnostički ispis
        if i % 1000 == 0 and i != 0:
            print("iteration {}: loss {}".format(i, loss))

    return model


def fcann2_classify(model, X):
    """
        Method that performs classification based on trained NN model.
    :param model: Trained NN model given from fcann2_train method
    :param X: dataset
    :return: classifications for each sample in X
    """
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = X.dot(W1) + b1
    a1 = np.array(relu(z1))
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


if __name__ == "__main__":
    np.random.seed(SEED)

    # get the training dataset
    X, Y_ = data.sample_gmm_2d(DISTRIBUTIONS, CLASSES, NUM_EXAMPLES)

    # use this for debugging
    colors = ['red', 'green', 'blue']
    plt.scatter(X[:, 0], X[:, 1], c=Y_.flatten(), cmap=ListedColormap(colors))

    one_hot = Y_.reshape(-1)
    one_hot = np.eye(CLASSES)[one_hot]

    # train the model
    model = fcann2_train(X, one_hot)

    print ("Best weights at iteration:", model['best_iter'])

    Y = fcann2_classify(model, X)

    data.plot_decision_boundary(X, lambda x: fcann2_classify(model, x))

    # graph the data points
    data.graph_data(X, Y_, Y)
