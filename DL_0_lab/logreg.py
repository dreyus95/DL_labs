import numpy as np
import data
from logreg_scores import accuracy, precision, recall, f1
from time import sleep

PARAM_NITER = 20000
PARAM_DELTA = 0.15
SEED = 1510


def print_logreg_stats(y_test, prediction):
    print("\nAcc:")
    print("Micro:", accuracy(y_test, prediction, "micro"))
    print("Macro:", accuracy(y_test, prediction, "macro"))
    print("\nPrecision:")
    print("Micro:", precision(y_test, prediction, "micro"))
    print("Macro:", precision(y_test, prediction, "macro"))
    print("\nRecall:")
    print("Micro:", recall(y_test, prediction, "micro"))
    print("Macro:", recall(y_test, prediction, "macro"))
    print("\nF1:")
    print("Micro:", f1(y_test, prediction, "micro"))
    print("Macro:", f1(y_test, prediction, "macro"))


def softmax(z, sum):
    res = np.divide(z, sum)
    return res


def logreg_decfun(w, b):
    def classify(X):
      return np.argmax(logreg_classify(X, w, b), axis=1)
    return classify


def logreg_classify(X, w, b):
    """

    :param X: np.array NxD, dataset
    :param w: weights of LogReg
    :param b: bias of LogReg
    :return: probs of classes
    """
    scores = np.dot(X, w) + b  # N x C
    expscores = np.exp(scores)  # N x C

    # nazivnik sofmaksa
    sumexp = np.sum(expscores, axis=1, keepdims=True)  # N x 1

    # logaritmirane vjerojatnosti razreda
    probs = softmax(expscores, sumexp)  # N x C

    return probs


def log_loss(Y_pred, Y_true, N):
    ones = np.ones(shape=(Y_true.shape[0], Y_true.shape[1]))

    logprobs = np.log(Y_pred)  # N x C
    second_loss_arg = np.log(np.add(ones - Y_pred, 1e-7))  # N x C, adding 1e-7 so we don't have log(0)

    loss1 = []
    for a, b in zip(logprobs, Y_true):
        loss1.append(np.dot(a, b.T))

    loss2 = []
    for a, b in zip(second_loss_arg, ones - Y_true):
        loss2.append(np.dot(a, b.T))

    loss = np.add(loss1, loss2)

    # gubitak
    loss = -1 * loss
    loss = np.sum(loss) / N  # scalar

    return loss


def logreg_train(X, Y_, eps=1e-7):
    """
            Optimizer function for MultinomialLogReg params W and b
        :param X: np.array NxD
        :param Y_: np.array NxC, correct labels
        :return: weights w, bias b
        """
    W = np.random.randn(X.shape[1], Y_.shape[1])  # D x C
    b = np.zeros(shape=[1, Y_.shape[1]])  # 1 x C

    N = X.shape[0]

    prev_loss = 9999

    for i in range(PARAM_NITER+1):
        # eksponencirani klasifikacijski rezultati
        scores = np.dot(X, W) + b  # N x C
        expscores = np.exp(scores)  # N x C

        # nazivnik sofmaksa
        sumexp = np.sum(expscores, axis=1, keepdims=True)  # N x 1

        # logaritmirane vjerojatnosti razreda
        probs = softmax(expscores, sumexp)  # N x C

        loss = log_loss(probs, Y_, N)

        # stop iterating
        if np.abs((prev_loss - loss)) < eps:
            print("iteration {}: loss {}".format(i, loss))
            return W, b, i
        else:
            prev_loss = loss

        # dijagnostički ispis
        if i % 1000 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivacije komponenata gubitka po rezultatu
        dL_ds = np.subtract(probs, Y_)  # N x C

        # gradijenti parametara
        grad_W = np.divide(np.dot(X.T, dL_ds), N)  # D x C
        grad_b = np.divide(np.sum(dL_ds, axis=0), N)  # 1 x C

        # poboljšani parametri
        W += -PARAM_DELTA * grad_W
        b += -PARAM_DELTA * grad_b

    return W, b, i


if __name__ == "__main__":
    np.random.seed(SEED)

    # get the training dataset
    X, Y_ = data.sample_gauss_2d(3, 100)

    classes = int(max(Y_) + 1)

    one_hot = Y_.reshape(-1)
    one_hot = np.eye(classes)[one_hot]

    # train the model
    w, b, iteration = logreg_train(X, one_hot)

    print("Training stopped at iteration:", iteration, "(at", 100*iteration/PARAM_NITER, "%)")

    probs = logreg_classify(X, w, b)
    Y = np.argmax(probs, axis=1)

    sleep(2)
    print_logreg_stats(Y_, Y)

    # graph the decision surface
    # decfun = logreg_decfun(w, b)
    # bbox = (np.min(X, axis=0), np.max(X, axis=0))
    # data.graph_surface(decfun, bbox, offset=0.0)

    # graph the data points
    data.graph_data(X, Y_, Y)