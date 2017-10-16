import numpy as np

import data

PARAM_NITER = 100000
PARAM_DELTA = 0.1
SEED = 1510


def sigmoid(z, y=1):
    if y == 1:
        s = np.divide(np.exp(z), (1.0 + np.exp(z)))
    elif y == 0:
        s = np.divide(1, (1.0 + np.exp(z)))
    return np.array(s)


def binlogreg_train(X, Y_):
    """
        Optimizer function for BinLogReg params w and b
    :param X: np.array NxD
    :param Y_: np.array Nx1, correct labels
    :return: weights w, bias b
    """
    N = X.shape[0]

    w = np.random.randn(X.shape[1], 1)  # D x 1
    b = np.random.randn(N, 1)  # N x 1

    for i in range(PARAM_NITER+1):
        # klasifikacijski rezultati
        scores = np.dot(X, w) + b  # N x 1

        # vjerojatnosti razreda c_1
        probs = sigmoid(scores, y=1)  # N x 1

        # gubitak
        loss = -1 * float(np.dot(Y_.T, np.log(probs)))  # scalar

        # dijagnostički ispis
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # if i % 1000 == 0:
        #     Y = np.around(probs, decimals=0)
        #     decfun = binlogreg_decfun(w, b)
        #     bbox = (np.min(X, axis=0), np.max(X, axis=0))
        #     data.graph_surface(decfun, bbox, offset=0.5)
        #     data.graph_data(X, Y_, Y)

        # derivacije gubitka po klasifikacijskom rezultatu
        dL_dscores = np.subtract(probs, Y_)  # N x 1

        # gradijenti parametara
        grad_w = np.divide(np.dot(X.T, dL_dscores), N)  # D x 1
        grad_b = np.divide(np.sum(dL_dscores), N)  # 1 x 1

        # poboljšani parametri
        w += -PARAM_DELTA * grad_w
        b += -PARAM_DELTA * grad_b

    return w, b


def binlogreg_decfun(w, b):
    def classify(X):
      return binlogreg_classify(X, w, b)
    return classify


def binlogreg_classify(X, w, b):
    """

    :param X: np.array NxD, dataset
    :param w: weights of LogReg
    :param b: bias of LogReg
    :return: probs of class c1
    """
    scores = np.dot(X, w) + b  # N x 1

    # vjerojatnosti razreda c_1
    probs = sigmoid(scores, y=1)  # N x 1

    return probs


if __name__ == "__main__":
    np.random.seed(SEED)

    # get the training dataset
    X, Y_ = data.sample_gauss_2d(2, 100)

    # train the model
    w, b = binlogreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w, b)
    Y = np.around(probs, decimals=0)

    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[np.squeeze(probs).argsort()])
    print(accuracy, recall, precision, AP)

    # graph the decision surface
    decfun = binlogreg_decfun(w, b)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    # graph the data points
    data.graph_data_2d(X, Y_, Y)
