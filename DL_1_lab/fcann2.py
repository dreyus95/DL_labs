import numpy as np
import data

HIDDEN_LAYER_DIM = 5
SEED = 100
NUM_EXAMPLES = 10
CLASSES = 2
DISTRIBUTIONS = 6
REG_LAMBDA = 1e-3


def relu(input):
    output = input * (input > 0)
    return np.array(output)


def softmax(z, sum):
    res = np.divide(z, sum)
    return res


def cross_entropy_softmax_loss_array(softmax_probs_array, y_onehot):
    indices = np.argmax(y_onehot, axis = 1).astype(int)
    predicted_probability = softmax_probs_array[np.arange(len(softmax_probs_array)), indices]
    log_preds = np.log(predicted_probability)
    loss = -1.0 * np.sum(log_preds) / len(log_preds)
    return loss


def fcann2_train(X, Y_, param_niter=1e5, param_delta=0.07):
    D = X.shape[1]
    C = Y_.shape[1]
    N = X.shape[0]

    W1 = np.random.randn(D, HIDDEN_LAYER_DIM)
    b1 = np.zeros(shape=[1, HIDDEN_LAYER_DIM])

    W2 = np.random.randn(HIDDEN_LAYER_DIM, C)
    b2 = np.zeros(shape=[1, C])

    model = {}

    prev_loss = 9999

    for i in range(int(param_niter)+1):

        ###### HIDDEN LAYER PASS ######
        input = np.dot(X, W1) + b1
        hidden = relu(input)
        output = np.dot(hidden, W2) + b2

        ##### SOFTMAX PART #####
        exp_scores = np.exp(output)
        sumexp = np.sum(exp_scores, axis=1, keepdims=True)
        probs = softmax(exp_scores, sumexp)

        ##### UPDATES WEIGHTS #####
        output_error_signal = (probs - Y_) / N

        error_signal_hidden = np.dot(output_error_signal, W2.T)
        error_signal_hidden[hidden <= 0] = 0  # reLU iz max(0, x), derivative is 0 if values are < 0

        dW2 = np.dot(hidden.T, output_error_signal)
        db2 = np.sum(output_error_signal, axis=0, keepdims=True)

        dW1 = np.dot(X.T, error_signal_hidden)
        db1 = np.sum(error_signal_hidden, axis=0, keepdims=True)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += REG_LAMBDA * W2
        dW1 += REG_LAMBDA * W1

        # Gradient descent parameter update
        W1 += -param_delta * dW1
        b1 += -param_delta * db1
        W2 += -param_delta * dW2
        b2 += -param_delta * db2

        loss = cross_entropy_softmax_loss_array(probs, Y_)

        if prev_loss > loss:
            # Assign new parameters to the model
            model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'best_iter': i}
            prev_loss = loss

        # dijagnostički ispis
        if i % 1000 == 0 and i != 0:
            print("iteration {}: loss {}".format(i, loss))

    return model


def fcann2_classify(model, X):
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

    classes = int(max(Y_) + 1)

    one_hot = Y_.reshape(-1)
    one_hot = np.eye(classes)[one_hot]

    # train the model
    model = fcann2_train(X, one_hot)

    print ("Best weights at iteration:", model['best_iter'])

    Y = fcann2_classify(model, X)

    data.plot_decision_boundary(X, lambda x: fcann2_classify(model, x))

    # graph the data points
    data.graph_data(X, Y_, Y)