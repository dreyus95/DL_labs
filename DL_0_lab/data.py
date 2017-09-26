import numpy as np
import matplotlib.pyplot as plt

NUM_EXAMPLES = 100
SEED = 1510


def myDummyDecision(X):
    scores = X[:, 0] + X[:, 1] - 5
    return np.array(scores)


def graph_surface(fun, rect, offset=0.0, width=0, height=0):
    if width == 0 and height == 0:
        width = np.linspace(start=rect[0][0], stop=rect[1][0], num=100)
        height = np.linspace(start=rect[0][1], stop=rect[1][1], num=100)
    ws_x, ws_y = np.meshgrid(width, height)

    cmap = plt.get_cmap('rainbow')
    XX = np.stack((ws_x.flatten(), ws_y.flatten())).transpose()

    Z = np.array([fun(X) for X in XX.reshape([-1, NUM_EXAMPLES, 2])])  # batches x NUM_EXAMPLES x features
    Z = Z.reshape(ws_x.shape)

    z_min, z_max = -np.abs(Z).max(), np.abs(Z).max()

    plt.contour(ws_x, ws_y, Z, colors='black', levels=[offset])

    plt.pcolormesh(ws_x, ws_y, Z, vmin=offset + z_min, vmax=offset + z_max, cmap=cmap)
    return


def graph_data(X, Y_, Y):
    markers = ['s', 'o']

    correct_preds_indices = [i for i, (a, b) in enumerate(zip(Y_, Y)) if a == b]
    wrong_preds_indices = [i for i, (a, b) in enumerate(zip(Y_, Y)) if a != b]

    classes = int(max(Y_) + 1)

    correct_all_indices = []
    wrong_all_indices = []

    for cls in range(classes):
        correct_all_indices.append([a for a in correct_preds_indices if Y_[a] == cls])
        wrong_all_indices.append([a for a in wrong_preds_indices if Y_[a] != cls])

    for cls in range(classes):
        plt.scatter(X[correct_all_indices[cls]][:, 0], X[correct_all_indices[cls]][:, 1], edgecolor='black',
                    marker=markers[1], label="correct {0}".format(cls))
        plt.scatter(X[wrong_all_indices[cls]][:, 0], X[wrong_all_indices[cls]][:, 1], edgecolor='black',
                    marker=markers[0], label="wrong {0}".format(cls))

    plt.legend(loc="upper left")

    # show the results
    plt.show()


def graph_data_2d(X, Y_, Y):
    colors = ['gray', 'white']
    markers = ['s', 'o']

    correct_preds_indices = [i for i, (a, b) in enumerate(zip(Y_, Y)) if a == b]
    wrong_preds_indices = [i for i, (a, b) in enumerate(zip(Y_, Y)) if a != b]

    correct_ones_indices = [a for a in correct_preds_indices if Y_[a] == 1]
    correct_zeroes_indices = [a for a in correct_preds_indices if Y_[a] == 0]

    wrong_ones_indices = [a for a in wrong_preds_indices if Y_[a] == 1]
    wrong_zeroes_indices = [a for a in wrong_preds_indices if Y_[a] == 0]

    plt.scatter(X[correct_ones_indices][:, 0], X[correct_ones_indices][:, 1], c=colors[1], edgecolor='black',
                marker=markers[1], label="correct ones")
    plt.scatter(X[wrong_ones_indices][:, 0], X[wrong_ones_indices][:, 1], c=colors[1], edgecolor='black',
                marker=markers[0], label="wrong ones")
    plt.scatter(X[correct_zeroes_indices][:, 0], X[correct_zeroes_indices][:, 1], c=colors[0], edgecolor='black',
                marker=markers[1], label="correct zeroes")
    plt.scatter(X[wrong_zeroes_indices][:, 0], X[wrong_zeroes_indices][:, 1], c=colors[0], edgecolor='black',
                marker=markers[0], label="wrong zeroes")

    plt.legend(loc="upper left")

    # show the results
    plt.show()


def eval_perf_binary(Y, Y_):
    TP = TN = FP = FN = 0
    for i in range(len(Y)):
        if Y[i] == 1 and Y_[i] == 1:
            TP += 1
        elif Y[i] == 1 and Y_[i] == 0:
            FP += 1
        elif Y[i] == 0 and Y_[i] == 1:
            FN += 1
        elif Y[i] == 0 and Y_[i] == 0:
            TN += 1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    return accuracy, recall, precision


def eval_AP(Y):
    num_predictions = len(Y)
    total_sum = 0.0
    further = [1]
    for i in range(num_predictions):
        if Y[i] == 0:
            continue
        past = Y[0:i]
        further = Y[i:len(Y)]
        past_ones = len([1 for x in past if x == 1])
        further_zeroes = len([0 for x in further if x == 0])
        precision_i = (num_predictions - past_ones - further_zeroes) / num_predictions
        total_sum += precision_i * Y[i]

    return float(total_sum / np.sum(Y))


def sample_gauss_2d(C, N):
    distributions = [Random2DGaussian() for _ in range(C)]
    datasets = [distribution.get_sample(N) for distribution in distributions]
    X = np.concatenate(datasets)
    np.random.shuffle(X)
    X = X[0:N]
    Y = []
    for x in X:
        for index, dataset in enumerate(datasets):
            if x in dataset:
                Y.append(index)
                break

    return np.array(X), np.reshape(np.array(Y), [N, -1])


class Random2DGaussian(object):
    def __init__(self):
        self.min_x = 0
        self.min_y = 0
        self.max_x = 10
        self.max_y = 10

        self.mean_x = (self.max_x - self.min_x) * np.random.random_sample() + self.min_x
        self.mean_y = (self.max_y - self.min_y) * np.random.random_sample() + self.min_y

        self.mean = [self.mean_x, self.mean_y]

        self.eigval_x = (np.random.random_sample() * (self.max_x - self.min_x) / 5) ** 2
        self.eigval_y = (np.random.random_sample() * (self.max_y - self.min_y) / 5) ** 2
        self.D = np.matrix([[self.eigval_x, 0], [0, self.eigval_y]])

        self.phi = 360 * np.random.random_sample()
        self.R = np.matrix([[np.cos(self.phi), -1 * np.sin(self.phi)], [np.sin(self.phi), np.cos(self.phi)]])

        self.cov = np.matmul(np.matmul(self.R.T, self.D), self.R)

    def get_sample(self, n):
        return np.random.multivariate_normal(self.mean, self.cov, size=n)


if __name__ == "__main__":
    np.random.seed(SEED)

    # get the training dataset
    X, Y_ = sample_gauss_2d(2, NUM_EXAMPLES)

    # get the class predictions
    Y = np.array([0 if score < 0.5 else 1 for score in myDummyDecision(X)])

    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(myDummyDecision, bbox, offset=0.0)

    # graph the data points
    graph_data_2d(X, Y_, Y)
    plt.plot()
