import numpy as np
import matplotlib.pyplot as plt

SEED = 103
NUM_EXAMPLES = 30
CLASSES = 2
DISTRIBUTIONS = 4


def myDummyDecision(X):
    scores = X[:, 0] + X[:, 1] - 5
    return np.array(scores)


def plot_decision_boundary(X, pred_func, offset=0.0):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    cmap = plt.get_cmap('rainbow')

    z_min, z_max = -np.abs(Z).max(), np.abs(Z).max()
    plt.contour(xx, yy, Z, colors='black', levels=[offset])
    plt.pcolormesh(xx, yy, Z, vmin=offset + z_min, vmax=offset + z_max, cmap=cmap)
    return


def graph_data(X, Y_, Y):
    markers = ['s', 'o']

    correct_preds_indices = [i for i, (a, b) in enumerate(zip(Y_, Y)) if a == b]
    wrong_preds_indices = [i for i, (a, b) in enumerate(zip(Y_, Y)) if a != b]

    classes = max(int(max(Y_) + 1), int(max(Y) + 1))

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


def sample_gmm_2d(K, C, N):
    distributions = [Random2DGaussian() for _ in range(K)]
    datasets = [distribution.get_sample(N) for distribution in distributions]

    Y = []
    for i, dataset in enumerate(datasets):
        # get a random class for each distribution sample
        y = np.random.randint(0, high=C)
        Y += [y for _ in range(len(dataset))]
    X = np.concatenate(datasets)
    return np.array(X), np.reshape(np.array(Y), [K*N, -1])


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
    X, Y_ = sample_gmm_2d(DISTRIBUTIONS, CLASSES, NUM_EXAMPLES)

    # get the class predictions
    Y = np.array([0 if score < 0.5 else 1 for score in myDummyDecision(X)])

    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    plot_decision_boundary(X, lambda x: myDummyDecision(x))

    # graph the data points
    graph_data(X, Y_, Y)
    plt.plot()