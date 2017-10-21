import tensorflow as tf
from sklearn.svm import SVC as SVM
import numpy as np
import data
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


NUM_EXAMPLES = 10
CLASSES = 2
DISTRIBUTIONS = 6
SEED = 106


class KSVMWrap(object):
    """
        Wrapper class for SVM.
    """
    def __init__(self, X, Y_, param_svm_c=1.0, kernel='rbf', param_svm_gamma='auto', decision_function_shape='ovo'):
        self.svm = SVM(C=param_svm_c, kernel=kernel, gamma=param_svm_gamma,
                       decision_function_shape=decision_function_shape)
        self.svm.fit(X, Y_)

    def predict(self, X):
        return self.svm.predict(X)

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

    def support(self):
        # return indices
        return self.svm.support_


if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(SEED)
    tf.set_random_seed(SEED)

    # instanciraj podatke X i labele Yoh_
    X, Y_ = data.sample_gmm_2d(DISTRIBUTIONS, CLASSES, NUM_EXAMPLES)

    # use this for debugging
    # colors = ['red', 'green', 'blue']
    # plt.scatter(X[:, 0], X[:, 1], c=Y_.flatten(), cmap=ListedColormap(colors))

    # izgradi graf:
    ksmwrap = KSVMWrap(X, Y_)

    # nauči parametre:
    preds = ksmwrap.predict(X)

    ksmwrap.eval_perf(preds, Y_.flatten())

    data.plot_decision_boundary(X, lambda x: ksmwrap.predict(x))
    # graph the data points
    data.graph_data(X, Y_, preds, special=ksmwrap.support())
