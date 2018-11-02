import numpy as np
import random
from numpy.linalg import norm
from numpy import ndarray
eps = np.finfo(float).eps


def _def_kernel(w: ndarray, x: ndarray):
    return w.dot(x)

def get_gauss_ker(sigm):
    def gauss_ker(w, x):
        return np.exp(-(np.linalg.norm(w-x)**2)/(2*(sigm**2)))

def get_polyn_ker(c, d):
    def polyn_kernel(w, x):
        return np.power(_def_kernel(w,x)+c, d)

class SVM:
    """ Models a Support Vector machine classifier based on the PEGASOS algorithm. """

    def __init__(self, n_epochs, lambDa, use_bias=True, kernel=None):
        """ Constructor method """

        # weights placeholder
        self._w = None
        self._original_labels = None
        self._n_epochs = n_epochs
        self._lambda = lambDa
        self._use_bias = use_bias
        self._kernel = _def_kernel if kernel == None else kernel


    def map_y_to_minus_one_plus_one(self, y):
        """
        Map binary class labels y to -1 and 1
        """
        ynew = np.array(y)
        self._original_labels = np.unique(ynew)
        assert len(self._original_labels) == 2
        ynew[ynew == self._original_labels[0]] = -1.0
        ynew[ynew == self._original_labels[1]] = 1.0
        return ynew

    def map_y_to_original_values(self, y):
        """
        Map binary class labels, in terms of -1 and 1, to the original label set.
        """
        ynew = np.array(y)
        ynew[ynew == -1.0] = self._original_labels[0]
        ynew[ynew == 1.0] = self._original_labels[1]
        return ynew

    def loss(self, y_true: ndarray, y_pred: ndarray):
        """
        The PEGASOS loss term

        Parameters
        ----------
        y_true: np.array
            real labels in {0, 1}. shape=(n_examples,)
        y_pred: np.array
            predicted labels in [0, 1]. shape=(n_examples,)

        Returns
        -------
        float
            the value of the pegasos loss.
        """
        """
        ###########################
        Write here the PEGASOS loss.
        ###########################
        """

        err = [np.max([0, 1-y_true[i]*y_pred[i]]) for i in range(0, y_true.size)]
        return np.sum(err) / y_true.size
        # return np.random.normal(loc=100.0, scale=5.0, size=(1,))[0]

    def fit_gd(self, X, Y, verbose=False):
        """
        Implements the gradient descent training procedure.

        Parameters
        ----------
        X: np.array
            data. shape=(n_examples, n_features)
        Y: np.array
            labels. shape=(n_examples,)
        verbose: bool
            whether or not to print the value of cost function.
        """

        if self._use_bias:
            X = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=-1)

        n_samples, n_features = X.shape
        Y = self.map_y_to_minus_one_plus_one(Y)

        # initialize weights
        self._w = np.zeros(shape=(n_features,), dtype=X.dtype)

        t = 0
        # loop over epochs
        for e in range(1, self._n_epochs + 1):
            for j in range(n_samples):
                t += 1
                n_t = 1 / (t * self._lambda)
                prod = self._kernel(self._w, X[j])
                rate = 1 - n_t * self._lambda
                if (Y[j] * prod < 1):
                    self._w = rate * self._w + n_t * Y[j] * X[j]
                else:
                    self._w = rate * self._w

            # predict training data
            cur_prediction = np.dot(X, self._w)

            # compute (and print) cost
            cur_loss = self.loss(y_true=Y, y_pred=cur_prediction)
            if verbose:
                print("Epoch {} Loss {}".format(e, cur_loss))

    def predict(self, X):
        if self._use_bias:
            X = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=-1)

        # prediction = np.array([random.choice(self._original_labels) for i in range(X.shape[0])])
        """
        ######################################
        Write here the PEGASOS inference rule.
        ######################################
        """
        prod = X.dot(self._w)
        prediction = np.sign(prod)
        return self.map_y_to_original_values(prediction)
