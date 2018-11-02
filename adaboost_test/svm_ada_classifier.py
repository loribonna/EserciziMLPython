import numpy as np
import random
from numpy.linalg import norm
from numpy import ndarray
from base_ada_classifier import BaseClassifier
eps = np.finfo(float).eps


def _def_kernel(w: ndarray, x: ndarray):
    return w.dot(x)

def get_gauss_ker(sigm):
    def gauss_ker(w, x):
        return np.exp(-(np.linalg.norm(w-x)**2)/(2*(sigm**2)))

def get_polyn_ker(c, d):
    def polyn_kernel(w, x):
        return np.power(_def_kernel(w,x)+c, d)

class SVM(BaseClassifier):
    """ Models a Support Vector machine classifier based on the PEGASOS algorithm. """
    _w_class: ndarray = None

    def __init__(self, w: ndarray, n_epochs = 100, lambDa = 0.01, kernel=None, norm_factor=0.45):
        """ Constructor method """
        super(SVM, self).__init__(w, norm_factor)

        self._n_epochs = n_epochs
        self._lambda = lambDa
        self._kernel = _def_kernel if kernel == None else kernel

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

        err = [np.max([0, 1 - y_true[i] * y_pred[i]]) for i in range(0, y_true.size)]
        return np.sum(err) / y_true.size
        # return np.random.normal(loc=100.0, scale=5.0, size=(1,))[0]

    def fit(self, X, Y):
        """
        Implements the gradient descent training procedure.

        Parameters
        ----------
        X: np.array
            data. shape=(n_examples, n_features)
        Y: np.array
            labels. shape=(n_examples,)

        Returns
        -------
        w: np.array
            updated weights
        """

        n_samples, n_features = X.shape

        # initialize weights
        self._w_class = np.zeros(shape=(n_features,), dtype=X.dtype)

        t = 0
        # loop over epochs
        for e in range(1, self._n_epochs + 1):
            for j in range(n_samples):
                t += 1
                n_t = 1 / (t * self._lambda)
                prod = self._kernel(self._w_class, self._w[j] * X[j])
                rate = 1 - n_t * self._lambda
                if (Y[j] * prod < 1):
                    self._w_class = rate * self._w_class + n_t * Y[j] * X[j]
                else:
                    self._w_class = rate * self._w_class

        # compute (and print) cost
        prediction = self.predict(X)
        # cur_loss = self.loss(y_true=Y, y_pred=np.dot(X, self._w_class))
        err = self._get_err(Y, prediction)

        self._error = err
        return self._get_updated_w(Y, prediction)

    def predict(self, X):
        prod = X.dot(self._w_class)
        prediction = np.sign(prod)
        return prediction
