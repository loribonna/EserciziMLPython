import numpy as np
from numpy import ndarray

eps = np.finfo(float).eps

def sigmoid(x):
    """
    Element-wise sigmoid function

    Parameters
    ----------
    x: np.array
        a numpy array of any shape

    Returns
    -------
    np.array
        an array having the same shape of x.
    """

    return np.exp(x) / (1 + np.exp(x))


def loss(y_true: ndarray, y_pred: ndarray):
    """
    The binary crossentropy loss.

    Parameters
    ----------
    y_true: np.array
        real labels in {0, 1}. shape=(n_examples,)
    y_pred: np.array
        predicted labels in [0, 1]. shape=(n_examples,)

    Returns
    -------
    float
        the value of the binary crossentropy.
    """

    n_el = y_true.size
    side_1 = - y_true * np.log(y_pred)
    side_2 = (1 - y_true) * np.log(1 - y_pred)
    val = (side_1 - side_2) / n_el
    return val


def dloss_dw(y_true: ndarray, y_pred: ndarray, X: ndarray):
    """
    Derivative of loss function w.r.t. weights.

    Parameters
    ----------
    y_true: np.array
        real labels in {0, 1}. shape=(n_examples,)
    y_pred: np.array
        predicted labels in [0, 1]. shape=(n_examples,)
    X: np.array
        predicted data. shape=(n_examples, n_features)

    Returns
    -------
    np.array
        derivative of loss function w.r.t weights.
        Has shape=(n_features,)
    """

    n_el = y_true.shape
    return - X.transpose().dot(y_pred - y_true) / n_el


class LogisticRegression:
    """ Models a logistic regression classifier. """

    def __init__(self):
        """ Constructor method """

        # weights placeholder
        self._w = None

    def fit_gd(self, X: ndarray, Y: ndarray, n_epochs, learning_rate, verbose=False):
        """
        Implements the gradient descent training procedure.

        Parameters
        ----------
        X: np.array
            data. shape=(n_examples, n_features)
        Y: np.array
            labels. shape=(n_examples,)
        n_epochs: int
            number of gradient updates.
        learning_rate: float
            step towards the descent.
        verbose: bool
            whether or not to print the value of cost function.
        """
        n_examples, n_features = X.shape
        self._w = np.random.rand(n_features)

        for e in range(0, n_epochs):
            product = X.dot(self._w)
            predict = sigmoid(product)

            # cost = loss(Y, predict)

            gradient = dloss_dw(Y, predict, X)
            self._w -= learning_rate * gradient

    def predict(self, X: ndarray):
        """
        Function that predicts.

        Parameters
        ----------
        X: np.array
            data to be predicted. shape=(n_test_examples, n_features)

        Returns
        -------
        prediction: np.array
            prediction in {0, 1}.
            Shape is (n_test_examples,)
        """

        product = X.dot(self._w)
        predict = sigmoid(product)

        return predict
