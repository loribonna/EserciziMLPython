import numpy as np
import random
from numpy.linalg import norm
from numpy import ndarray
from base_ada_classifier import BaseClassifier
from random_ada_classifier import RandomClassifier

eps = np.finfo(float).eps


class AdaBoost:
    """ Models a Support Vector machine classifier based on the PEGASOS algorithm. """
    _classifiers: list = []
    _n_classifiers: int = None
    _use_bias: bool = True
    _original_labels: ndarray = None

    def __init__(self, n_classifiers, Base_Classifier = RandomClassifier, use_bias=True):
        """ Constructor method """

        self._n_classifiers = n_classifiers
        self._use_bias = use_bias
        self._Base_Classifier = Base_Classifier

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
        w = np.repeat(1 / n_samples, n_samples)

        # loop over epochs
        for e in range(1, self._n_classifiers + 1):
            cl = self._Base_Classifier(w, norm_factor=0.975)
            w = cl.fit(X, Y)

            self._classifiers.append(cl)

            if verbose:
                print("Classifier {} Accuracy {} Reputation {}".format(e, 1 - cl.get_error(), cl.get_reputation()))

    def predict(self, X):
        if self._use_bias:
            X = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=-1)

        n_samples, n_features = X.shape
        combination: ndarray = np.zeros(n_samples)

        for cl_i in range(0, self._n_classifiers):
            cl: BaseClassifier = self._classifiers[cl_i]
            combination += cl.get_reputation() * cl.predict(X)

        prediction = np.sign(combination)
        return self.map_y_to_original_values(prediction)
