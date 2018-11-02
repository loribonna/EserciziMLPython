import numpy as np
from numpy import ndarray
from abc import abstractmethod

class BaseClassifier:
    _w: ndarray = None
    _error = 1
    _norm_factor = 0.975

    def __init__(self, w: ndarray, norm_factor):
        self._w = w
        self._norm_factor = norm_factor

    def get_error(self):
        return self._error

    def _get_err(self, Y_test, Y_pred):
        return np.sum(self._w * (Y_pred != Y_test))

    def get_reputation(self):
        return 0.5 * np.log((1 - self._error) / self._error)

    def _get_updated_w(self, Y_test: ndarray, Y_pred: ndarray):
        w = np.zeros(self._w.size)
        for i in range(0, self._w.size):
            rep = self.get_reputation()
            exponent = rep if Y_pred[i] == Y_test[i] else -rep
            w[i] = (self._w[i] / self._norm_factor) * np.exp(exponent)

        return w

    def fit(self, X: ndarray, Y: ndarray) -> ndarray: pass

    def predict(self, X: ndarray, f_index=None, f_value=None) -> ndarray: pass
