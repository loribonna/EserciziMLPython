import numpy as np
from numpy import ndarray
from base_ada_classifier import BaseClassifier

class RandomClassifier(BaseClassifier):
    _feature_index: int = None
    _feature_value: float = None
    _max_cycle = 1000

    def __init__(self, w: ndarray, norm_factor = 1):
        super(RandomClassifier, self).__init__(w, norm_factor)

    def fit(self, X: ndarray, Y: ndarray) -> ndarray:
        n_el, n_features = X.shape
        err = 1
        i = 0
        f_index = np.random.randint(0, n_features)
        v_min = np.min(X[:, f_index])
        v_max = np.max(X[:, f_index])
        step_size = n_features
        step = (v_max - v_min) / step_size

        #  and i < self._max_cycle
        while (err > 0.5):
            i += 1

            if step == 0:
                f_value = v_min
            else:
                rng = np.arange(v_min, v_max, step)
                f_value = np.random.choice(rng)

            prediction = self.predict(X, f_index, f_value)
            err = self._get_err(Y, prediction)

            if (i % (n_el) == 0 or step == 0):
                f_index = np.random.randint(0, n_features)
                v_min = np.min(X[:, f_index])
                v_max = np.max(X[:, f_index])
                step = (v_max - v_min) / step_size



        if (i == self._max_cycle):
            print("Cycles Error")
            assert (i != self._max_cycle)

        self._error = err
        self._feature_index = f_index
        self._feature_value = f_value

        return self._get_updated_w(Y, prediction)

    def predict(self, X: ndarray, f_index=None, f_value=None) -> ndarray:
        f_index = self._feature_index if f_index == None else f_index
        f_value = self._feature_value if f_value == None else f_value

        pred = X[:, f_index] > f_value

        classes = np.ones(pred.size)

        # classes[pred == True] = 1
        classes[pred == False] = -1
        return classes
