import functools
import tensorflow as tf
import numpy as np
from tensorflow import Tensor
from tensorflow.data import Iterator
from google_drive_downloader import GoogleDriveDownloader as gdd


def define_scope(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name__):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


def sigmoid(x):
    """
      Code for the sigmoid function. https://en.wikipedia.org/wiki/Logistic_function
    """
    calc = tf.div(1., (1. + tf.exp(-x)))
    return calc


class LogisticRegressionModel:

    def __init__(self, data: Tensor, target: Tensor, n_features: int, learning_rate=1e-1):
        self.data = tf.cast(data, tf.float32)
        self.target = tf.cast(target, tf.float32)

        self.n_features = n_features
        self._optimizer=tf.train.GradientDescentOptimizer(learning_rate)

        self.prediction_round
        self.prediction
        self.optimize
        self.accuracy
        self.loss

    @define_scope
    def prediction(self):
        weight = tf.get_variable(
            "weight",
            initializer=tf.random_normal((self.n_features, ))
        )
        bias = tf.get_variable(
            "bias",
            initializer=tf.zeros((1))
        )

        incoming = tf.einsum('ij,i->j', tf.transpose(self.data), weight) + bias
        #incoming = tf.matmul(self.data, weight) + bias
        #incoming = tf.squeeze(incoming)

        return sigmoid(incoming)

    @define_scope
    def loss(self):
        side_1 = self.target * tf.log(self.prediction)
        side_2 = (1 - self.target) * tf.log(1 - self.prediction)

        return - tf.reduce_mean(side_1 + side_2)

    @define_scope
    def optimize(self):
        return self._optimizer.minimize(self.loss)

    @define_scope
    def prediction_round(self):
        return tf.round(self.prediction)

    @define_scope
    def accuracy(self):
        return tf.reduce_mean(
            tf.cast(
                tf.equal(self.target, self.prediction_round),
                tf.float32)
        )
