
import tensorflow as tf
from tensorflow import Tensor
import numpy as np
import functools

def define_scope(name_scope=False):
    def _(function):
        attribute = '_cache_' + function.__name__

        @property
        @functools.wraps(function)
        def decorator(self):
            if not hasattr(self, attribute):
                if(name_scope):
                    with tf.name_scope(function.__name__):
                        setattr(self, attribute, function(self))
                else:
                    with tf.variable_scope(function.__name__):
                        setattr(self, attribute, function(self))
            return getattr(self, attribute)

        return decorator
    return _


def perceptron_layer(x: Tensor, out_dim, batch_norm=False, activation_fn=tf.sigmoid, name="hidden_layer"):
    in_dim = x.get_shape().as_list()[1]
    with tf.variable_scope(name) as v_scope:
        w = tf.Variable(initial_value=tf.random_normal((in_dim, out_dim)))
        b = tf.Variable(initial_value=tf.random_normal((out_dim,)))
        prod = tf.einsum('ij,jk->ik', x, w)
        if(batch_norm):
            prod, b_scope = batch_normalization(prod, out_dim, b)
        else:
            prod = prod + b

        total_v_coll = v_scope.global_variables()
        if(b_scope):
            total_v_coll.append(b_scope.global_variables())

        if(activation_fn != None):
            return activation_fn(prod), total_v_coll
        else:
            return prod, total_v_coll


def apply_dropout(x, keep_prob):
    def _():
        p = tf.random_normal([tf.shape(x)[0]])
        return tf.where(tf.less_equal(p, keep_prob), x, tf.zeros_like(x))
    return _


def dropout(x: Tensor, is_training, keep_prob=0.5):
    with tf.variable_scope("dropout"):
        return tf.cond(tf.equal(is_training, True), true_fn=apply_dropout(x, keep_prob), false_fn=lambda: x)


def batch_normalization(x: Tensor, out_dim, bias):
    with tf.variable_scope('batch_norm') as scope:
        mean, var = tf.nn.moments(x, axes=0)
        v = tf.Variable(tf.ones([out_dim]))
        x = tf.div_no_nan(x - mean, tf.sqrt(var + 1e-3))
        return v * x + bias, scope


class MultiLayerPerceptron:
    def __init__(self, layer_structure: list, data: Tensor, targets: Tensor, is_training: Tensor, learning_rate=0.01, use_tf_model=True, step=None):
        if(use_tf_model):
            self.model, self._v_coll = self.inference_tf(data, is_training, layer_structure)
        else:
            self.model, self._v_coll = self.inference(data, is_training, layer_structure)

        self.step = step
        self.data = data
        self.targets = targets
        self.learning_rate = learning_rate
        self.accuracy
        self.optimize
        self.loss
        self.summary

    @define_scope(True)
    def summary(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("accuracy", self.accuracy)
            tf.summary.histogram("histogram loss", self.loss)
            return tf.summary.merge_all()

    def get_vars_to_save(self):
        return self._v_coll

    def inference_tf(self, x, is_training, layer_structure):
        with tf.variable_scope('network') as n_scope:
            data = x

            for l_index in range(0, len(layer_structure) - 1):
                l_structure = layer_structure[l_index]
                data = tf.contrib.layers.fully_connected(
                    data, l_structure, activation_fn=None)
                data = tf.layers.batch_normalization(
                    data, training=is_training)
                data = tf.nn.relu(data)
                data = tf.contrib.layers.dropout(
                    data, keep_prob=0.8, is_training=is_training)

            y = tf.contrib.layers.fully_connected(
                data, layer_structure[-1], activation_fn=None)
            y = tf.layers.batch_normalization(y, training=is_training)

            total_v_coll = n_scope.global_variables()

            return tf.nn.softmax(y), total_v_coll

    def inference(self, x, is_training, layer_structure):
        with tf.variable_scope('network') as n_scope:
            data = x

            total_v_coll = n_scope.global_variables()

            for l_index in range(0, len(layer_structure) - 1):
                l_structure = layer_structure[l_index]
                data, l_scope = perceptron_layer(
                    data, l_structure, batch_norm=True)
                data = dropout(data, is_training, keep_prob=0.4)
                if(l_scope):
                    total_v_coll.append(l_scope)

            y, l_scope = perceptron_layer(
                data, layer_structure[-1], batch_norm=True, activation_fn=tf.nn.softmax, name="out_layer")

            if(l_scope):
                total_v_coll.append(l_scope)

            return y, total_v_coll

    @define_scope()
    def accuracy(self):
        pred_index = tf.argmax(self.model, axis=1)
        targ_index = tf.argmax(self.targets, axis=1)
        return tf.reduce_mean(
            tf.cast(tf.equal(pred_index, targ_index), dtype=tf.float32))

    @define_scope()
    def loss(self):
        return -tf.reduce_sum(tf.reduce_mean(self.targets * tf.log(self.model), axis=1))

    @define_scope()
    def optimize(self):
        # UPDATE_OPS necessary to use batch normalization
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            # Define train step using the AdamOptimizer with global_step increment
            return tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss, global_step=self.step)
