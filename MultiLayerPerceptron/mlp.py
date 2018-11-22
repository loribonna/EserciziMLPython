
# %%
import tensorflow as tf
from tensorflow import Tensor
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python import debug as tf_debug

mnist = input_data.read_data_sets('/tmp/mnist', one_hot=True)

# %% [markdown]
# # MNIST Dataset Visualization

# %%
num_row, num_col = 1, 10
f, subplots = plt.subplots(num_row, num_col, sharex='col', sharey='row')

X, y = mnist.train.images, mnist.train.labels
X = np.reshape(X, (-1, 28, 28))

for i in range(num_col):
    X_img = X[np.argmax(y, axis=1) == i].reshape((-1, 28, 28))
    idx = np.random.choice(np.arange(0, X_img.shape[0]))
    subplots[i].imshow(X_img[idx], cmap='gray',
                       interpolation='nearest', aspect='auto')
    title = 'Digit {}'.format(i)
    subplots[i].set_title(title, fontweight="bold")
    subplots[i].grid(b=False)
    subplots[i].axis('off')

f.set_size_inches(18.5, 4.5)

# %% [markdown]
# # Write your MLP

# %%
"""
Placeholders for input 
NB images are expressed in terms of vector and not matrices. 
"""
x = tf.placeholder(tf.float32, shape=[None, 784])

# Placeholder for targets
targets = tf.placeholder(tf.float32, shape=[None, 10])

# Placeholder for discerning train/eval mode
is_training = tf.placeholder(dtype=tf.bool)


def perceptronLayer(x: Tensor, out_dim, batch_norm=False, activation_fn=tf.sigmoid, name="hidden_layer"):
    in_dim = x.get_shape().as_list()[1]
    with tf.variable_scope(name):
        w = tf.Variable(initial_value=tf.random_normal((in_dim, out_dim)))
        b = tf.Variable(initial_value=tf.random_normal((out_dim,)))
        prod = tf.einsum('ij,jk->ik', x, w)
        if(batch_norm):
            with tf.variable_scope('batch_norm'):
                mean, var = tf.nn.moments(prod, axes=0)
                v = tf.Variable(tf.ones([out_dim]))
                prod = tf.div_no_nan(prod - mean, tf.sqrt(var + 1e-3))
                prod = v * prod + b
        else:
            prod = prod + b

        if(activation_fn != None):
            return activation_fn(prod)
        else:
            return prod


def apply_dropout(x, keep_prob):
    def _():
        p = tf.random_normal([tf.shape(x)[0]])
        return tf.where(tf.less_equal(p, keep_prob), x, tf.zeros_like(x))
    return _


def dropout(x: Tensor, is_training, keep_prob=0.5):
    with tf.variable_scope("dropout"):
        return tf.cond(tf.equal(is_training, True), true_fn=apply_dropout(x, keep_prob), false_fn=lambda: x)


def batch_normalization(x: Tensor):
    mean = tf
    pass


def inference_tf(x, is_training, n_hidden=256):

    input_dim = 784
    n_classes = 10
    layers = [n_hidden, 64]

    with tf.variable_scope('network'):
        """
        Write HERE your multi layer perceptron (MLP), with one hidden layer 
        characterised by n_hidden neurons, activated by one OF the following
        activation functions: sigmoid, relu, leaky_relu. 

        Please note that the last layer should be followed by a softmax 
        activation, the latter giving a conditional distribution across n_classes. 

        Here a list of functions that you may use: 
          - tf.Variable, tf.matmul for hand-made dense layers
          - tf.contrib.layers.fully_connected
          - tf.contrib.layers.dropout
          - tf.layers.batch_normalization

        OPTIONALLY: 
        i) Add more than just one hidden layers. 
        ii) Put dropout and batch normalization layers to respectively improve
        the generalization capabilities and speedup training procedures.
        """

        data = x

        for l in layers:
            data = tf.contrib.layers.fully_connected(
                data, l, activation_fn=None)
            data = tf.layers.batch_normalization(data, training=is_training)
            data = tf.nn.relu(data)
            data = tf.contrib.layers.dropout(
                data, keep_prob=0.8, is_training=is_training)

        y = tf.contrib.layers.fully_connected(
            data, n_classes, activation_fn=None)
        y = tf.layers.batch_normalization(y, training=is_training)

        return tf.nn.softmax(y)


def inference(x, is_training, n_hidden=256):

    # input_dim = 784
    n_classes = 10
    layers = [n_hidden, 64]

    with tf.variable_scope('network'):
        """
        Write HERE your multi layer perceptron (MLP), with one hidden layer 
        characterised by n_hidden neurons, activated by one OF the following
        activation functions: sigmoid, relu, leaky_relu. 

        Please note that the last layer should be followed by a softmax 
        activation, the latter giving a conditional distribution across n_classes. 

        Here a list of functions that you may use: 
          - tf.Variable, tf.matmul for hand-made dense layers
          - tf.contrib.layers.fully_connected
          - tf.contrib.layers.dropout
          - tf.layers.batch_normalization

        OPTIONALLY: 
        i) Add more than just one hidden layers. 
        ii) Put dropout and batch normalization layers to respectively improve
        the generalization capabilities and speedup training procedures.
        """

        data = x

        for l in layers:
            data = perceptronLayer(data, l, batch_norm=True)
            data = dropout(data, is_training, keep_prob=0.4)

        y = perceptronLayer(
            data, n_classes, batch_norm=True, activation_fn=tf.nn.softmax, name="out_layer")

        return y


# Define model output
y = inference(x, is_training)

# %%

# Define metrics
pred_index = tf.argmax(y, axis=1)
targ_index = tf.argmax(targets, axis=1)
accuracy = tf.reduce_mean(
    tf.cast(tf.equal(pred_index, targ_index), dtype=tf.float32))

# Define loss function, namely the Categorical Cross Entropy
loss = -tf.reduce_sum(tf.reduce_mean(targets * tf.log(y), axis=1))

# UPDATE_OPS necessary to use batch normalization
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    # Define train step using the AdamOptimizer
    train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

# %% [markdown]
# # Training Procedure

# %%
init_op = tf.global_variables_initializer()

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)

    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    # Initialize all variables
    sess.run(init_op)

    # Training parameters
    training_epochs = 15
    batch_size = 128

    # Number of batches to process to see whole dataset
    batches_each_epoch = mnist.train.num_examples // batch_size

    for epoch in range(training_epochs):

        # During training measure accuracy on validation set to have an idea of what's happening
        val_accuracy = sess.run(fetches=accuracy,
                                feed_dict={x: mnist.validation.images,
                                           targets: mnist.validation.labels,
                                           is_training: False})

        print(
            'Epoch: {:06d} - VAL accuracy: {:.03f} %'.format(epoch, val_accuracy * 100))

        for _ in range(batches_each_epoch):

            # Load a batch of training data
            x_batch, target_batch = mnist.train.next_batch(batch_size)

            # Actually run one training step here
            sess.run(fetches=[train_step],
                     feed_dict={x: x_batch, targets: target_batch, is_training: True})

    test_accuracy = sess.run(fetches=accuracy,
                             feed_dict={x: mnist.test.images,
                                        targets: mnist.test.labels,
                                        is_training: False})
    print('*' * 50)
    print('Training ended. TEST accuracy: {:.03f} %'.format(
        test_accuracy * 100))
writer.close()
