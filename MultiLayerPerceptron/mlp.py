
# %%
import tensorflow as tf
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


def perceptronLayer(x, in_dim, out_dim, activation_fn=tf.sigmoid, name="hidden_layer"):
    with tf.variable_scope(name):
        w = tf.Variable(initial_value=tf.random_normal((in_dim, out_dim)))
        b = tf.Variable(initial_value=tf.random_normal((out_dim,)))
        prod = tf.einsum('ij,jk->ik', x, w) + b
        if(activation_fn != None):
            return activation_fn(prod)
        else:
            return prod

def inference(x, is_training, n_hidden=256):

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

        #layer = tf.contrib.layers.fully_connected(x, n_hidden, activation_fn=tf.nn.sigmoid)
        #out_t = tf.contrib.layers.fully_connected(layer, 10, activation_fn=tf.nn.softmax)
        data = x
        for l in layers:
            data = perceptronLayer(data, input_dim, l)
            input_dim = l

        y = perceptronLayer(data, layers[-1], n_classes, activation_fn=None, name="out_layer")

        return tf.nn.softmax(y)


# Define model output
y = inference(x, is_training)

# %%

# Define metrics
pred_index = tf.argmax(y, axis=1)
targ_index = tf.argmax(targets, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_index, targ_index), dtype=tf.float32))

# Define loss function, namely the Categorical Cross Entropy
loss = -tf.reduce_sum(tf.reduce_mean(targets * tf.log(y), axis=1))

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
    print('Training ended. TEST accuracy: {:.03f} %'.format(test_accuracy * 100))
writer.close()
