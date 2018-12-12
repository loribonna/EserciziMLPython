#%% [markdown]
# These lines are for making tensorboard visualization work within the iPython notebook environment. 

#%%
get_ipython().system(' wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip')
get_ipython().system(' unzip -o ngrok-stable-linux-amd64.zip')
get_ipython().system_raw('./ngrok http 6006 &')

# Start Tensorboard server
LOG_DIR = '/tmp/logs'
get_ipython().system_raw('rm -rf {}'.format(LOG_DIR))
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOG_DIR)
)

#%% [markdown]
# Print the public url in which we can find tensorboard.

#%%
get_ipython().system(' curl -s http://localhost:4040/api/tunnels | python -c     "import sys, json; print(json.load(sys.stdin)[\'tunnels\'][0][\'public_url\'])"')

#%% [markdown]
# Import packages as usual.

#%%
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from os.path import join

#%% [markdown]
# Helper functions for data loading (MNIST).

#%%
def get_mnist_data(download_data_path, one_hot=True, verbose=False):
    """

    Parameters
    ----------
    download_data_path : string
        Directory where MNIST data are downloaded and extracted.
    one_hot : bool
        If True, targets are returned into one-hot format
    verbose : bool
        If True, print dataset tensors dimensions

    Returns
    -------
    mnist : Dataset
        Structure containing train, val and test mnist dataset in a friendly format.
    """

    # Download and read in MNIST dataset
    mnist = input_data.read_data_sets(download_data_path, one_hot=one_hot)

    if verbose:

        # Print image tensors shapes
        print('TRAIN tensor shape: {}'.format(mnist.train.images.shape))
        print('VAL   tensor shape: {}'.format(mnist.validation.images.shape))
        print('TEST  tensor shape: {}'.format(mnist.test.images.shape))

        # Print labels shape (encoded as one-hot vectors)
        print('TRAIN labels shape: {}'.format(mnist.train.labels.shape))
        print('VAL   labels shape: {}'.format(mnist.validation.labels.shape))
        print('TEST  labels shape: {}'.format(mnist.test.labels.shape))

    return mnist

#%% [markdown]
# Our magic epsilon for cross-entropy loss regularization.

#%%
epsilon = np.finfo(np.float32).eps

#%% [markdown]
# A ConvNet model, it takes placeholder and other information in the constructor. Your job is to implement the four make_* functions, as usual, defining the graph for inference, the loss, the training step and the accuracy. Useful tensorflow API functions: [tf.reshape](https://www.tensorflow.org/api_docs/python/tf/reshape), [tf.layers.conv2d](https://www.tensorflow.org/api_docs/python/tf/layers/conv2d), [tf.nn.relu](https://www.tensorflow.org/api_docs/python/tf/nn/relu), [tf.layers.dropout](https://www.tensorflow.org/api_docs/python/tf/layers/dropout), [tf.layers.dense](https://www.tensorflow.org/api_docs/python/tf/layers/dense)

#%%
class TinyConvnet:

    def __init__(self, x, targets, training, n_classes, data_shape):
        """
        x: placeholder for input data
        targets: placeholder for labels
        training: placeholder for training phase (bool)
        n_classes: integer
        data_shape: tuple (28, 28, 1)
        """

        self.x = x
        self.targets = targets
        self.training = training
        self.n_classes = n_classes
        self.data_shape = data_shape

        self.inference = None
        self.loss = None
        self.train_step = None
        self.accuracy = None

        self.make_inference()
        self.make_loss()
        self.make_train_step()
        self.make_accuracy()

    def make_inference(self):

        h, w, c = self.data_shape

        # Reshape flattened input into images
        x = tf.reshape(self.x, (-1, 28, 28, 1))
        
        # Apply a 3x3 convolution with 32 filters, relu activated and followed by a 2x2 max-pooling 
        h = tf.layers.conv2d(x, 32, (3,3), padding='same', activation='relu')
        h = tf.layers.max_pooling2d(h, 2, 2)
        
        
        # Apply a 3x3 convolution with 64 filters, relu activated and followed by a 2x2 max-pooling 
        h = tf.layers.conv2d(h, 64, (3,3), padding='same', activation='relu')
        h = tf.layers.max_pooling2d(h, 2, 2)
        
        # Flatten out the activation map
        h = tf.reshape(h, (-1, 64*7*7))
        
        # Apply dropout
        h = tf.layers.dropout(h, 0.5, training=self.training)

        # Final classification fully connected layer
        self.inference = tf.layers.dense(h, 10, activation='softmax')

    def make_loss(self):
        # Make crossentropy loss
        self.loss = - tf.reduce_mean(self.targets * tf.log(self.inference + epsilon))

    def make_train_step(self):
        self.train_step = tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.9).minimize(self.loss)

    def make_accuracy(self):
        # make accuracy, using tf.argmax, tf.equal, tf.cast, tf.reduce_mean
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.targets, axis=1), tf.argmax(self.inference, axis=1)), tf.float32))

#%% [markdown]
# Some parameters...

#%%
# MNIST parameters
n_classes = 10
h, w, c = 28, 28, 1

# Training parameters
training_epochs = 10
batch_size = 128

#%% [markdown]
# Get the MNIST dataset.

#%%
# Load MNIST data
mnist = get_mnist_data('/tmp/mnist', verbose=True)

#%% [markdown]
# Define placeholders. As usual, define for each placeholder shapes and dtype.

#%%
# Placeholders
x = tf.placeholder(shape=(None, 784), dtype=tf.float32)
targets = tf.placeholder(shape=(None, 10), dtype=tf.float32)
training = tf.placeholder(shape=(), dtype=tf.bool)

#%% [markdown]
# Instantiate a TinyConvnet model.

#%%
# Define model
model = TinyConvnet(x, targets, training, n_classes, data_shape=(h, w, c))

#%% [markdown]
# This is how you define summaries to be logged for tensorboard visualization. Summaries are then evaluated within the session as graph nodes and provided to a SummaryWriter (see below).

#%%
# Loss summary
merged_train = tf.summary.merge([tf.summary.scalar('Loss', model.loss)])
train_writer = tf.summary.FileWriter(join(LOG_DIR, 'train'))
train_steps = 0

# Accuracy summary
merged_val = tf.summary.merge([tf.summary.scalar('Accuracy', model.accuracy)])
val_writer = tf.summary.FileWriter(join(LOG_DIR, 'val'))
val_steps = 0

#%% [markdown]
# Start session and initialize variables.

#%%
sess = tf.Session()

# Initialize all variables
sess.run(tf.global_variables_initializer())

#%% [markdown]
# Training loop! Now with summaries!

#%%
# Number of batches to process to see whole dataset
batches_each_epoch = mnist.train.num_examples // batch_size

for epoch in range(training_epochs):

    # During training measure accuracy on validation set to have an idea of what's happening
    val_accuracy, summary = sess.run([model.accuracy, merged_val], feed_dict={
        x: mnist.validation.images,
        targets: mnist.validation.labels,
        training: False
    })
    val_writer.add_summary(summary, global_step=val_steps)
    val_steps += 1

    print('Epoch: {:06d} - VAL accuracy: {:.03f}'.format(epoch, val_accuracy))

    for _ in range(batches_each_epoch):

        # Load a batch of training data
        x_batch, target_batch = mnist.train.next_batch(batch_size)

        # Actually run one training step here, and compute summaries for train
        _, summary = sess.run([model.train_step, merged_train], feed_dict={
            x: x_batch,
            targets: target_batch,
            training: True
        })
        
        train_writer.add_summary(summary, global_step=train_steps)
        train_steps += 1

#%% [markdown]
# Test.

#%%
# Eventually evaluate on whole test set when training ends
average_test_accuracy = 0.0
num_test_batches = mnist.test.num_examples // batch_size
for _ in range(num_test_batches):
    x_batch, target_batch = mnist.test.next_batch(batch_size)
    
    # Compute batch accuracy
    average_test_accuracy += sess.run(model.accuracy, feed_dict={
        x: x_batch,
        targets: target_batch,
        training: False
    })
    
average_test_accuracy /= num_test_batches
print('*' * 50)
print('Training ended. TEST accuracy: {:.03f}'.format(average_test_accuracy))


