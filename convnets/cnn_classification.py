import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from os.path import join
LOG_DIR = '/tmp/logs'

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
        x = tf.reshape(self.x, shape=(-1, h, w, c))
        
        # Apply a 3x3 convolution with 32 filters, relu activated and followed by a 2x2 max-pooling 
        with tf.name_scope("Conv3x3_32"):
            c1 = tf.layers.conv2d(x, filters=32, kernel_size=(3,3), padding="same",activation=tf.nn.relu)
            c1 = tf.layers.max_pooling2d(c1, pool_size=(2,2), strides=2)
        
        # Apply a 3x3 convolution with 64 filters, relu activated and followed by a 2x2 max-pooling 
        with tf.name_scope("Conv3x3_64"):
            c2 = tf.layers.conv2d(c1, filters=64, kernel_size=(3,3), padding="same",activation=tf.nn.relu)
            c2 = tf.layers.max_pooling2d(c2, pool_size=(2,2), strides=2)
        
        # Flatten out the activation map
        flat = tf.reshape(c2, shape=(-1, 7 * 7 * 64))
        
        # Apply dropout
        drop = tf.layers.dropout(flat, rate=0.5, training=self.training)
        
        # Final classification fully connected layer
        self.inference = tf.layers.dense(drop, self.n_classes, activation=tf.nn.softmax)

    def make_loss(self):
        # Make crossentropy loss
        self.loss = -tf.reduce_sum(
            tf.reduce_mean(self.targets * tf.log(self.inference), axis=1)
        )

    def make_train_step(self):
        self.train_step = tf.train.AdamOptimizer(0.001).minimize(self.loss)

    def make_accuracy(self):
        # make accuracy, using tf.argmax, tf.equal, tf.cast, tf.reduce_mean
        pred_index = tf.argmax(self.inference, axis=1)
        targ_index = tf.argmax(self.targets, axis=1)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(pred_index, targ_index), dtype=tf.float32)
        )

n_classes = 10
h, w, c = 28, 28, 1
training_epochs = 10
batch_size = 128
mnist = get_mnist_data('/tmp/mnist', verbose=True)

x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
targets = tf.placeholder(dtype=tf.float32, shape=[None, 10])
training = tf.placeholder(dtype=tf.bool)

# Define model
model = TinyConvnet(x, targets, training, n_classes, data_shape=(h, w, c))

# Loss summary
merged_train = tf.summary.merge([tf.summary.scalar('Loss', model.loss)])
train_writer = tf.summary.FileWriter(join(LOG_DIR, 'train'))
train_steps = 0

# Accuracy summary
merget_val = tf.summary.merge([tf.summary.scalar('Accuracy', model.accuracy)])
val_writer = tf.summary.FileWriter(join(LOG_DIR, 'val'))
val_steps = 0

config = tf.ConfigProto()
sess = tf.Session(config=config)

# Initialize all variables
sess.run(tf.global_variables_initializer())

# Number of batches to process to see whole dataset
batches_each_epoch = mnist.train.num_examples // batch_size

for epoch in range(training_epochs):

    # During training measure accuracy on validation set to have an idea of what's happening
    val_accuracy, summary = sess.run([model.accuracy, merget_val], feed_dict={
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


