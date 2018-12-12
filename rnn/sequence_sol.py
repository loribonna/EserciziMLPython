#%% [markdown]
# Import proper packages

#%%
import pickle
import argparse
import numpy as np
import tensorflow as tf
from os import makedirs
from os.path import exists
from os.path import dirname
from random import shuffle

epsilon = np.finfo(float).eps

#%% [markdown]
# This class models the dataset used for today's lecture: each example is composed of binary sequences, and the target variable encodes how many "1" there are within each sequence.

#%%
class SyntheticSequenceDataset:

    def __init__(self, max_sequence_length=10, dataset_cache='/tmp/synthetic_dataset.pickle', force_recompute=True):

        self._data = None
        self.max_sequence_length = max_sequence_length
        self.force_recompute = force_recompute
        self.dataset_cache = dataset_cache

    @property
    def data(self):

        if not self._data:
            if not self.force_recompute and exists(self.dataset_cache):
                print('Loading dataset from cache...')
                with open(self.dataset_cache, 'rb') as dump_file:
                    dataset = pickle.load(dump_file)
            else:
                print('Recomputing dataset...')
                dataset = self._compute_dataset()
                if not exists(dirname(self.dataset_cache)):
                    makedirs(dirname(self.dataset_cache))
                with open(self.dataset_cache, 'wb') as dump_file:
                    pickle.dump(dataset, dump_file)

            # Store data
            self._data = dataset

        return self._data

    def _compute_dataset(self,):

        n = self.max_sequence_length
        num_examples = 2 ** n
        num_classes = n + 1

        # How many examples to use for training (others are for test)
        num_train_examples = int(0.8 * num_examples)

        # Generate 2**20 binary strings
        data_strings = [('{' + '0:0{}b'.format(n) + '}').format(i) for i in range(num_examples)]

        # Shuffle sequences
        shuffle(data_strings)

        # Cast to numeric each generated binary string
        data_x, data_y = [], []
        for i in range(num_examples):
            train_sequence = []
            for binary_char in data_strings[i]:
                value = int(binary_char)
                train_sequence.append([value])
            data_x.append(train_sequence)           # examples are binary sequences of int {0, 1}
            data_y.append(np.sum(train_sequence))   # targets are the number of ones in the sequence

        # Convert from categorical to one-hot
        data_y_one_hot = np.eye(num_classes)[data_y]

        # Separate suggested training and test data
        train_data      = data_x[:num_train_examples]
        train_targets   = data_y_one_hot[:num_train_examples]
        test_data       = data_x[num_train_examples:]
        test_targets    = data_y_one_hot[num_train_examples:]

        return train_data, train_targets, test_data, test_targets

#%% [markdown]
# Create and load the sequence dataset

#%%
hidden_size = 40
batch_size = 128
n_epochs = 128
max_sequence_length = 10

# Load dataset
synthetic_dataset = SyntheticSequenceDataset(max_sequence_length=max_sequence_length)
train_data, train_targets, test_data, test_targets = synthetic_dataset.data

#%% [markdown]
# This class represents our recurrent module for learning to count. Your job is to fill in all the methods building the graph for inference, loss function, train step.
# You may find [tf.contrib.rnn.LSTMCell](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMCell), [tf.nn.dynamic_rnn](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn), [tf.layers.dense](https://www.tensorflow.org/api_docs/python/tf/layers/dense) useful

#%%
# Define placeholders
data = tf.placeholder(shape=(None, max_sequence_length, 1), dtype=tf.float32) # Define the placeholder for the input data sequences
targets = tf.placeholder(shape=(None, max_sequence_length+1), dtype=tf.float32) # Define the placeholder for the ground truth

class DeepCounter:

    def __init__(self, x, targets, hidden_size):

        self.x = x
        self.targets = targets
        self.n_classes = targets.get_shape()[-1]

        self.hidden_size = hidden_size

        self.inference = None
        self.loss = None
        self.train_step = None
        self.accuracy = None

        self.make_inference()
        self.make_loss()
        self.make_train_step()
        self.make_accuracy()

    def make_inference(self):
        if self.inference is None:
            """
            TODO:
            1) Create a recurrent cell (RNN, LSTM, GRU etc..)
            2) Create a recurrent neural network specified by the cell, thus 
            performing a fully dynamic unrolling of the inputs (stored in self.x)
            3) Take the last output of the sequence
            4) Append a classification layer.
            """
            cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
            _, (final_state, _) = tf.nn.dynamic_rnn(cell=cell, inputs=self.x,
                                               dtype=tf.float32)
            print(final_state)
            self.inference = tf.layers.dense(final_state,
                                             max_sequence_length + 1,
                                             activation=tf.nn.softmax)

    def make_loss(self):
        if self.loss is None:
            self.loss = - tf.reduce_mean(self.targets * tf.log(self.inference + epsilon))

    def make_train_step(self):
        if self.train_step is None:
            optimizer = tf.train.AdamOptimizer()
            self.train_step = optimizer.minimize(self.loss)

    def make_accuracy(self):
        if self.accuracy is None:
            correct_predictions = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.inference, 1), tf.argmax(self.targets, 1)), tf.float32))
            self.accuracy = correct_predictions

deep_counter = DeepCounter(x=data, targets=targets, hidden_size=hidden_size)

#%% [markdown]
# #Training and Test
# 
# 
# 
# 
#%% [markdown]
# Training loop. Your role is to fill the code to run one optimization step

#%%
# Open session
sess = tf.Session()

# Initialize variables
sess.run(tf.global_variables_initializer())

batches_each_epoch = int(len(train_data)) // batch_size

print('\n' + 50*'*' + '\nTraining\n' + 50*'*')

# Train batch by batch
for epoch in range(0, n_epochs):

    start_idx = 0
    loss_current_epoch = []
    for _ in range(0, batches_each_epoch):

        # Load batch
        end_idx = start_idx + batch_size
        data_batch, target_batch = train_data[start_idx:end_idx], train_targets[start_idx:end_idx]

        # Run one optimization step on current step
        cur_loss, _ = sess.run([deep_counter.loss, deep_counter.train_step], feed_dict={data: data_batch, targets: target_batch})
        loss_current_epoch.append(cur_loss)

        # Update data pointer
        start_idx += batch_size
    
    if epoch % 10 == 0:
      print('Epoch {:02d} - Loss on train set: {:.02f}'.format(epoch, sum(loss_current_epoch)/batches_each_epoch))

#%% [markdown]
# Run test. Your role is to fill code to run the graph and compute accuracy

#%%
print('\n' + 50 * '*' + '\nTesting\n' + 50 * '*')

accuracy_score = 0.0
num_test_batches = int(len(test_data)) // batch_size
start_idx = 0

# Test batch by batch
for _ in range(0, num_test_batches):
    end_idx = start_idx + batch_size
    data_batch, target_batch = test_data[start_idx:end_idx], test_targets[start_idx:end_idx]
    # compute accuracy on batch
    accuracy_score += sess.run(deep_counter.accuracy, feed_dict={data: data_batch, targets: target_batch})
    start_idx += batch_size

print('Average accuracy on test set: {:.03f}'.format(accuracy_score / num_test_batches))

#%% [markdown]
# Interactive section, just try to stress the model typing sequences yourself! :)

#%%
print('\n' + 50 * '*' + '\nInteractive Session\n' + 50 * '*')

while True:
    my_sequence = raw_input('Write your own binary sequence 10 digits in {0, 1}:\n')
    if my_sequence:

        # Pad shorter sequences
        if len(my_sequence) < max_sequence_length:
            my_sequence = (max_sequence_length - len(my_sequence))*'0' + my_sequence

        # Crop longer sequences
        my_sequence = my_sequence[:max_sequence_length]

        # Prepare example
        test_example = []
        for binary_char in my_sequence:
            test_example.append([float(binary_char)])

        pred = sess.run(deep_counter.inference, feed_dict={data: np.expand_dims(test_example, 0)})
        print('Predicted number of ones: {} - Real: {}\n'.format(int(np.argmax(pred)), int(np.sum(test_example))))
    else:
        break


