#%% [markdown]
# Install proper packages

#%%
#get_ipython().system('pip install googledrivedownloader')

#%% [markdown]
# Import proper packages and download tweets data

#%%
import re
import csv
import numpy as np
import tensorflow as tf
from collections import Counter
from google_drive_downloader import GoogleDriveDownloader


GoogleDriveDownloader.download_file_from_google_drive(file_id='1fHezNVY4YWJVWYb_3P3kx2e9RstjY1OK',
                                                      dest_path='data/tweets.zip',
                                                      unzip=True)

#%% [markdown]
# Global parameters, nothing fancy for now

#%%
EPS = np.finfo('float32').eps       # machine precision for float32
MAX_TWEET_CHARS = 140               # each tweet is made by max. 140 characters

#%% [markdown]
# Data loading primitives, useful to load tweets and their annotation in a numerical representation

#%%
def preprocess(line):
    """
    Pre-process a string of text. Eventually add additional pre-processing here.
    """
    line = line.lower()               # turn to lowercase
    line = line.replace('\n', '')     # remove newlines
    line = re.sub(r'\W+', ' ', line)  # keep characters only (\W is short for [^\w])

    return line


def get_dictionary(filename, dict_size=2000):
    """
    Read the tweets and return a list of the 'max_words' most common words.
    """
    all_words = []
    with open(filename, 'r') as csv_file:
        r = csv.reader(csv_file, delimiter=',', quotechar='"')
        for row in r:
            tweet = row[3]
            if len(tweet) <= MAX_TWEET_CHARS:
                words = preprocess(tweet).split()
                all_words += words

    # Make the dictionary out of only the N most common words
    word_counter = Counter(all_words)
    dictionary, _ = zip(*word_counter.most_common(min(dict_size, len(word_counter))))

    return dictionary


class TweetLoader(object):

    def __init__(self, filename_train, filename_val, batchsize, max_len, dict_size):

        self._filename_train = filename_train
        self._filename_val = filename_val
        self._batchsize = batchsize
        self._max_len = max_len
        self._dict_size = dict_size

        # get the list of words that will constitute our dictionary (once only)
        self._dictionary = get_dictionary(self._filename_train, dict_size)

        self._train_rows = self.read_data(self._filename_train)
        self._val_rows = self.read_data(self._filename_val)

    def read_data(self, filename):
        # read training data
        rows = []
        with open(filename, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',', quotechar='"')
            for row in reader:
                rows.append(row)
        return rows

    def vectorize(self, tweet):
        words = preprocess(tweet).split()

        X = np.zeros(shape=(1, self._max_len, self._dict_size + 1))

        # Vectorization
        for j, w in enumerate(words):
            if j < self._max_len:
                try:
                    w_idx = self._dictionary.index(w)
                    X[0, j, w_idx + 1] = 1
                except ValueError:
                    # Word not found, using the unknown
                    X[0, j, 0] = 1

        return X

    def load_tweet_batch(self, mode):
        """
        Generate a batch of training data
        """
        assert mode in ['train', 'val']
        if mode == 'train':
            rows = self._train_rows
        else:
            rows = self._val_rows

        # prepare data structures
        X_batch = np.zeros((self._batchsize, self._max_len, len(self._dictionary) + 1), dtype=np.float32)
        Y_batch = np.zeros((self._batchsize, 2), dtype=np.float32)

        tweet_loaded = 0
        while tweet_loaded < self._batchsize:

            rand_idx = np.random.randint(0, len(rows))
            Y_batch[tweet_loaded, int(rows[rand_idx][1])] = 1

            random_tweet = rows[rand_idx][3]
            if len(random_tweet) <= MAX_TWEET_CHARS:

                X = self.vectorize(tweet=random_tweet)
                X_batch[tweet_loaded] = X[0]
                tweet_loaded += 1

        return X_batch, Y_batch

#%% [markdown]
# Define our recurrent model for sentiment analysis on tweets: the functions you have to implement are make_inference, make_loss, make_train_step.

#%%
def length(sequence):
    """
    Returns the useful lenght of the sequence.
    """
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


def last_relevant(output, length):
    """
    Returns the indexes of the last relevant element of the sequence.
    """
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant


class TweetModel(object):

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
        cell = tf.nn.rnn_cell.GRUCell(self.hidden_size, activation=tf.nn.tanh)

        out, state = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)

        last = out[:,-1,:]

        self.inference = tf.layers.dense(last, self.n_classes, activation=tf.nn.softmax)

    def make_loss(self):
        self.loss = -tf.reduce_mean(tf.reduce_sum(self.targets * tf.log(self.inference)))

    def make_train_step(self):
        self.train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
        
    def make_accuracy(self):
        pred_index = tf.argmax(self.inference, axis=1)
        targ_index = tf.argmax(self.targets, axis=1)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(pred_index, targ_index), dtype=tf.float32)
        )

#%% [markdown]
# #My solution

#%%
def length(sequence):
    """
    Returns the useful lenght of the sequence.
    """
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


def last_relevant(output, length):
    """
    Returns the indexes of the last relevant element of the sequence.
    """
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant

"""
class TweetModel(object):

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

        # Create LSTM cell with proper hidden size
        cell = tf.contrib.rnn.LSTMCell(self.hidden_size)

        # Get LSTM output
        val, state = tf.nn.dynamic_rnn(cell, self.x, sequence_length=length(self.x), dtype=tf.float32)

        # Get last output of LSTM
        last = last_relevant(val, length(val))

        # Define the final prediction applying a fully connected layer with softmax
        self.inference = tf.layers.dense(state[0], self.n_classes, activation=tf.nn.softmax)

    def make_loss(self):
        self.loss = - tf.reduce_sum(self.targets * tf.log(self.inference + EPS))

    def make_train_step(self):
        self.train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

    def make_accuracy(self):
        mistakes = tf.equal(tf.argmax(self.inference, axis=1), tf.argmax(self.targets, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(mistakes, tf.float32))
"""

#%% [markdown]
# #Training and Test
#%% [markdown]
# Define some hyperparameters

#%%
# Model parameters
max_seq_len = 20  # Maximum length (in words) of tweets
max_dict_size = 1000  # Maximum dictionary size
hidden_size = 10  # LSTM cell dimension
train_tweets_path = 'data/tweets_train.csv'
val_tweets_path = 'data/tweets_val.csv'

# Training parameters
training_epochs = 20 # Number of training epochs
batch_size = 32 # Size of each training batch
batches_each_epoch = 500 # Number of batches for each epochs

#%% [markdown]
# Get the TweetLoader that will provide us training batches

#%%
# Get tweet loader
loader = TweetLoader(train_tweets_path, val_tweets_path, batch_size, max_seq_len, max_dict_size)
x_batch, y_batch = loader.load_tweet_batch(mode='train')

#%% [markdown]
# Define placeholders for tweets (x) and sentiments (target). Be extra-careful to shapes!

#%%
# Declare placeholders
x = tf.placeholder(shape=(None, max_seq_len, max_dict_size+1), dtype=tf.float32)
targets = tf.placeholder(shape=(None, 2), dtype=tf.float32)

#%% [markdown]
# Instantiate the model.

#%%
# Get a model
model = TweetModel(x, targets, hidden_size)

#%% [markdown]
# Open tensorflow session...

#%%
# Open new session
sess = tf.Session()

#%% [markdown]
# ... and initialize variables.

#%%
# Initialize all variables
sess.run(tf.global_variables_initializer())

#%% [markdown]
# Training loop, fill in blanks following comments

#%%
for epoch in range(training_epochs):

    x_batch, y_batch = loader.load_tweet_batch(mode='train')
    print('Epoch: {}\tTRAIN: Loss: {:.02f} Accuracy: {:.02f}'.format(
        epoch,
        # Compute batch loss
        sess.run(model.loss, feed_dict={x: x_batch, targets: y_batch}),
        # Compute batch accuracy
        sess.run(model.accuracy, feed_dict={x: x_batch, targets: y_batch})
    ))

    x_batch, y_batch = loader.load_tweet_batch(mode='val')
    print('Epoch: {}\tVAL: Loss: {:.02f} Accuracy: {:.02f}'.format(
        epoch,
        # Compute batch loss
        sess.run(model.loss, feed_dict={x: x_batch, targets: y_batch}),
        # Compute batch accuracy
        sess.run(model.accuracy, feed_dict={x: x_batch, targets: y_batch})
    ))

    for _ in range(batches_each_epoch):

        # Load a batch of training data
        x_batch, y_batch = loader.load_tweet_batch(mode='train')

        # Actually run one training step here
        sess.run(fetches=[model.train_step], feed_dict={x: x_batch, targets: y_batch})


#%%
# Interactive session
while True:
    tw = input('Try tweeting something!')
    if tw:
        x_num = loader.vectorize(tweet=tw)
        p, = sess.run([model.inference], feed_dict={x: x_num})
        if np.argmax(p) == 0:
            # Negative tweet
            print('Prediction:{}\t:('.format(p))
        else:
            print('Prediction:{}\t:)'.format(p))
    else:
        break


