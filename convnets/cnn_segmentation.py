#%% [markdown]
# These lines are for making tensorboard visualization work within the iPython notebook environment. 

#%%
#get_ipython().system(' wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip')
#get_ipython().system(' unzip -o ngrok-stable-linux-amd64.zip')
#get_ipython().system_raw('./ngrok http 6006 &')

# Start Tensorboard server
from datetime import datetime
now = datetime.now()
LOG_DIR = './tmp/logs/' + now.strftime("%Y%m%d-%H%M%S") + "/"
#get_ipython().system_raw('rm -rf {}'.format(LOG_DIR))
#get_ipython().system_raw(
#    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
#    .format(LOG_DIR)
#)

#%% [markdown]
# Print the public url in which we can find tensorboard.

#%%
#get_ipython().system(' curl -s http://localhost:4040/api/tunnels | python -c     "import sys, json; print(json.load(sys.stdin)[\'tunnels\'][0][\'public_url\'])"')

#%% [markdown]
# Install googledrivedownloader.

#%%
#get_ipython().system('pip install googledrivedownloader')

#%% [markdown]
# Import packages as usual, download the dataset.

#%%
import cv2
import pickle
import numpy as np
import tensorflow as tf
import os.path as path
from time import time

from google_drive_downloader import GoogleDriveDownloader

# Download tiles data
GoogleDriveDownloader.download_file_from_google_drive(file_id='1W58D4qVZtUAFprDdoC9KRyBWT0k0ie5r',
                                                      dest_path='./tmp/tiles.zip',
                                                      overwrite=False,
                                                      unzip=True)

#%% [markdown]
# This class models the Tiles dataset for segmentation of images. Once instantiated, use its members train_x, train_y, validation_x, validation_y, test_x, test_y.

#%%
class TilesDataset:

    def __init__(self, dataset_root):

        self.dataset_root = dataset_root

        # Store locations of train, val and test directories
        self.train_x_dir      = path.join(dataset_root, 'X_train')
        self.train_y_dir      = path.join(dataset_root, 'Y_train')
        self.validation_x_dir = path.join(dataset_root, 'X_validation')
        self.validation_y_dir = path.join(dataset_root, 'Y_validation')
        self.test_x_dir       = path.join(dataset_root, 'X_test')
        self.test_y_dir       = path.join(dataset_root, 'Y_test')

        # Number of dataset examples
        self.train_num_examples      = 10000
        self.validation_num_examples = 1000
        self.test_num_examples       = 1000

        # Initialize empty structures to contain data
        self.train_x      = []
        self.train_y      = []
        self.validation_x = []
        self.validation_y = []
        self.test_x       = []
        self.test_y       = []

        # Load images from `dataset_root`
        self._fill_data_arrays()

    def _fill_data_arrays(self):

        # Load training images
        for i in range(1, self.train_num_examples + 1):
            #print('Loading training examples. {} / {}...'.format(i, self.train_num_examples))
            x_image = cv2.imread(path.join(self.train_x_dir, '{:05d}.png'.format(i)))
            y_image = cv2.imread(path.join(self.train_y_dir, '{:05d}.png'.format(i)), cv2.IMREAD_GRAYSCALE)
            self.train_x.append(x_image.astype(np.float32))
            self.train_y.append(np.expand_dims(y_image.astype(np.float32), 2))

        # Load validation examples
        for i in range(1, self.validation_num_examples + 1):
            #print('Loading validation examples. {} / {}...'.format(i, self.validation_num_examples))
            x_image = cv2.imread(path.join(self.validation_x_dir, '{:05d}.png'.format(i)))
            y_image = cv2.imread(path.join(self.validation_y_dir, '{:05d}.png'.format(i)), cv2.IMREAD_GRAYSCALE)
            self.validation_x.append(x_image.astype(np.float32))
            self.validation_y.append(np.expand_dims(y_image.astype(np.float32), 2))

        # Load test examples
        for i in range(1, self.test_num_examples + 1):
            #print('Loading test examples. {} / {}...'.format(i, self.test_num_examples))
            x_image = cv2.imread(path.join(self.test_x_dir, '{:05d}.png'.format(i)))
            y_image = cv2.imread(path.join(self.test_y_dir, '{:05d}.png'.format(i)), cv2.IMREAD_GRAYSCALE)
            self.test_x.append(x_image.astype(np.float32))
            self.test_y.append(np.expand_dims(y_image.astype(np.float32), 2))

    def dump_to_file(self, file_path, protocol=pickle.HIGHEST_PROTOCOL):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f, protocol=protocol)


def convert_target_to_one_hot(target_batch):
    """
    Convert a batch of targets from (height,width,1) to (height,width,2) one-hot encoding.
    """
    b, h, w, c = target_batch.shape
    out_tensor = np.zeros(shape=(b, h, w, 2))
    for k, cur_example in enumerate(target_batch):
        foreground_mask = np.squeeze(cur_example > 0)
        background_mask = np.squeeze(cur_example == 0)
        out_tensor[k, background_mask, 0] = 1.0
        out_tensor[k, foreground_mask, 1] = 1.0
    return out_tensor

#%% [markdown]
# This class implements the deep model for tiles segmentation. Implement all make_* methods (except for summaries).

#%%
class TileSegmenter:

    def __init__(self, x, targets, data_shape):

        self.x = x
        self.targets = targets
        self.data_shape = data_shape

        self.inference = None
        self.loss = None
        self.train_step = None
        self.summaries = None

        self.make_inference()
        self.make_loss()
        self.make_train_step()
        self.make_summaries()

    def make_inference(self):
      
        # Use 2D conv with strides. Stack enough layers to reduce dimensions a bit, 
        # then upsample using tf.image.resize_bilinear, then apply a final conv to reach 2
        # channel outputs.

        h, w, c = self.data_shape
        x = self.x

        conv = tf.layers.conv2d(x, filters=8, kernel_size=(3,3), activation=tf.nn.relu)
        for l in range(0, 10):
            conv = tf.layers.conv2d(conv, filters=16, kernel_size=(3,3), activation=tf.nn.relu)

        conv = tf.image.resize_bilinear(conv, size=(h, w))
 
        self.inference = tf.layers.conv2d(conv, padding="same", filters=2, kernel_size=(3,3), activation=tf.nn.softmax)
        #self.inference = tf.layers.dense(conv, 2, activation=tf.nn.softmax)


    def make_loss(self):
        # Define loss function
        
        loss = tf.reduce_mean(self.targets * tf.log(self.inference), axis=3)

        loss = tf.reduce_mean(loss)
        self.loss = -loss

    def make_train_step(self):
        self.train_step = tf.train.AdamOptimizer(0.001).minimize(self.loss)

    def make_summaries(self):
        
        # Add TensorBoard Summaries
        how_many_images = 3

        # --- scalar summaries
        tf.summary.scalar('loss', self.loss)
        tf.summary.image('input', self.x, max_outputs=how_many_images)

        # --- foreground image summaries
        fg_target_image = tf.expand_dims(tf.gather(tf.transpose(self.targets, [3, 0, 1, 2]), 1), axis=3)
        fg_pred_image = tf.expand_dims(tf.gather(tf.transpose(self.inference, [3, 0, 1, 2]), 1), axis=3)
        fg_pred_image_rounded = tf.round(tf.nn.softmax(self.inference, dim=3))
        fg_pred_image_rounded = tf.expand_dims(tf.gather(tf.transpose(fg_pred_image_rounded, [3, 0, 1, 2]), 1), axis=3)

        tf.summary.image('FOREGROUND_(targets)', fg_target_image, max_outputs=how_many_images)
        tf.summary.image('FOREGROUND_(prediction)', fg_pred_image, max_outputs=how_many_images)
        tf.summary.image('FOREGROUND_ROUNDED_(prediction)', fg_pred_image_rounded, max_outputs=how_many_images)

        # --- merge all summaries and initialize the summary writer
        self.summaries = tf.summary.merge_all()

#%% [markdown]
# Some parameters...

#%%
# Training parameters
h, w, c = 64, 64, 3
training_epochs = 1000
batch_size = 64

#%% [markdown]
# Instantiate the dataset...

#%%
# Load tiles dataset
tiles_dataset = TilesDataset(dataset_root='./tmp/toy_dataset_tiles')

#%% [markdown]
# Placeholders

#%%
# Placeholders
x = tf.placeholder(dtype=tf.float32, shape=[None, h, w, c])
targets = tf.placeholder(dtype=tf.float32, shape=[None, h, w, 2])

#%% [markdown]
# Instantiate the model

#%%
# Define model
model = TileSegmenter(x, targets, data_shape=(h, w, c))

#%% [markdown]
# Start session and initialize variables.

#%%
sess = tf.Session()
        
# Initialize all variables
sess.run(tf.global_variables_initializer())

#%% [markdown]
# Set up the writer for tensorboard logs.

#%%
# FileWriter to save Tensorboard summary
train_writer = tf.summary.FileWriter(LOG_DIR, graph=sess.graph)

#%% [markdown]
# Training loop, with summaries!

#%%
# Number of batches to process to see whole dataset
batches_each_epoch = tiles_dataset.train_num_examples // batch_size

for epoch in range(training_epochs):

    epoch_loss = 0.0

    idx_start = 0
    for _ in range(batches_each_epoch):

        idx_end = idx_start + batch_size

        # Load a batch of training data
        x_batch = np.array(tiles_dataset.train_x[idx_start:idx_end])
        target_batch = np.array(tiles_dataset.train_y[idx_start:idx_end])
        # Convert the target batch into one-hot encoding (from 64x64x1 to 64x64x2)
        target_batch = convert_target_to_one_hot(target_batch)

        # Preprocess train batch
        x_batch -= 128.0

        # Actually run one training step here
        _, cur_loss = sess.run([model.train_step, model.loss], feed_dict={
            x: x_batch,
            targets: target_batch
        })

        idx_start = idx_end

        epoch_loss += cur_loss
    
    # Get summaries for the last batch
    summaries = sess.run(model.summaries, feed_dict={
        x: x_batch,
        targets: target_batch
    })
    train_writer.add_summary(summaries, epoch)

    print('Epoch: {:03d} - Loss: {:.02f}'.format(epoch, epoch_loss / batches_each_epoch))


