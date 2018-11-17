
#%%
get_ipython().system('pip install googledrivedownloader')
get_ipython().system('pip install scikit-image --upgrade')


#%%
import numpy as np
from matplotlib import pyplot as plt

from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id='1pzx_VdV_SwEgYnAZBQ0iZg0QDptA2UtO',dest_path='/tmp/X_img_train.npy',unzip=False)
gdd.download_file_from_google_drive(file_id='1SMvIaSEsiDgBDmk7IvtQObk7x7s2rRwx',dest_path='/tmp/X_feat_train.npy',unzip=False)
gdd.download_file_from_google_drive(file_id='1FbygsCk1g-Lq6TXX7y-CPZSGc7d04LSx',dest_path='/tmp/X_img_test.npy',unzip=False)
gdd.download_file_from_google_drive(file_id='1v_owXbovNXDz6EYSDgsgVH7VvQKF8Jwq',dest_path='/tmp/X_feat_test.npy',unzip=False)
gdd.download_file_from_google_drive(file_id='1s47YpYmZkH9l-u1sXHIWF51BYx9HjFBz',dest_path='/tmp/Y_test.npy',unzip=False)
gdd.download_file_from_google_drive(file_id='1vrnS0GbyQkQQUwyVY8TAzo_m0ESGxRWl',dest_path='/tmp/Y_train.npy',unzip=False)

X_img_train, X_feat_train, Y_train, X_img_test, X_feat_test, Y_test = np.load('/tmp/X_img_train.npy'), np.load('/tmp/X_feat_train.npy'), np.load('/tmp/Y_train.npy'), np.load('/tmp/X_img_test.npy'), np.load('/tmp/X_feat_test.npy'), np.load('/tmp/Y_test.npy')


#%%
print("Img. shape: {0} - values in [{1},{2:.2f}]".format(X_img_train.shape,np.min(X_img_train),np.max(X_img_train)))
print("Feature shape: {0} - values in [{1},{2:.2f}]".format(X_feat_train.shape,np.min(X_feat_train),np.max(X_feat_train)))
print("Ground truth shape: {0} - classes: {1}".format(Y_train.shape,np.unique(Y_train)))


#%%
plt.subplot(121)
plt.title('Class 0. Non people')
X_0 = X_img_train[Y_train == 0.0]
random_idx_1 = np.random.choice(np.arange(0, X_0.shape[0]))
plt.imshow(X_0[random_idx_1], cmap='gray')
plt.grid(b=False)

plt.subplot(122)
plt.title('Class 1. People')
X_1 = X_img_train[Y_train == 1.0]
random_idx_2 = np.random.choice(np.arange(0, X_1.shape[0]))
plt.imshow(X_1[random_idx_2], cmap='gray')
plt.grid(b=False)

plt.show()


#%%
import tensorflow as tf

_, n_features = X_feat_train.shape

# define placeholders
X = tf.placeholder(dtype=tf.float32, shape=(None, n_features))
Y = tf.placeholder(dtype=tf.float32, shape=(None,))
W = tf.Variable(initial_value=np.random.randn(n_features,1), dtype=tf.float32)
b = tf.Variable(initial_value=np.zeros(1), dtype=tf.float32)

def sigmoid(X):
  return 1 / (1 + tf.exp((-X))) # return tf.nn.sigmoid(X)

def binary_crossentropy(y_true, y_pred):
  return - tf.reduce_mean(y_true * tf.log(y_pred) + (1 - y_true) * tf.log(1-y_pred))

def apply_threshold(y, threshold=0.5):
  return tf.where(tf.greater_equal(y, 0.5), tf.ones(tf.shape(y)), tf.zeros(tf.shape(y)))

def accuracy(y_true, y_pred):
  correct = tf.where(tf.math.equal(y_true,y_pred), tf.ones(tf.shape(y_pred)), tf.zeros(tf.shape(y_pred)))
  return tf.reduce_mean(correct)

NUM_EPOCHS = 10000
    
Z = tf.matmul(X,W) + b
Z = tf.squeeze(Z)
Z = sigmoid(Z)

Z_0_1 = apply_threshold(Z)
loss = binary_crossentropy(y_true=Y, y_pred=Z)
acc = accuracy(y_true=Y, y_pred=Z_0_1)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1)

train_step = optimizer.minimize(loss)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for i in range(0, NUM_EPOCHS):
      sess.run(train_step, feed_dict={X: X_feat_train, Y:Y_train})
      if i % 1000 == 0: 
        train_loss, train_acc = sess.run([loss, acc], feed_dict={X: X_feat_train, Y:Y_train})   
        test_loss, test_acc = sess.run([loss, acc], feed_dict={X: X_feat_test, Y:Y_test})  
        Y_test_pred = sess.run(Z_0_1, feed_dict={X: X_feat_test, Y:Y_test})  
        print("Training Loss (CE) : {:0.2f}; Training acc. : {:0.2f}; Test Loss (CE) : {:0.2f} Test acc. : {:0.2f}; ".format(train_loss, train_acc, test_loss, test_acc))


#%%
labels = ['Non people', 'People']
num_row, num_col = 2, 6
f,subplots = plt.subplots(num_row, num_col, sharex='col', sharey='row')

for i in range(num_row):
    for j in range(num_col):
        idx = np.random.choice(np.arange(0, X_img_test.shape[0]))
        subplots[i,j].imshow(X_img_test[idx], cmap='gray', interpolation='nearest', aspect='auto')
        title = 'GT: {} \n Pred: {}'.format(labels[int(Y_test[idx])], labels[int(Y_test_pred[idx])])
        color_title = 'green' if int(Y_test[idx]) == int(Y_test_pred[idx]) else 'red'
        subplots[i,j].set_title(title, color=color_title, fontweight="bold")
        subplots[i,j].grid(b=False)

f.set_size_inches(13.5, 7.5)


