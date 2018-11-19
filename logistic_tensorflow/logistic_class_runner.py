import tensorflow as tf
from tensorflow import contrib
from tensorflow.contrib import eager as tfe
from matplotlib import pyplot as plt
import numpy as np
from google_drive_downloader import GoogleDriveDownloader as gdd
from logistic_class import LogisticRegressionModel, LogisticRegressionEagerModel


def eager_main(n_epochs=10000, plot=True):
    tfe.enable_eager_execution()
    X_img_train, X_feat_train, Y_train, X_img_test, X_feat_test, Y_test = load_data()

    _, n_features = X_feat_train.shape

    model = LogisticRegressionEagerModel(n_features, learning_rate=0.01)

    for e in range(0, n_epochs):
        model.train(X_feat_train, Y_train)

        if e % 1000 == 0:
            loss = model.train(X_feat_train, Y_train)
            acc = model.accuracy(X_feat_train, Y_train)
            print("Epoch Train {}: Loss {}; Accuracy: {}".format(int(e / 1000), loss, acc))

def load_data():
    gdd.download_file_from_google_drive(
        file_id='1pzx_VdV_SwEgYnAZBQ0iZg0QDptA2UtO', dest_path='/tmp/X_img_train.npy', unzip=False)
    gdd.download_file_from_google_drive(
        file_id='1SMvIaSEsiDgBDmk7IvtQObk7x7s2rRwx', dest_path='/tmp/X_feat_train.npy', unzip=False)
    gdd.download_file_from_google_drive(
        file_id='1FbygsCk1g-Lq6TXX7y-CPZSGc7d04LSx', dest_path='/tmp/X_img_test.npy', unzip=False)
    gdd.download_file_from_google_drive(
        file_id='1v_owXbovNXDz6EYSDgsgVH7VvQKF8Jwq', dest_path='/tmp/X_feat_test.npy', unzip=False)
    gdd.download_file_from_google_drive(
        file_id='1s47YpYmZkH9l-u1sXHIWF51BYx9HjFBz', dest_path='/tmp/Y_test.npy', unzip=False)
    gdd.download_file_from_google_drive(
        file_id='1vrnS0GbyQkQQUwyVY8TAzo_m0ESGxRWl', dest_path='/tmp/Y_train.npy', unzip=False)

    return np.load('/tmp/X_img_train.npy'), np.load('/tmp/X_feat_train.npy'), np.load(
        '/tmp/Y_train.npy'), np.load('/tmp/X_img_test.npy'), np.load('/tmp/X_feat_test.npy'), np.load('/tmp/Y_test.npy')


def main(n_epochs=100000, plot=True):
    X_img_train, X_feat_train, Y_train, X_img_test, X_feat_test, Y_test = load_data()

    _, n_features = X_feat_train.shape

    X = tf.placeholder(tf.float32, shape=(None, n_features))
    Y = tf.placeholder(tf.float32, shape=(None,))

    model = LogisticRegressionModel(X, Y, n_features, learning_rate=0.45)

    writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

    with tf.Session() as sess, tf.device('/gpu:0'):
        writer = tf.summary.FileWriter('./graphs', sess.graph)

        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        sess.run(tf.global_variables_initializer())

        for i in range(0, n_epochs):
            sess.run(model.optimize, feed_dict={X: X_feat_train, Y: Y_train})

            if i % 1000 == 0:
                train_loss, train_acc = sess.run(
                    [model.loss, model.accuracy],
                    feed_dict={X: X_feat_train, Y: Y_train}
                )
                test_loss, test_acc = sess.run(
                    [model.loss, model.accuracy],
                    feed_dict={X: X_feat_test, Y: Y_test}
                )
                Y_test_pred = sess.run(
                    model.prediction_round,
                    feed_dict={X: X_feat_test, Y: Y_test}
                )

                print("Training Loss Epoch {} (CE) : {:0.2f}; Training acc. : {:0.2f}; Test Loss (CE) : {:0.2f} Test acc. : {:0.2f}; ".format(
                    int(i / 1000) + 1, train_loss, train_acc, test_loss, test_acc))

    writer.close()

    print("FINAL TEST ACCURACY RESULT {}".format(test_acc))

    if(plot):
        plot_result(X_img_test, Y_test, Y_test_pred)

    return Y_test_pred, test_acc


def plot_result(X_img_test, Y_test, Y_test_pred):
    labels = ['Non people', 'People']
    num_row, num_col = 2, 6
    f, subplots = plt.subplots(num_row, num_col, sharex='col', sharey='row')

    for i in range(num_row):
        for j in range(num_col):
            idx = np.random.choice(np.arange(0, X_img_test.shape[0]))
            subplots[i, j].imshow(X_img_test[idx], cmap='gray',
                                  interpolation='nearest', aspect='auto')
            title = 'GT: {} \n Pred: {}'.format(
                labels[int(Y_test[idx])], labels[int(Y_test_pred[idx])])
            color_title = 'green' if int(Y_test[idx]) == int(
                Y_test_pred[idx]) else 'red'
            subplots[i, j].set_title(
                title, color=color_title, fontweight="bold")
            subplots[i, j].grid(b=False)

    f.set_size_inches(13.5, 7.5)
    plt.show()


if __name__ == '__main__':
    #Y_test_pred, test_acc = main(plot=False)
    eager_main()