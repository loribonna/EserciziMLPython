import tensorflow as tf
from tensorflow import Session
from matplotlib import pyplot as plt
import numpy as np
from google_drive_downloader import GoogleDriveDownloader as gdd
from logistic_class import LogisticRegressionModel


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


def main(n_epochs=100000, plot=True, batch_size=128):
    X_img_train, X_feat_train, Y_train, X_img_test, X_feat_test, Y_test = load_data()

    _, n_features = X_feat_train.shape

    # Create dataset from data
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (X_feat_train, Y_train)).batch(batch_size)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_feat_test, Y_test))
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.cache()

    # Create from structure to model graph structure for both test and train
    iterator = tf.data.Iterator.from_structure(
        train_dataset.output_types, train_dataset.output_shapes)

    # Initilize train and test iterators with data
    train_initializer = iterator.make_initializer(train_dataset)
    test_initializer = iterator.make_initializer(test_dataset)

    X, Y = iterator.get_next()

    model = LogisticRegressionModel(
        X, Y, n_features, learning_rate=0.01)

    writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

    with tf.Session() as sess, tf.device('/gpu:0'):
        writer = tf.summary.FileWriter('./graphs', sess.graph)

        # Debug
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        
        sess.run(tf.global_variables_initializer())
        sess.run(train_initializer)

        for i in range(0, n_epochs):
            try:
                _ = sess.run(model.optimize)
            except tf.errors.OutOfRangeError:
                sess.run(train_initializer)
                pass

            if i % 1000 == 0:
                try:
                    train_acc, train_loss = sess.run(
                        [model.accuracy, model.loss])
                except tf.errors.OutOfRangeError:
                    sess.run(train_initializer)
                    train_acc, train_loss = sess.run(
                        [model.accuracy, model.loss])
                print("Training Loss {} at Epoch {} with accuracy {}".format(
                    train_loss, int(i / 1000), train_acc))


        sess.run(test_initializer)
        test_acc_mean = 0
        test_loss_mean = 0
        n_batches = 0
        test_pred = []
        try:
            while(True):
                test_loss, test_acc = sess.run([model.loss, model.accuracy])
                test_pred.append(sess.run(model.prediction_round))
                test_acc_mean += test_acc
                test_loss_mean += test_loss
                n_batches += 1
        except tf.errors.OutOfRangeError:
            test_acc_mean /= n_batches
            test_loss_mean /= n_batches
            print("Test Loss (CE) : {:0.2f} Test acc. : {:0.2f}; ".format(
                test_loss_mean, test_acc_mean))

    writer.close()

    print("FINAL TEST ACCURACY RESULT {}".format(test_acc))

    if(plot):
        plot_result(X_img_test, Y_test, Y_test_pred)

    return test_pred, test_acc_mean


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
    Y_test_pred, test_acc = main(plot=False)