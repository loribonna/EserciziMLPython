import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
from tensorflow.python import debug as tf_debug
import numpy as np
from mlp_class import MultiLayerPerceptron
import os
from datetime import datetime
now = datetime.now()
logdir = "graphs/" + now.strftime("%Y%m%d-%H%M%S") + "/"

def main(restore_save=False):
    """
    Placeholders for input 
    NB images are expressed in terms of vector and not matrices. 
    """
    x = tf.placeholder(tf.float32, shape=[None, 784])

    # Placeholder for targets
    targets = tf.placeholder(tf.float32, shape=[None, 10])

    # Placeholder for discerning train/eval mode
    is_training = tf.placeholder(dtype=tf.bool)

    # Define global step to indicize checkpoint saves
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")

    mnist = input_data.read_data_sets('/tmp/mnist', one_hot=True)

    model = MultiLayerPerceptron([256, 64, 10], x, targets, is_training, step=global_step, learning_rate=0.0000975)
    init_op = tf.global_variables_initializer()

    writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    saver = tf.train.Saver(model.get_vars_to_save())
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(logdir, sess.graph)

        if(restore_save):
           # check if there is checkpoint
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))

            # check if there is a valid checkpoint path
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Checkpoint restored: {}".format(ckpt.model_checkpoint_path))

        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        # Initialize all variables
        sess.run(init_op)

        # Training parameters
        training_epochs = 50
        batch_size = 128

        # Number of batches to process to see whole dataset
        batches_each_epoch = mnist.train.num_examples // batch_size

        for epoch in range(training_epochs):

            # During training measure accuracy on validation set to have an idea of what's happening
            val_accuracy = sess.run(fetches=model.accuracy,
                                    feed_dict={x: mnist.validation.images,
                                               targets: mnist.validation.labels,
                                               is_training: False})
            
            saver.save(sess, 
                'checkpoints/mlp.ckpt',
                global_step=global_step)

            print(
                'Epoch: {:06d} - VAL accuracy: {:.03f} %'.format(epoch, val_accuracy * 100))

            for index in range(batches_each_epoch):

                # Load a batch of training data
                x_batch, target_batch = mnist.train.next_batch(batch_size)

                # Actually run one training step here
                _, summary = sess.run(fetches=[model.optimize, model.summary],
                         feed_dict={x: x_batch, targets: target_batch, is_training: True})
                writer.add_summary(summary, global_step=index * (epoch + 1))

        test_accuracy = sess.run(fetches=model.accuracy,
                                 feed_dict={x: mnist.test.images,
                                            targets: mnist.test.labels,
                                            is_training: False})
        print('*' * 50)
        print('Training ended. TEST accuracy: {:.03f} %'.format(
            test_accuracy * 100))
    writer.close()


if __name__ == "__main__":
    main()
