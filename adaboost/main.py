import numpy as np
import matplotlib.pyplot as plt
from datasets import h_shaped_dataset
from datasets import two_moon_dataset
from datasets import gaussians_dataset
from utils import plot_boundary
from utils import cmap
from utils import plot_2d_dataset
from numpy import ndarray

plt.ion()


class AdaBoostClassifier:
    """
    Function that models a Adaboost classifier
    """

    def __init__(self, n_learners):
        """
        Model constructor

        Parameters
        ----------
        n_learners: int
            number of weak classifiers.
        """

        # initialize a few stuff
        self.n_learners = n_learners
        self.alphas = np.zeros(shape=n_learners)
        self.dims = np.zeros(shape=n_learners, dtype=np.int32)
        self.splits = np.zeros(shape=n_learners)
        self.label_above_split = np.zeros(shape=n_learners, dtype=np.int32)
        self.label_below_split = np.zeros(shape=n_learners, dtype=np.int32)
        self.possible_labels = None

    def fit(self, X: ndarray, Y: ndarray, verbose=False):
        """
        Trains the model.

        Parameters
        ----------
        X: ndarray
            features having shape (n_samples, dim).
        Y: ndarray
            class labels having shape (n_samples,).
        verbose: bool
            whether or not to visualize the learning process.
            Default is False
        """

        n, d = X.shape

        self.possible_labels = np.unique(Y)

        # only binary problems please
        assert self.possible_labels.size == 2, 'Error: data is not binary'

        # initialize the sample weights as equally probable
        sample_weights = np.ones(shape=n) / n

        # start training
        for l in range(0, self.n_learners):
            # choose the indexes of 'difficult' samples (np.random.choice)
            indexes = np.random.choice(np.arange(0, n), size=n, p=sample_weights)
            # extract 'difficult' samples
            samples = X[indexes]
            # search for a weak classifier
            error = 1
            n_trials = 0
            while error > 0.5:
                # select random feature (np.random.choice)
                f_index = np.random.choice(np.arange(0, d))
                f_set: ndarray = samples[:, f_index]
                w_set: ndarray = sample_weights[indexes]
                y_set: ndarray = Y[indexes]
                # select random split (np.random.uniform)
                threshold = np.random.uniform(f_set.min(), f_set.max())
                # select random verse (np.random.choice)
                v = self.possible_labels
                np.random.shuffle(v)
                l_above = v[1]
                l_below = v[0]
                # compute assignment
                prediction = np.zeros(shape=n)
                prediction[f_set <= threshold] = v[0]
                prediction[f_set > threshold] = v[1]
                # compute error
                error = np.sum(w_set[prediction != y_set])

                n_trials += 1
                if n_trials > 100:
                    # initialize the sample weights again
                    sample_weights = np.ones(shape=n) / n

            # save weak learner parameter
            # self.alphas[l] = ...
            # self.dims[l] = ...
            # self.splits[l] = ...
            # self.label_above_split[l] = ...
            alpha = 0.5 * np.log((1 - error) / error)
            self.alphas[l] = alpha if error > 0 else 100
            self.dims[l] = f_index
            self.label_above_split[l] = l_above
            self.label_below_split[l] = l_below
            self.splits[l] = threshold

            # update sample weights

            for i in range(n):
                index = indexes[i]
                a = -self.alphas[l] if prediction[i] == y_set[i] else self.alphas[l]
                sample_weights[index] = sample_weights[index] * np.exp(a)

            z = 1 / np.sum(sample_weights)
            sample_weights *= z

            if (verbose):
                print('Classifier {}: error {}, alpha {}, dim {}, split {}, label above {}, Z {}'.format(
                    l + 1, error, alpha, f_index, threshold, l_above, z
                ))

    def predict(self, X: ndarray):
        """
        Function to perform predictions over a set of samples.
        
        Parameters
        ----------
        X: ndarray
            examples to predict. shape: (n_examples, d).

        Returns
        -------
        ndarray
            labels for each examples. shape: (n_examples,).

        """
        n, d = X.shape

        pred_all_learners = np.zeros(shape=(n, self.n_learners))

        for l, cur_dim, cur_split, label_above_split, label_below_split in \
                zip(range(0, self.n_learners), self.dims, self.splits,
                    self.label_above_split, self.label_below_split):
            # compute assignment
            p = np.zeros(shape=n)
            p[X[:, cur_dim] > cur_split] = label_above_split
            p[X[:, cur_dim] <= cur_split] = label_below_split
            # Save assignment of lth weak learner
            pred_all_learners[:, l] = self.alphas[l] * p
        # weight for learners efficiency

        # compute predictions
        pred_learners = np.sum(pred_all_learners, axis=1)
        pred = np.sign(pred_learners)
        return pred


def main_adaboost():
    """
    Main function for testing Adaboost.
    """

    X_train, Y_train, X_test, Y_test = h_shaped_dataset()
    # X_train, Y_train, X_test, Y_test = gaussians_dataset(2, [100, 150], [[1, 3], [-4, 8]], [[2, 3], [4, 1]])
    # X_train, Y_train, X_test, Y_test = two_moon_dataset(n_samples=300, noise=0.2)

    # visualize dataset
    plot_2d_dataset(X_train, Y_train, 'Training')

    # train model and predict
    model = AdaBoostClassifier(n_learners=100)
    model.fit(X_train, Y_train, verbose=True)
    P = model.predict(X_test)

    # visualize the boundary!
    plot_boundary(X_train, Y_train, model)

    # evaluate and print error
    error = float(np.sum(P != Y_test)) / Y_test.size
    print('Error: {}, Accuracy (1-Error): {}'.format(error, 1 - error))


# entry point
if __name__ == '__main__':
    main_adaboost()
