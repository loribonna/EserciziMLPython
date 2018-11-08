import numpy as np
import matplotlib.pyplot as plt
from datasets import h_shaped_dataset
from datasets import two_moon_dataset
from datasets import gaussians_dataset
from utils import plot_boundary
from utils import cmap
from utils import plot_2d_dataset

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

        self.possible_labels = None

    def fit(self, X, Y, verbose=False):
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

            # extract 'difficult' samples

            # search for a weak classifier
            error = 1
            n_trials = 0
            while error > 0.5:

                # select random feature (np.random.choice)

                # select random split (np.random.uniform)

                # select random verse (np.random.choice)

                # compute assignment

                # compute error

                n_trials += 1
                if n_trials > 100:
                    # initialize the sample weights again
                    sample_weights = np.ones(shape=n) / n

            # save weak learner parameter
            # self.alphas[l] = ...
            # self.dims[l] = ...
            # self.splits[l] = ...
            # self.label_above_split[l] = ...

            # update sample weights


    def predict(self, X):
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

        for l, cur_dim, cur_split, label_above_split in zip(range(0, self.n_learners), self.dims, self.splits,
                                                            self.label_above_split):

            label_below_split = -label_above_split

            # compute assignment

            # Save assignment of lth weak learner

        # weight for learners efficiency

        # compute predictions

        pred = np.random.choice([-1, 1], size=len(X))  # todo remove me
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
    print(error)

# entry point
if __name__ == '__main__':
    main_adaboost()
