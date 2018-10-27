"""
Class that models a Naive Bayes Classifier
"""

import numpy as np


class NaiveBayesClassifier:
    """
    Naive Bayes Classifier.
    Training:
    For each class, a naive likelyhood model is estimated for P(X/Y),
    and the prior probability P(Y) is computed.
    Inference:
    performed according with the Bayes rule:
    P = argmax_Y (P(X/Y) * P(Y))
    or
    P = argmax_Y (log(P(X/Y)) + log(P(Y)))
    """

    def __init__(self):
        """
        Class constructor
        """

        self._classes = None
        self._n_classes = 0

        self._eps = np.finfo(np.float32).eps

        # array of classes prior probabilities
        self._class_priors = []

        # array of probabilities of a pixel being active (for each class)
        self._pixel_probs_given_class = []

    def fit(self, X, Y):
        """
        Computes, for each class, a naive likelyhood model (self._pixel_probs_given_class),
        and a prior probability (self.class_priors).
        Both quantities are estimated from examples X and Y.

        Parameters
        ----------
        X: np.array
            input MNIST digits. Has shape (n_train_samples, h, w)
        Y: np.array
            labels for MNIST digits. Has shape (n_train_samples,)
        """

        n_train_samples, h, w = X.shape

        self._classes = sorted(np.unique(Y))
        self._n_classes = len(self._classes)

        # compute prior and pixel probabilities for each class
        for c in self._classes:
            c_label = Y[Y == self._classes[c]]
            c_img = X[Y == self._classes[c]]

            # prior probability
            self._class_priors.insert(c, len(c_label) / len(Y))
            # estimate naive pixel likelihoods
            self._pixel_probs_given_class.insert(
                c, np.average(c_img, axis=0)
            )

    def predict(self, X):
        """
        Performs inference on test data.
        Inference is performed according with the Bayes rule:
        P = argmax_Y (log(P(X/Y)) + log(P(Y)) - log(P(X)))

        Parameters
        ----------
        X: np.array
            MNIST test images. Has shape (n_test_samples, h, w).

        Returns
        -------
        prediction: np.array
            model predictions over X. Has shape (n_test_samples,)
        """

        n_test_samples, h, w = X.shape

        # initialize log probabilities of class
        class_log_probs = np.zeros(shape=(n_test_samples, self._n_classes))

        for c in range(0, self._n_classes):
            # extract class models
            pass

            # prior probability of this class
            cur_prior = self._class_priors[c]
            # likelyhood of examples given class
            cur_prob = self._pixel_probs_given_class[c]
            # bayes rule for logarithm
            l_hoods = np.ma.log(X) + np.ma.log(cur_prob)
            l_hoods = l_hoods.filled(0)
            # set class probability for each test example
            class_log_probs[:, c] = np.sum(np.sum(l_hoods, axis=1), axis=1)
        # predictions
        predictions = class_log_probs.argmax(axis=1)

        return predictions

    @staticmethod
    def _estimate_pixel_probabilities(images):
        """
        Estimates pixel probabilities from data.

        Parameters
        ----------
        images: np.array
            images to estimate pixel probabilities from. Has shape (n_images, h, w)

        Returns
        -------
        pix_probs: np.array
            probabilities for each pixel of being 1, estimated from images.
            Has shape (h, w)
        """

        return

    def _get_log_likelyhood_under_model(self, images, model):
        """
        Returns the likelyhood of many images under a certain model.
        Naive:
        the likelyhood of the image is the product of the likelyhood of each pixel.
        or
        the log-likelyhood of the image is the sum of the log-likelyhood of each pixel.

        Parameters
        ----------
        images: np.array
            input images. Having shape (n_images, h, w).
        model: np.array
            a model of pixel probabilities, having shape (h, w)

        Returns
        -------
        lkl: np.array
            the likelyhood of each pixel under the model, having shape (h, w).
        """

        return
