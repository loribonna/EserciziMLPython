import numpy as np
from datasets import two_moon_dataset, gaussians_dataset
from numpy.linalg import eigh
from scipy.linalg import fractional_matrix_power
from sklearn.cluster import KMeans
import skimage.io
import matplotlib.pyplot as plt
import skimage.transform

# from kmeans_clustering import kmeans
from numpy import ndarray

import matplotlib.pyplot as plt

plt.ion()

class SpectralClustering:
    _sim_t = 0.3

    def __init__(self, temperature=0.06):
        self._sim_t = temperature

    def similarity_function(self, a: ndarray, b: ndarray) -> float:
        distance: float = np.sum(np.square(a - b))
        return np.exp(-(distance / self._sim_t))

    def spectral_clustering(self, data: ndarray, n_cl: int, sigma=1.):
        """
        Spectral clustering.

        Parameters
        ----------
        data: ndarray
            data to partition, has shape (n_samples, dimensionality).
        n_cl: int
            number of clusters.
        sigma: float
            std of radial basis function kernel.

        Returns
        -------
        ndarray
            computed assignment. Has shape (n_samples,)
        """
        n_samples, dim = data.shape

        # Compute Affinity Matrix
        affinity_matrix = np.zeros((n_samples, n_samples))

        for i in range(0, n_samples):
            # print('{0} out of {1}'.format(i,n_samples))
            for j in range(i + 1, n_samples):
                affinity_matrix[i, j] = self.similarity_function(
                    data[i], data[j])

        # fix the matrix
        affinity_matrix += np.transpose(affinity_matrix)

        # Degree matrix
        degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))

        # DO LAPLACIAN
        laplacian = degree_matrix - affinity_matrix

        # Compute eigenvalues and vectors
        eig_vals, eig_vects = np.linalg.eig(laplacian)

        labels = np.zeros(n_samples)

        # set labels
        labels[eig_vects[:, 1] > 0] = 1

        # use Kmeans
        labels = KMeans(n_cl).fit((eig_vects[:, 0:n_cl])).labels_
        return labels


def main_spectral_clustering():
    """
    Main function for spectral clustering.
    """

    # generate the dataset
    data, cl = two_moon_dataset(n_samples=300, noise=0.1)

    # visualize the dataset
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(data[:, 0], data[:, 1], c=cl, s=40)
    # plt.waitforbuttonpress()

    # run spectral clustering
    labels = SpectralClustering().spectral_clustering(data, n_cl=2, sigma=0.1)

    # visualize results
    ax[1].scatter(data[:, 0], data[:, 1], c=labels, s=40)
    plt.waitforbuttonpress()


def main_spectral_clustering_image():
    """
    Main function for spectral clustering.
    """
    num_cluster = 2
    # generate the dataset
    Img = skimage.io.imread('./img/minions.jpg')
    Img = skimage.transform.rescale(Img, 0.01, preserve_range=True)
    w, h, c = Img.shape

    # prepare data
    colors = np.reshape(Img, (w * h, c))
    colors /= 255.0

    # #add ij coordinates
    #
    # colors = np.zeros((w*h,5))
    #
    # count=0
    # for i in range (0,w):
    #     for j in range(0,h):
    #         colors[count,:]=np.hstack((Img[i,j]/255.0, np.float(i)/w,np.float(j)/h))
    #         count+=1

    # visualize the dataset
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(Img)
    # plt.waitforbuttonpress()

    # run spectral clustering
    labels = SpectralClustering().spectral_clustering(colors, n_cl=num_cluster, sigma=0.1)

    # visualize results
    imglabels = np.reshape(np.float32(labels) * (255.0 / num_cluster), (w, h))

    ax[1].imshow(np.uint8(imglabels))
    plt.waitforbuttonpress()

if __name__ == '__main__':
    main_spectral_clustering()
