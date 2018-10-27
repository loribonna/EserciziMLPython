"""
Eigenfaces main script.
"""

import numpy as np

from utils import show_eigenfaces, show_3d_faces_with_class
from data_io import get_faces_dataset

from numpy import ndarray

import matplotlib.pyplot as plt
plt.ion()

def eigenfaces(X: ndarray, n_comp):
    """
    Performs PCA to project faces in a reduced space.

    Parameters
    ----------
    X: ndarray
        faces to project (shape: (n_samples, w*h))
    n_comp: int
        number of principal components

    Returns
    -------
    tuple
        proj_faces: the projected faces shape=(n_samples, n_comp).
        ef: eigenfaces (the principal directions)
    """

    n_samples, dim = X.shape

    # compute mean vector
    mean_v:ndarray=X.mean(axis=0)
    # show mean face
#    plt.imshow(np.reshape(mean_v,newshape=(112, 92)))
    # normalize data (remove mean)
    data_norm:ndarray=X-mean_v
    # trick (transpose data matrix)
    data_transpose=data_norm.transpose()
    # compute covariance
    cov:ndarray=data_norm.dot(data_transpose)
    # compute (sorted) eigenvectors of the covariance matrix
    eig_v, eig_V=np.linalg.eig(cov)
    eig_index_sorted=eig_v.argsort()[::-1][:n_comp]
    eig_V_sorted=eig_V[:,eig_index_sorted]
    # retrieve original eigenvec
    eig_V_org:ndarray=data_transpose.dot(eig_V_sorted)
    # show eigenfaces
#    show_eigenfaces(eig_V_org,size=(112, 92))
    # project faces according to the computed directions
    data_project=data_norm.dot(eig_V_org)
    return data_project, eig_V_org


def main():
    """
    Main function.
    """

    # number of principal components
    n_comp = 10

    # get_data
    X_train, Y_train, X_test, Y_test = get_faces_dataset(path='att_faces')

    proj_train, ef = eigenfaces(X_train, n_comp=n_comp)

    # visualize projections if 3d
    if n_comp == 3:
        show_3d_faces_with_class(proj_train, Y_train)

    # project test data
    test_proj = np.dot(X_test, ef)

    # predict test faces
    predictions = np.zeros_like(Y_test)
    nearest_neighbors = np.zeros_like(Y_test, dtype=np.int32)
    for i in range(0, test_proj.shape[0]):

        cur_test = test_proj[i]

        # compute distances w.r.t every training face

        # nearest neighbor classification


    print('Error: {}'.format(float(np.sum(predictions != Y_test))/len(predictions)))

    # visualize nearest neighbors
    _, (ax0, ax1) = plt.subplots(1, 2)
    while True:

        # extract random index
        test_idx = np.random.randint(0, X_test.shape[0])

        ax0.imshow(np.reshape(X_test[test_idx], newshape=(112, 92)), cmap='gray')
        ax0.set_title('Test face')
        ax1.imshow(np.reshape(X_train[nearest_neighbors[test_idx]], newshape=(112, 92)), cmap='gray')
        ax1.set_title('Nearest neighbor')

        # plot faces
        plt.waitforbuttonpress()


if __name__ == '__main__':
    main()
