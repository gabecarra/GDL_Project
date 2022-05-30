import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import seaborn as sns
from fastdtw import fastdtw
from sklearn.preprocessing import normalize
from numba import jit


def correntropy_PCA(A, s=5):
    """
    https://github.com/phuijse/correntropy_demos/blob/master/correntropy_pca_demo.ipynb
    :param A: matrix (N, M)
    :param s:
    :return: correlation matrix (N, N)
    """
    A = normalize(A, axis=1, norm='l1')
    res = np.empty((A.shape[0], A.shape[0]))
    for i, elem1 in enumerate(A):
        for j, elem2 in enumerate(A):
            N = len(elem1)
            V = np.average(np.exp(-0.5 * (elem1 - elem2) ** 2 / s ** 2))
            # mean in feature space should be subtracted!!
            CIP = 0.0
            for k in range(0, N):
                CIP += np.average(
                    np.exp(-0.5 * (elem1 - elem2[k]) ** 2 / s ** 2)) / N
            res[i, j] = V - CIP
    return res


def granger_causality(A, maxlag=3):
    """
    https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.grangercausalitytests.html
    :param A: matrix (N, M)
    :param maxlag: number of lag variables
    :return: correlation matrix (N, N)
    """
    res = np.empty((A.shape[0], A.shape[0]))
    for i, elem1 in enumerate(A):
        for j, elem2 in enumerate(A):
            test_res = grangercausalitytests(
                np.vstack((elem1, elem2)).T,
                maxlag=[maxlag],
                verbose=False
            )
            res[i, j] = test_res[maxlag][0]['ssr_ftest'][1]
    return res


@jit(nopython=True, fastmath=True)
def cosine_similarity(A):
    """
    https://en.wikipedia.org/wiki/Cosine_similarity
    :param A: matrix (N, M)
    :return: correlation matrix (N, N)
    """
    res = np.empty((A.shape[0], A.shape[0]))
    for i, elem1 in enumerate(A):
        for j, elem2 in enumerate(A):
            res[i, j] = \
                (np.ascontiguousarray(elem1) @ np.ascontiguousarray(elem2)) / \
                (np.linalg.norm(elem1) * np.linalg.norm(elem2))
    return res


def dynamic_time_warping(A):
    """
    # https://towardsdatascience.com/dynamic-time-warping-3933f25fcdd
    # https://en.wikipedia.org/wiki/Dynamic_time_warping
    :param A:
    :return:
    """
    res = np.empty((A.shape[0], A.shape[0]))
    for i, elem1 in enumerate(A):
        for j, elem2 in enumerate(A):
            distance, _ = fastdtw(elem1, elem2, dist=euclidean)
            res[i, j] = distance
    # scale to be between 0 and 1
    res = np.abs(res - np.max(res)) / (np.max(res) - np.min(res))
    return res


def dummy_correlation(A):
    return np.ones((A.shape[0], A.shape[0]))


if __name__ == '__main__':
    matrix = np.random.randint(1, 30, (50, 10, 14)).astype(float)
    plt.figure(1)
    ax = sns.heatmap(dummy_correlation(matrix[0]), cmap='viridis',
                     linewidth=0.5)
    plt.show()

    plt.figure(2)
    ax2 = sns.heatmap(granger_causality(matrix[0]), cmap='viridis', linewidth=0.5)
    plt.show()
    pass
