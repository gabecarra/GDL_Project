import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.spatial.distance import euclidean
from numpy.linalg import norm
import matplotlib.pyplot as plt
import seaborn as sns
from fastdtw import fastdtw
from sklearn.preprocessing import normalize
from numba import jit
from spektral.utils.convolution import gcn_filter
from utility.constant import VERBOSE

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


def granger_causality(A, maxlag=1):
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
    res = np.abs(res - np.min(res)) / (np.max(res) - np.min(res))
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
    res = np.abs(res - np.min(res)) / (np.max(res) - np.min(res))
    return res


@jit(nopython=True, fastmath=True, nogil=True)
def dtw(A, window=3):
    """
    https://towardsdatascience.com/dynamic-time-warping-3933f25fcdd
    https://en.wikipedia.org/wiki/Dynamic_time_warping
    :param A:
    :param window:
    :return:
    """
    res = np.empty((A.shape[0], A.shape[0]))
    for i, elem1 in enumerate(A):
        for j, elem2 in enumerate(A):
            n, m = len(elem1), len(elem2)
            w = np.nanmax([window, abs(n - m)])
            dtw_matrix = np.zeros((n + 1, m + 1))

            for k in range(n + 1):
                for z in range(m + 1):
                    dtw_matrix[k, z] = np.inf
            dtw_matrix[0, 0] = 0

            for k in range(1, n + 1):
                for z in range(np.nanmax([1, k - w]), np.nanmin([m, k + w]) + 1):
                    dtw_matrix[k, z] = 0

            for k in range(1, n + 1):
                for z in range(np.nanmax([1, k - w]), np.nanmin([m, k + w]) + 1):
                    cost = abs(elem1[k - 1] - elem2[z - 1])
                    # take last min from a square box
                    last_min = np.nanmin(
                        [dtw_matrix[k - 1, z], dtw_matrix[k, z - 1],
                         dtw_matrix[k - 1, z - 1]])
                    dtw_matrix[k, z] = cost + last_min
            res[i, j] = dtw_matrix[-1, -1]
    res = 1 - np.abs(res - np.min(res)) / (np.max(res) - np.min(res))
    return res


def dummy_correlation(A):
    return np.ones((A.shape[0], A.shape[0]))


def pearson_correlation(A):
    res = np.corrcoef(A)
    res = np.abs(res - np.min(res)) / (np.max(res) - np.min(res))
    return res


if __name__ == '__main__':
    matrix = np.random.randint(1, 30, (50, 10, 14)).astype(float)
    plt.figure(1)
    ax = sns.heatmap(1 - np.abs(np.corrcoef(matrix[0])), cmap='viridis',
                     linewidth=0.5)
    plt.show()
    plt.figure(2)
    ax = sns.heatmap(np.corrcoef(matrix[0]), cmap='viridis',
                     linewidth=0.5)
    plt.show()

    # corr_functions = [
    #     pearson_correlation,
    #     dtw,
    #     cosine_similarity,
    #     granger_causality
    # ]
    #
    # for i, fn in enumerate(corr_functions):
    #     plt.figure(i)
    #     ax = sns.heatmap(fn(matrix[0]), cmap='viridis',
    #                      linewidth=0.5)
    #     plt.show()