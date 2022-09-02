import seaborn as sns
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from tsl.ops.connectivity import adj_to_edge_index
from statsmodels.tsa.stattools import grangercausalitytests
from numba import jit


class Correlation:
    def __init__(self, dataset):
        self.dataset = dataset
        df = dataset.dataframe()
        df = df.droplevel(1, axis=1).T
        self.data = df.to_numpy()
        self.n_nodes = self.data.shape[0]

    def get_data(self):
        return self.data

    def _get_similarity(self, method):
        if method == 'random':
            return np.random.rand(self.n_nodes, self.n_nodes)
        elif method == 'full':
            return np.ones((self.n_nodes, self.n_nodes))
        elif method == 'identity':
            return np.eye(self.n_nodes)
        elif method == 'pearson':
            return np.corrcoef(self.data)
        elif method == 'cosine':
            return _cosine_similarity(self.data)
        elif method == 'granger':
            return _granger_causality(self.data)
        elif method == 'dtw':
            return _dtw(self.data)
        else:
            raise NotImplementedError
        pass

    def get_correlation_methods(self):
        return ['random', 'full', 'identity', 'pearson', 'cosine', 'granger', 'dtw']

    def get_correlation(self,
                        method,
                        threshold=None,
                        include_self=None,
                        normalize_axis=None,
                        layout='dense'
                        ):
        adj = self._get_similarity(method)
        if threshold is not None:
            adj[adj < threshold] = 0
        if not include_self:
            np.fill_diagonal(adj, 0)
        if normalize_axis:
            adj = np.abs(adj - np.min(adj)) / (np.max(adj) - np.min(adj))
        if layout == 'dense':
            return adj
        elif layout == 'edge_index':
            return adj_to_edge_index(adj)


# correlation methods
@jit(nopython=True, fastmath=True)
def _cosine_similarity(A):
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


def _granger_causality(A, maxlag=2):
    """
    https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.grangercausalitytests.html
    :param A: matrix (N, M)
    :param maxlag: number of lag variables
    :return: correlation matrix (N, N)
    """
    res = np.empty((A.shape[0], A.shape[0]))
    for i, elem1 in tqdm(enumerate(A), total=len(A)):
        for j, elem2 in enumerate(A):
            test_res = grangercausalitytests(
                np.vstack((elem1, elem2)).T,
                maxlag=[maxlag],
                verbose=False
            )
            res[i, j] = test_res[maxlag][0]['ssr_ftest'][1]
    return res

def _dtw(A, window=3):
    """
    https://towardsdatascience.com/dynamic-time-warping-3933f25fcdd
    https://en.wikipedia.org/wiki/Dynamic_time_warping
    :param A: matrix (N, M)
    :param window:
    :return:
    """
    res = np.empty((A.shape[0], A.shape[0]))
    for i, elem1 in tqdm(enumerate(A), total=len(A)):
        for j, elem2 in enumerate(A):
            distance, path = fastdtw(elem1, elem2, dist=euclidean)
            res[i, j] = distance
    return res
