import numpy as np
from sklearn.metrics import average_precision_score


def mean_average_precision(correct):
    """
    :param correct: Boolean ndarray of shape (n_queries, n_retrieved) in which
    True value corresponds to a relevant retrieval. Order is important.
    :return: Mean Average Precision of the queries.
    """
    return np.mean([average_precision_score(c, np.arange(len(c), 0, -1)) if np.any(c) else 0 for c in correct])
