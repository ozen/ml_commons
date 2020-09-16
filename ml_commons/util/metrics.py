import numpy as np
from sklearn.metrics import average_precision_score, roc_curve


def mean_average_precision(correct):
    """
    :param correct: Boolean ndarray of shape (n_queries, n_retrieved) in which
    True value corresponds to a relevant retrieval. Order is important.
    :return: Mean Average Precision of the queries.
    """
    return np.mean([average_precision_score(c, np.arange(len(c), 0, -1)) if np.any(c) else 0 for c in correct])


def tpr_at_tnr(probs, labels, at_tnr):
    import torch
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    probs = probs.flatten()
    labels = labels.flatten()

    fpr, tpr, _ = roc_curve(labels, probs)
    tnr = 1 - fpr   # in decreasing order
    for idx in range(1, len(tnr)):
        if tnr[idx] < at_tnr:
            return tpr[idx-1]
    return 0
