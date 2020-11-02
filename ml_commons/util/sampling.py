def stratified_select_indices(labels, select):
    """
    Parameters
    ----------
    labels : array-like
        The target variable for supervised learning problems.
        Stratification is done based on the labels.

    select : float or int
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the indices to be used. If int, represents
        the absolute number of indices.

    Returns
    ------
    selected : list
        Selected indices.
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, train_size=select)
    selected, unselected = list(sss.split([0] * len(labels), labels))[0]
    return list(selected)
