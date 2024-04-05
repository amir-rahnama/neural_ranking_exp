import numpy as np
from scipy.stats import rankdata

def iter_overlap(list1, list2):
    """Calculates the % of overlap for two equal sized lists of ranked values in incremental sets
    of five, from 0 to number features

    Args:
        list1: list of ranked values (integers)
        list2: list of ranked values (integers). Same length as list1
    Returns:
       List of floats
    """
    list_overlap = []
    for x in range(5, len(_name_features) + 5, 5):
        included_values = [i for i in range(x)]
        indices1 = np.where(np.isin(list1, included_values))[0]
        indices2 = np.where(np.isin(list2, included_values))[0]
        inter = np.size(np.intersect1d(indices1, indices2))
        list_overlap.append(inter / x)
    return list_overlap
def rbo(list1, list2, p=0.9):
    """Calculates the rbo score for two equal sized ranks

    Args:
        list1: list of ranked values (integers)
        list2: list of ranked values (integers). Same length as list1
        p: weight assigned to items at different positions in the ranked lists, ranges [0,1]
    Returns:
       rbo (float)
    """
    # tail recursive helper function
    def helper(ret, i, d):
        l1 = set(list1[:i]) if i < len(list1) else set(list1)
        l2 = set(list2[:i]) if i < len(list2) else set(list2)
        a_d = len(l1.intersection(l2)) / i
        term = math.pow(p, i) * a_d
        if d == i:
            return ret + term
        return helper(ret + term, i + 1, d)

    k = max(len(list1), len(list2))
    x_k = len(set(list1).intersection(set(list2)))
    summation = helper(0, 1, k)

    return ((float(x_k) / k) * math.pow(p, k)) + ((1 - p) / p * summation)


def calculate_metrics(gam_explanations, lime_explanations, k_top):
    """Calculates the rbo and the overlap metrics for two set of explanations

    Args:
        gam_explanations: ndarray of subscores from GAM model
        lime_explanations: ndarray of subscores from LIME
    Returns:
       rbo (float) and overlap list (floats)
    """
    flatten_GAM = gam_explanations.numpy().flatten()
    flatten_LIME = lime_explanations.numpy().flatten()

    ranked_GAM = rankdata([-1 * i for i in flatten_GAM]).astype(int) - 1
    ranked_LIME = rankdata([-1 * i for i in flatten_LIME]).astype(int) - 1
    rbo_metric = rbo(ranked_GAM, ranked_LIME, k_top)
    overlap_metric = iter_overlap(ranked_GAM, ranked_LIME)
    return rbo_metric, overlap_metric
    