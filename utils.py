import numpy as np
from scipy.stats import pearsonr


def correlation(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]