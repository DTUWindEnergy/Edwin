# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:58:11 2020

@author: juru
"""
import numpy as np
import scipy.io


def crossover(x1, x2):
    """
    This is the uniform cross-over
    """
    """
    if test_mode:
        mat = scipy.io.loadmat('../alpha.mat')
        alpha = mat['alpha_all'][k]
    else:
        if not seed==None:
            np.random.seed(seed)
    """
    alpha = np.random.randint(0, 2, x1.size).ravel().astype(bool)
    y1 = alpha * x1 + (1 - alpha) * x2
    y2 = alpha * x2 + (1 - alpha) * x1
    y1 = y1.astype(bool)
    y2 = y2.astype(bool)
    return y1, y2


if __name__ == '__main__':
    x1 = np.array([False, False, True, False, False, False, False, True, False,
                   True, False, False, True, True, False, True, True, True,
                   True, True, True, False, False, False, True, False, True,
                   False, False, False, False, False, True, False, False, False,
                   False, False, True, True, False, False, False, True, True,
                   False, False, True, False, False, False, True, True, False,
                   True, False, True, False, True, False, True, False, False,
                   False, False, False, False, True, False, False, True, False,
                   True, False, False, True, False, True, False, False, False,
                   False, True, False, False, True, True])
    x2 = np.array([True, True, False, False, True, True, False, True, True,
                   True, False, True, False, False, False, False, True, False,
                   True, False, False, True, False, True, False, False, True,
                   True, True, False, False, False, True, True, False, True,
                   False, True, False, False, True, False, False, False, False,
                   False, True, True, True, True, False, True, False, True,
                   True, True, True, False, False, True, True, True, True,
                   True, True, False, False, True, True, False, True, False,
                   True, False, False, True, True, True, True, True, True,
                   False, True, False, True, True, True])

    print(crossover(x1, x2))
