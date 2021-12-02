# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:33:23 2020

@author: juru
"""
import numpy as np
import scipy.io


def roulette_wheel_selection(P):
    #    r = np.random.rand(1)
    #    r = float(input('Insert r: '))

    r = np.random.rand(1)
    c = np.cumsum(P)
    i = np.argwhere(r <= c)[0][0]
#    print(r)
#    print(i)
    return i


if __name__ == '__main__':
    P = np.array([1.08689101e-02, 1.08689101e-02, 1.08689100e-02, 1.08689100e-02,
                  1.08689098e-02, 1.08689097e-02, 1.08689097e-02, 1.08689097e-02,
                  1.08689097e-02, 1.08689097e-02, 1.08689097e-02, 1.08689096e-02,
                  1.08689096e-02, 1.08689096e-02, 1.08689096e-02, 1.08689096e-02,
                  1.08689096e-02, 1.08689095e-02, 1.08689095e-02, 1.08689095e-02,
                  1.08689094e-02, 1.08689094e-02, 1.08689094e-02, 1.08689094e-02,
                  1.08689094e-02, 1.08689094e-02, 1.08689094e-02, 1.08689094e-02,
                  1.08689093e-02, 1.08689093e-02, 1.08689092e-02, 1.08689092e-02,
                  1.08689092e-02, 1.08689092e-02, 1.08689092e-02, 1.08689092e-02,
                  1.08689092e-02, 1.08689092e-02, 1.08689092e-02, 1.08689092e-02,
                  1.08689091e-02, 1.08689091e-02, 1.08689091e-02, 1.08689091e-02,
                  1.08689091e-02, 1.08689090e-02, 1.08689090e-02, 1.08689090e-02,
                  1.08689090e-02, 1.08689089e-02, 1.08689089e-02, 1.08689089e-02,
                  1.08689089e-02, 1.08689089e-02, 1.08689089e-02, 1.08689089e-02,
                  1.08689089e-02, 1.08689089e-02, 1.08689089e-02, 1.08689089e-02,
                  1.08689088e-02, 1.08689088e-02, 1.08689088e-02, 1.08689088e-02,
                  1.08689088e-02, 1.08689088e-02, 1.08689088e-02, 1.08689088e-02,
                  1.08689088e-02, 1.08689088e-02, 1.08689088e-02, 1.08689088e-02,
                  1.08689088e-02, 1.08689088e-02, 1.08689087e-02, 1.08689087e-02,
                  1.08689087e-02, 1.08689087e-02, 1.08689087e-02, 1.08689087e-02,
                  1.08689086e-02, 1.08689086e-02, 1.08689086e-02, 1.08689086e-02,
                  1.08689086e-02, 1.08689085e-02, 1.08689085e-02, 1.08689085e-02,
                  1.08689085e-02, 1.08689083e-02, 1.08689082e-02, 1.08689082e-02,
                  7.54537492e-06, 7.54537467e-06, 7.54537462e-06, 7.54537449e-06,
                  7.54537439e-06, 7.54537432e-06, 7.54537431e-06, 7.54537400e-06])

    print(roulette_wheel_selection(P))
    0.640305652380855
    0.880640348452745
