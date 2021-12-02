# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:00:05 2020

@author: juru
"""
import numpy as np

def mutate(x, mu):
    nVar=x.size
    nmu=np.ceil(mu*nVar).astype(int)
    j = np.random.choice(nVar, nmu)
    y=x
    y[j]=1-x[j]
    y=y.astype(bool)
    return y

if __name__=='__main__':
    x = np.array([False, False,  True, False, False, False, False,  True,  True,
           False, False,  True,  True,  True,  True,  True,  True,  True,
            True,  True, False, False, False, False, False,  True, False,
            True,  True, False, False, False, False, False, False,  True,
            True, False, False,  True, False,  True,  True, False,  True,
            True,  True, False, False, False, False,  True, False,  True,
            True, False,  True,  True,  True, False,  True, False, False,
           False,  True, False, False, False,  True, False, False, False,
            True,  True, False,  True, False,  True, False, False,  True,
           False,  True, False,  True,  True, False])
    mu =  0.022988505747126436
    print(mutate(x, mu))