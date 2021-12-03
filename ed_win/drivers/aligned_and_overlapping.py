# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 10:05:41 2021

@author: mikf
"""
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

line1 = np.asarray([[ 997.31201137, 4000.        ],
                 [   0.,            0.        ]])
line2 = np.asarray([[ 498.65600569, 2000.        ],
                 [   0.,            0.        ]])

a1, a2 = line1[0], line1[1]
b1, b2 = line2[0], line2[1]

# check for overlapping
overlapping = 0
for a, b, p in zip([a1, a1, b1, b1], [a2, a2, b2, b2], [b1, b2, a1, a2]):
    ba = b - a
    pa = p - a
    ba_len = np.linalg.norm(ba)
    pa_len = np.dot(pa, ba) / ba_len
    if abs(pa_len - ba_len) < 1e-8 or abs(pa_len) < 1e-8: # help it to numerically find out that it can share a node
        pass
    else:
        overlapping += (0 < pa_len) & (pa_len < ba_len)
overlapping = overlapping > 0
print(overlapping)

# check for alignment
a2a1 = a2 - a1
b1a1 = b1 - a1
b2a1 = b2 - a1
a2a1_unit = a2a1 / np.linalg.norm(a2a1)
b1a1_unit = b1a1 / np.linalg.norm(b1a1)
b2a1_unit = b2a1 / np.linalg.norm(b2a1)
aligned = (np.abs(np.cross(a2a1_unit, b1a1_unit)) < 1e-8) & (np.abs(np.cross(a2a1_unit, b2a1_unit)) < 1e-8)
print(aligned)

print(aligned & overlapping)

plt.figure()
plt.plot(line1[:, 0], line1[:, 1], '+b')
plt.plot(line2[:, 0], line2[:, 1], 'xr')


