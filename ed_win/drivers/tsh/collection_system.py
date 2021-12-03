# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 10:59:47 2020

@author: juru
"""
import numpy as np
from ed_win.drivers.tsh.c_mst import capacitated_spanning_tree
from ed_win.drivers.tsh.c_mst_cables import cmst_cables


def collection_system(X=[], Y=[], option=3, crossing_constraint=True, max_it=20000, Cables=[], plot=False):
    UL = max(Cables[:, 1])
    T, feasible = capacitated_spanning_tree(X, Y, option, UL, crossing_constraint)
    if crossing_constraint and not feasible:
        T, feasible = capacitated_spanning_tree(X, Y, option, UL, False)
        print('The two step heuristic algorithm did not obtain a fasible soltion without crossings. Consider using the Planrize module.')
    T_cables = cmst_cables(X, Y, T, Cables, plot)
    T_cables_cost = T_cables[:, -1].sum()
    return T_cables, T_cables_cost


if __name__ == "__main__":
    # X=[387100,383400,383400,383900,383200,383200,383200,383200,383200,383200,383200,383200,383300,384200,384200,384100,384000,383800,383700,383600,383500,383400,383600,384600,385400,386000,386100,386200,386300,386500,386600,386700,386800,386900,387000,387100,387200,383900,387400,387500,387600,387800,387900,388000,387600,386800,385900,385000,384100,384500,384800,385000,385100,385200,385400,385500,385700,385800,385900,385900,385500,385500,386000,386200,386200,384500,386200,386700,386700,386700,384300,384400,384500,384600,384300,384700,384700,384700,385500,384300,384300]
    # Y=[6109500,6103800,6104700,6105500,6106700,6107800,6108600,6109500,6110500,6111500,6112400,6113400,6114000,6114200,6115100,6115900,6116700,6118400,6119200,6120000,6120800,6121800,6122400,6122000,6121700,6121000,6120000,6119100,6118100,6117200,6116200,6115300,6114300,6113400,6112400,6111500,6110700,6117600,6108900,6108100,6107400,6106300,6105200,6104400,6103600,6103600,6103500,6103400,6103400,6104400,6120400,6119500,6118400,6117400,6116500,6115500,6114600,6113500,6112500,6111500,6105400,6104200,6110400,6109400,6108400,6121300,6107500,6106400,6105300,6104400,6113300,6112500,6111600,6110800,6110100,6109200,6108400,6107600,6106500,6106600,6105000]
    # X=np.array(X)
    # Y=np.array(Y)
    X = np.array([0., 2000., 4000., 6000.,
                  8000., 498.65600569, 2498.65600569, 4498.65600569,
                  6498.65600569, 8498.65600569, 997.31201137, 2997.31201137,
                  4997.31201137, 11336.25662483, 8997.31201137, 1495.96801706,
                  3495.96801706, 5495.96801706, 10011.39514341, 11426.89538545,
                  1994.62402275, 3994.62402275, 5994.62402275, 7994.62402275,
                  10588.90471566])
    Y = np.array([0., 0., 0., 0.,
                  0., 2000., 2000., 2000.,
                  2000., 2000., 4000., 4000.,
                  4000., 6877.42528387, 4000., 6000.,
                  6000., 6000., 3179.76530545, 5953.63051694,
                  8000., 8000., 8000., 8000.,
                  4734.32972738])

    option = 3
    # UL=5
    crossing_contr = False
    Cables = np.array([[500, 3, 100000], [800, 5, 150000], [1000, 10, 250000]])

    T, amount = collection_system(X, Y, option, crossing_contr, Cables=Cables, plot=True)
