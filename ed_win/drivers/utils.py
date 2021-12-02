import numpy as np


def half_edges(X=[], Y=[]):
    """
    Calculate the first half of the complete graph

    Parameters
    ----------
    *X, Y: list([n_wt_oss]) or array type as well
            X,Y positions of the wind turbines and oss

    :return: n_wt_oss : int. Defining number of wind turbines with OSS
             edges_tot: Array. Array containing the complete graph
             half: int. True is solution is feasible. False if not.

    """
# %%  Initializing arrays, lists, variables (until line 46 .m file)
    n_wt_oss = len(X)  # Defining number of wind turbines with OSS
    half = int(n_wt_oss * (n_wt_oss - 1) / 2)
    edges_tot = np.zeros((2 * half, 5))  # Defining the matrix with Edges information
    cont_edges = 0
    for i in range(n_wt_oss):
        for j in range(i + 1, n_wt_oss):
            edges_tot[cont_edges, 0] = i + 1  # First element is first node (Element =1 is the OSS. and from 2 to Nwt the WTs)
            edges_tot[cont_edges, 1] = j + 1  # Second element is second node
            edges_tot[cont_edges, 2] = np.sqrt((X[j] - X[i])**2 + (Y[j] - Y[i])**2)  # Third element is the length of the edge
            cont_edges += 1
    return n_wt_oss, edges_tot, half
