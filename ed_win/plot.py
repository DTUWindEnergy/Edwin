import numpy as np
import matplotlib.pyplot as plt


def plot_network(X, Y, Cables, T_d):
    plt.figure()
    plt.plot(X[1:], Y[1:], 'r+', markersize=10, label='Turbines')
    plt.plot(X[0], Y[0], 'ro', markersize=10, label='OSS')
    for i in range(len(X)):
        plt.text(X[i] + 50, Y[i] + 50, str(i + 1))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'bg', 'gr', 'rc', 'cm']
    for i in range(Cables.shape[0]):
        index = T_d[:, 3] == i
        if index.any():
            n1xs = X[T_d[index, 0].astype(int) - 1].ravel().T
            n2xs = X[T_d[index, 1].astype(int) - 1].ravel().T
            n1ys = Y[T_d[index, 0].astype(int) - 1].ravel().T
            n2ys = Y[T_d[index, 1].astype(int) - 1].ravel().T
            xs = np.vstack([n1xs, n2xs])
            ys = np.vstack([n1ys, n2ys])
            plt.plot(xs, ys, '{}'.format(colors[i]))
            plt.plot([], [], '{}'.format(colors[i]), label='Cable: {} mm2'.format(Cables[i, 0]))
    plt.legend()
