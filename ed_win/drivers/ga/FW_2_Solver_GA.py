# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 09:19:24 2020

@author: mikf
"""

import copy
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from numpy import newaxis as na

from data import Mat2Py
from CostFunction import cost_function
from RouletteWheelSelection import roulette_wheel_selection
from Crossover import crossover
from Mutate import mutate
from dict_to_class import DictToClass


class WindFarm(DictToClass):
    def __init__(self, **kwargs):
        self._get_default()
        super().__init__(kwargs)

    def _get_default(self):
        self.P = 3.6
        self.GV = 33
        self.F = 50
        self.Feeders = 4


class Cable(DictToClass):
    def __init__(self, **kwargs):
        self._get_default()
        super().__init__(kwargs)

    def _get_default(self):
        self.CrossSection = np.array([100, 500, 1000])
        self.Capacities = np.array([2, 4, 6])
        self.Price = np.array([223742, 340134, 555792])  # Unitary price [euros/km]


class Settings(DictToClass):
    def __init__(self, **kwargs):
        self._get_default()
        super().__init__(kwargs)

    def _get_default(self):
        self.MaxIt = 150  # Maximum number of iterations
        self.StallIt = 10  # Maximum number of iterations without change of the fitness value
        self.nPop = 100  # Number of individuals per generation
        self.pc = 0.2  # Crossover percentage
        self.pm = 0.2  # Mutation percentage 1 pair (Value not used, it is hardcoded in each iteration) NR
        self.pm2 = 0.1  # Mutation percentage 2 pairs of variables (Value not used, it is hardcoded in each iteration) NR
        self.pm3 = 0.1  # Mutation percentage 3 pairs of variables (Value not used, it is hardcoded in each iteration) NR
        self.pm4 = 0.1  # Mutation percentage 1 variable (Value not used, it is hardcoded in each iteration) NR
        self.AnimatedPlot = 1  # Animated plot status [0=off, 1=on]
        self.PEdgesCut = 0.3  # Search space, reduces percentage of edges explored in the optimization by removing the larger ones for each node. All edges to the substation are always considered [1-0]
        self.PerformancePlots = 1  # Perfomance plots status: Creates plots related to the time performance of the GA [0=off, 1=on]
        # self.CableAvailable = np.array([7, 9, 11])-1    #Cables used for optimization process. Examples: [1:11], [1,3,6], [1:3].
        self.beta = 8


class Penalization(DictToClass):
    def __init__(self, Cable, Edges, n_wt, **kwargs):
        self.Cable = Cable
        self.Edges = Edges
        self.n_wt = n_wt
        self._get_default()
        super().__init__(kwargs)

    def _get_default(self):
        self.BaseRough = (np.max(self.Edges[:, 2]) * (self.n_wt - 1)) * np.max(self.Cable.Price)   # Find base penalization according to the number of edges and the total length of them.
        self.Base = np.floor(np.log10(self.BaseRough))                       # Find order of magnitude of base penalization.
        self.ConnectedComponents = 10**(self.Base + 5)                 # Base penalization: Total connecitvity constraint
        self.EdgesCount = 10**(self.Base + 4)                 # Base penalization: Edges = Nodes - 1 constraint
        self.NodesFeeder = 10**(self.Base + 2)                 # Base penalization: Cable capacity constraint
        self.Crossing = 10**(self.Base + 1)                 # Base penalization: Cable crossings constraint
        self.Feeders = 10**(self.Base + 1)                 # Base penalization: Number of feeders connected to OSS


def upper_tri_masking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:, None] < r
    return A[mask]


def xy_to_edges(x, y, x0, y0):
    CoordX = np.append(x0, x)
    CoordY = np.append(y0, y)
    n_wt = int(CoordX.size)
    dx = CoordX[:, na] - CoordX[na, :]
    dy = CoordY[:, na] - CoordY[na, :]
    lengths = np.sqrt(dx ** 2 + dy ** 2)
    lengths = upper_tri_masking(lengths).ravel()
    nodes = np.zeros((n_wt, n_wt, 2))
    nodes[:, :, 0] = np.repeat(np.arange(1, n_wt + 1), n_wt).reshape(n_wt, n_wt)
    nodes[:, :, 1] = np.tile(np.arange(1, n_wt + 1), n_wt).reshape(n_wt, n_wt)
    nodes = upper_tri_masking(nodes)
    edges = np.hstack((nodes, lengths[:, na]))
    return CoordX, CoordY, n_wt, edges


class ECSGA():  # Electrical Collection System Genetic Algorithm
    def __init__(self, WindFarm, Cable, Settings, verbose=False, CoordX=None, CoordY=None):
        self.verbose = verbose
        # %% EXTERNAL INPUTS FOR THE GA
        # Cable.Available = Settings.CableAvailable                                #Cables considered for optimization (structure)

        # Cable.ID =Cable.ID[Cable.Available]                                       #
        # Cable.CrossSection = Cable.CrossSection[Cable.Available]                   #Cable cross section (mm2)(Only cables considered for opt)
        # Cable.NomCurrent= Cable.NomCurrent[Cable.Available]
        # Cable.Sn= Cable.Sn[Cable.Available]                                       #Cable apparent power capacity [Only cables considered for opt)
        # Cable.Capacities = np.floor(Cable.Sn/Cable.Sbase)                               #Maximum amount of WT supported for each cable
        # Cable.Price=Cable.Price[Cable.Available]

        self.WindFarm = WindFarm
        self.Cable = Cable
        self.Settings = Settings
        Cable.MaxCap = np.max(Cable.Capacities)  # Maximum amount of WT supported from all cables

        self.state = None
        self.CoordX = CoordX
        self.CoordY = CoordY

#
    def run(self, x=None, y=None, x0=None, y0=None, state=None):
        if not isinstance(x, type(None)):
            CoordX, CoordY, n_wt, Edges = xy_to_edges(x, y, x0, y0)
        self.CoordX = CoordX
        self.CoordY = CoordY
        penalization = Penalization(self.Cable, Edges, n_wt)

        # %% Filtering search space according the input PEdgesCut
        self.WindFarm.VarSize = Edges.shape[0]  # Complete number of edges (variables)
        PEdgesCut = self.Settings.PEdgesCut  # Per unit of smaller edges for each node to be kept

        EdgesCut = int(np.round(PEdgesCut * (n_wt - 1), 0))  # Number of smaller edges for each node to be kept
        Edges3 = None  # New list of edges

        # Reduce the number of edges to the predetermined percentage, all nodes
        # keep the edges to OSS no matter the distance.
        for i in range(int(n_wt)):
            Edges1 = Edges[Edges[:, 0] == i + 1]
            Edges2 = Edges[Edges[:, 1] == i + 1]
            Edges12 = np.append(Edges1, Edges2, axis=0)
            Edges12 = Edges12[Edges12[:, 2].argsort()]
            if i > 0:
                Edges12 = Edges12[:EdgesCut, :]
            if isinstance(Edges3, type(None)):
                Edges3 = Edges12
            else:
                Edges3 = np.append(Edges3, Edges12, axis=0)
        Edges3 = np.unique(Edges3, axis=0)
        if state:
            state_pos = np.full((Edges3.shape[0]), False)
            for s in range(state['edges'].shape[0]):
                remap = (state['edges'][s, 0].astype(int) == Edges3[:, 0].astype(int)) & (state['edges'][s, 1].astype(int) == Edges3[:, 1].astype(int))
                if remap.sum() == 1:
                    state_pos[remap] = True
                elif remap.sum() == 0:
                    remap2 = (state['edges'][s, 0].astype(int) == Edges[:, 0].astype(int)) & (state['edges'][s, 1].astype(int) == Edges[:, 1].astype(int))
                    Edges3 = np.append(Edges3, Edges[remap2], axis=0)
                    state_pos = np.append(state_pos, True)
                else:
                    'this should not happen'
        #
        # % Newly reduced set of edges (variables)
        Edges = Edges3
        W = Edges3
        #
        self.WindFarm.NewVarSize = Edges.shape[0]  # Search space considered: New number of variables after applying PEdgesCut
        #
        nVar = Edges.shape[0]                     # Number of Decision Variables
        VarSize = (1, nVar)                       # Decision Variables Matrix Size
        #

        # %% Stopping criteria for GA and pre-setting mutation parameters
        MaxIt = self.Settings.MaxIt                             # Maximum Number of Iterations to stop
        StallIt = self.Settings.StallIt                         # Number of iterations w/o change to stop

        # Population size
        nPop = self.Settings.nPop                               # Population Size

        # Crossover parameters
        pc = self.Settings.pc                                   # Crossover Percentage
        nc = 2 * np.round(pc * nPop / 2, 0)                                              # Number of Offsprings (also Parents)

        # Mutation parameters
        # Mutation 1 pair of edges at the same time
        pm = self.Settings.pm       # Mutation Percentage 1
        nm = np.round(pm * nPop, 0).astype(int)                      # Number of Mutants 1
        mu = 2 * 1 / nVar                            # Mutation Rate 1
        # Mutation 2 pairs of edges at the same time
        pm2 = self.Settings.pm2     # Mutation Percentage 2
        nm2 = np.round(pm2 * nPop, 0)                    # Number of Mutants 2
        mu2 = 4 * 1 / nVar                           # Mutation Rate 2
        # Mutation 3 pairs of edges at the same time
        pm3 = self.Settings.pm3     # Mutation Percentage 3
        nm3 = np.round(pm3 * nPop, 0)                    # Number of Mutants 3
        mu3 = 6 * 1 / nVar                           # Mutation Rate 3
        # Mutation 1 edge
        pm4 = self.Settings.pm4     # Mutatuon Percentage 4
        nm4 = np.round(pm4 * nPop, 0)                    # Number of Mutants 4
        mu4 = 1 / nVar                             # Mutation Rate 4
        # %% GA Penalization factors
        # %%
        # TODO: CAN WE INDEX X ?
        #
        # TODO: vectorize this loop. Create the numpy array directly and not through Pandas?
        #
        position_ij = np.full((nPop, nVar), False)  # pop
        cons_ik = np.full((nPop, 5), False)
        cost_i = np.zeros((nPop))
        tree_i = np.full((nPop), None)
        for i in range(nPop):
            if state and i == 0:  # if initialized with state, include this in the new populations.
                pop = state_pos
            else:
                pop = np.random.randint(0, 2, VarSize).ravel().astype(bool)
            position_ij[i, :] = pop
            tree, cost, cons = cost_function(pop, W, self.WindFarm, penalization, self.Cable, n_wt, CoordX, CoordY)
            cons_ik[i, :] = cons
            cost_i[i] = cost
            tree_i[i] = tree

        sort_index = cost_i.argsort()
        position_ij, cons_ik, cost_i, tree_i = position_ij[sort_index], cons_ik[sort_index], cost_i[sort_index], tree_i[sort_index]
        beta = self.Settings.beta
        worst_cost = cost_i[-1]
        termination_cost = []

        for it in range(MaxIt):
            # GA parameters can be changed at each stage (constraints met) of
            # the iterative process. Mainly the parameters changed relate to
            # the mutation operator
            if (cons_ik[0] == np.array([1, 0, 0, 0, 0])).all():
                pm = 0.4               # Per unit of the population with 1 pair of variable mutation
                nm = np.ceil(pm * nPop).astype(int)       # Number of individuals with a 1 pair of variable mutation (Has to be rounded)
                pm2 = 0.01             # 2 pairs of variables mutation
                nm2 = np.ceil(pm2 * nPop).astype(int)
                pm3 = 0.8              # 3 pairs of variables mutation
                nm3 = np.ceil(pm3 * nPop).astype(int)
                pm4 = 0.001             # 1 variable mutation
                nm4 = np.ceil(pm4 * nPop).astype(int)
            elif (cons_ik[0] == [1, 1, 0, 0, 0]).all():
                pm = 0.8
                nm = np.ceil(pm * nPop).astype(int)
                pm2 = 0.2
                nm2 = np.ceil(pm2 * nPop).astype(int)
                pm3 = 0.4
                nm3 = np.ceil(pm3 * nPop).astype(int)
                pm4 = 0.001
                nm4 = np.ceil(pm4 * nPop).astype(int)
            elif (cons_ik[0] == [1, 1, 1, 0, 0]).all():
                pm = 1.2
                nm = np.ceil(pm * nPop).astype(int)
                pm2 = 0.3
                nm2 = np.ceil(pm2 * nPop).astype(int)
                pm3 = 0.4
                nm3 = np.ceil(pm3 * nPop).astype(int)
                pm4 = 0.0001
                nm4 = np.ceil(pm4 * nPop).astype(int)
            elif (cons_ik[0] == [1, 1, 1, 0, 1]).all():
                pm = 3
                nm = np.ceil(pm * nPop).astype(int)
                pm2 = 1
                nm2 = np.ceil(pm2 * nPop).astype(int)
                pm3 = 1
                nm3 = np.ceil(pm3 * nPop).astype(int)
                pm4 = 0.2
                nm4 = np.ceil(pm4 * nPop).astype(int)
            elif (cons_ik[0] == [1, 1, 1, 1, 1]).all():
                pm = 3
                nm = np.ceil(pm * nPop).astype(int)
                pm2 = 1
                nm2 = np.ceil(pm2 * nPop).astype(int)
                pm3 = 1
                nm3 = np.ceil(pm3 * nPop).astype(int)
                pm4 = 0.2
                nm4 = np.ceil(pm4 * nPop).astype(int)

            P = np.exp(-beta * cost_i / worst_cost)
            P = P / np.sum(P)
            position_cj = np.full((int(nc / 2) * 2, nVar), False)  # pop
            cons_ck = np.full((int(nc / 2) * 2, 5), False)
            cost_c = np.zeros((int(nc / 2) * 2))
            tree_c = np.full((int(nc / 2) * 2), None)

            for k in range(int(nc / 2)):

                # Select Parents Indices
                i1 = roulette_wheel_selection(P)
                i2 = roulette_wheel_selection(P)

                # Select Parents
                p1 = copy.deepcopy(position_ij[i1])
                p2 = copy.deepcopy(position_ij[i2])

                # Perform Crossover
                y1, y2 = crossover(p1, p2)

                tree, cost, cons = cost_function(y1, W, self.WindFarm, penalization, self.Cable, n_wt, CoordX, CoordY)
                position_cj[k, :] = y1
                cons_ck[k, :] = cons
                cost_c[k] = cost
                tree_c[k] = tree

                tree, cost, cons = cost_function(y2, W, self.WindFarm, penalization, self.Cable, n_wt, CoordX, CoordY)
                position_cj[int(k + nc / 2), :] = y2
                cons_ck[int(k + nc / 2), :] = cons
                cost_c[int(k + nc / 2)] = cost
                tree_c[int(k + nc / 2)] = tree
        # %%
            """
            m : number of mutations, nm (changes with outer loop)
            f : mutation type (4 types of mutations)
            j : number of variables, nVar
            k : number of constraints (5 types of constraints)
            """
            position_m1j = np.full((nm, nVar), False)  # pop
            cons_m1k = np.full((nm, 5), False)
            cost_m1 = np.zeros(nm)
            tree_m1 = np.full(nm, None)

            position_m2j = np.full((nm2, nVar), False)  # pop
            cons_m2k = np.full((nm2, 5), False)
            cost_m2 = np.zeros(nm2)
            tree_m2 = np.full(nm2, None)

            position_m3j = np.full((nm3, nVar), False)  # pop
            cons_m3k = np.full((nm3, 5), False)
            cost_m3 = np.zeros(nm3)
            tree_m3 = np.full(nm3, None)

            position_m4j = np.full((nm4, nVar), False)  # pop
            cons_m4k = np.full((nm4, 5), False)
            cost_m4 = np.zeros(nm4)
            tree_m4 = np.full(nm4, None)

            for k in range(nm):
                index = np.random.randint(0, nPop)

                # Perform Mutation
                pom1 = copy.deepcopy(position_ij[index])
                xm1 = mutate(pom1, mu)

                tree, cost, cons = cost_function(xm1, W, self.WindFarm, penalization, self.Cable, n_wt, CoordX, CoordY)
                position_m1j[k, :] = xm1
                cons_m1k[k, :] = cons
                cost_m1[k] = cost
                tree_m1[k] = tree

            for k in range(nm2):
                index = np.random.randint(0, nPop)

                # Perform Mutation
                pom2 = copy.deepcopy(position_ij[index])
                xm2 = mutate(pom2, mu2)

                tree, cost, cons = cost_function(xm2, W, self.WindFarm, penalization, self.Cable, n_wt, CoordX, CoordY)
                position_m2j[k, :] = xm2
                cons_m2k[k, :] = cons
                cost_m2[k] = cost
                tree_m2[k] = tree

            for k in range(nm3):
                index = np.random.randint(0, nPop)

                # Perform Mutation
                pom3 = copy.deepcopy(position_ij[index])
                xm3 = mutate(pom3, mu3)

                tree, cost, cons = cost_function(xm3, W, self.WindFarm, penalization, self.Cable, n_wt, CoordX, CoordY)
                position_m3j[k, :] = xm3
                cons_m3k[k, :] = cons
                cost_m3[k] = cost
                tree_m3[k] = tree
            for k in range(nm4):
                index = np.random.randint(0, nPop)

                # Perform Mutation
                pom4 = copy.deepcopy(position_ij[index])
                xm4 = mutate(pom4, mu4)

                tree, cost, cons = cost_function(xm4, W, self.WindFarm, penalization, self.Cable, n_wt, CoordX, CoordY)
                position_m4j[k, :] = xm4
                cons_m4k[k, :] = cons
                cost_m4[k] = cost
                tree_m4[k] = tree

            position_ij = np.vstack([position_ij, position_cj, position_m1j, position_m2j, position_m3j, position_m4j])
            cons_ik = np.vstack([cons_ik, cons_ck, cons_m1k, cons_m2k, cons_m3k, cons_m4k])
            cost_i = np.vstack([cost_i[:, na], cost_c[:, na], cost_m1[:, na], cost_m2[:, na], cost_m3[:, na], cost_m4[:, na]]).ravel()
            tree_i = np.vstack([tree_i[:, na], tree_c[:, na], tree_m1[:, na], tree_m2[:, na], tree_m3[:, na], tree_m4[:, na]]).ravel()

            sort_index = cost_i.argsort()
            position_ij, cons_ik, cost_i, tree_i = position_ij[sort_index, :], cons_ik[sort_index, :], cost_i[sort_index], tree_i[sort_index]

            worst_cost = max(worst_cost, cost_i[-1])
            position_ij, cons_ik, cost_i, tree_i = position_ij[:nPop, :], cons_ik[:nPop, :], cost_i[:nPop], tree_i[:nPop]
            termination_cost.append(cost_i[0])
            if self.verbose:
                print(it)
                print(cons_ik[0])
                print(cost_i[0])
            self.tree_i = tree_i
            state = {'pos': position_ij[0, :],
                     'cons': cons_ik[0, :],
                     'cost': cost_i[0],
                     'tree': tree_i[0],
                     'edges': W[position_ij[0, :]][:, 0:2]}
            if len(termination_cost) > StallIt and (len(set(termination_cost[-(StallIt + 1):])) == 1 and (cons_ik[0]).all()):
                break
        self.state = state
        return cost_i[0], state

    def plot(self):
        CoordX = self.CoordX
        CoordY = self.CoordY
        tree_i = self.tree_i
#        plt.close()
        plt.plot(CoordX[1:], CoordY[1:], 'r+', markersize=10, label='Turbines')
        plt.plot(CoordX[0], CoordY[0], 'ro', markersize=10, label='OSS')
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'bg', 'gr', 'rc', 'cm']

        if (tree_i[0].shape)[1] == 2:  # If no feasible solution was found
            n1xs = CoordX[(tree_i[0].T[0:1] - 1).astype(int)].ravel().T
            n2xs = CoordX[(tree_i[0].T[1:2] - 1).astype(int)].ravel().T
            n1ys = CoordY[(tree_i[0].T[0:1] - 1).astype(int)].ravel().T
            n2ys = CoordY[(tree_i[0].T[1:2] - 1).astype(int)].ravel().T
            xs = np.vstack([n1xs, n2xs])
            ys = np.vstack([n1ys, n2ys])
            plt.plot(xs, ys, '{}'.format(colors[0]))
            plt.plot([], [], '{}'.format(colors[0]), label='N/A')
        else:
            for n, cable_type in enumerate(self.Cable.CrossSection):
                index = tree_i[0][:, 3] == n
                if index.any():
                    n1xs = CoordX[(tree_i[0][index].T[0:1] - 1).astype(int)].ravel().T
                    n2xs = CoordX[(tree_i[0][index].T[1:2] - 1).astype(int)].ravel().T
                    n1ys = CoordY[(tree_i[0][index].T[0:1] - 1).astype(int)].ravel().T
                    n2ys = CoordY[(tree_i[0][index].T[1:2] - 1).astype(int)].ravel().T
                    xs = np.vstack([n1xs, n2xs])
                    ys = np.vstack([n1ys, n2ys])
                    plt.plot(xs, ys, '{}'.format(colors[n]))
                    plt.plot([], [], '{}'.format(colors[n]), label='Cable: {} mm2'.format(self.Cable.CrossSection[n]))
                #    plt.plot(xs,ys,color=HSV_tuples[n])
                #    plt.plot([],[],color=HSV_tuples[n],label='Cable: {} mm2'.format(Cable.CrossSection[n]))
        plt.legend()


if __name__ == '__main__':
    x = np.array([7113.10301763, 7924.86160243, 7564.14443282,
                  8375.90301763, 9187.66160243, 9999.42018723, 8015.18584802,
                  8826.94443282, 9638.70301763, 10450.46160243, 9277.98584802,
                  10089.74443282, 10901.50301763, 11713.26160243, 9729.02726322,
                  10540.78584802, 11352.54443282, 12164.30301763, 11803.58584802,
                  12615.34443282])
    y = np.array([9426, 8278, 10574, 9426, 8278, 7130, 11722, 10574,
                  9426, 8278, 11722, 10574, 9426, 8278, 12870, 11722, 10574,
                  9426, 11722, 10574])
    x0, y0 = (10000, 10000)

    ecsga = ECSGA(WindFarm(),
                  Cable(),
                  Settings(MaxIt=5000, StallIt=100),
                  verbose=True)
    plt.close('all')
    cable_cost, state = ecsga.run(x, y, x0, y0, ecsga.state)
    ecsga.plot()
    plt.show()
    """
    for i in range(5):
        cable_cost, state = ecsga.run(x, y, x0, y0, ecsga.state)
        x[np.random.randint(0, len(x) - 1)] *= (100 + np.random.randint(-5, 5)) / 100
        y[np.random.randint(0, len(y) - 1)] *= (100 + np.random.randint(-5, 5)) / 100
        print(i, cable_cost)
        plt.figure(i)
        ecsga.plot()
        plt.show()
   """
