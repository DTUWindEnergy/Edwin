# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 09:36:35 2021

@author: juru
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from ed_win.drivers.two_lines_intersecting import two_lines_intersecting


def detecting_and_fixing_crossings(x, y, edges, T, Cables, nodes, per_num_neigh_crossings=1, plot=False):
    edges = edges[:, :3]
    int_counter = 0
    int_counter2 = 0
    num_neigh_crossings = round(per_num_neigh_crossings * nodes)
    T = np.copy(T)
    # %% Identifying crossings
    while(True):
        if int_counter == 0 and int_counter2 == 0:
            crossings = []
            number_crossings = 0
            for i in range(len(T)):
                aux = [i]
                for j in range(i + 1, len(T)):
                    node1 = T[i, 0].astype(int) - 1
                    node2 = T[i, 1].astype(int) - 1
                    node3 = T[j, 0].astype(int) - 1
                    node4 = T[j, 1].astype(int) - 1
                    line1 = np.array([[x[node1], y[node1]], [x[node2], y[node2]]])
                    line2 = np.array([[x[node3], y[node3]], [x[node4], y[node4]]])
                    if two_lines_intersecting(line1, line2):
                        aux += [j]
                        number_crossings += 1
                if len(aux) > 1:
                    crossings.append(aux)
            crossings.sort(key=len)
            crossings.reverse()
            print('Current number of crossings', number_crossings)
            if number_crossings == 0:
                infeasible = 0
                break
            # %% Identifying complement edges
            # edges_c=np.copy(edges)
            ind_list = []
            for i in range(len(T)):
                aux = np.where(((T[i, 0] == edges[:, 0]) & (T[i, 1] == edges[:, 1])) | ((T[i, 0] == edges[:, 1]) & (T[i, 1] == edges[:, 0])))[0]
                if len(aux) != 1:
                    print('Error')
                    raise Exception('Error')
                ind_list += [aux]
            ind_list = [x.item() for x in ind_list]
            edges_c = np.delete(edges, ind_list, axis=0)
            # %% Finding edge to replace
        if int_counter2 == len(crossings):
            print('I could not find a feasible solution. Giving the best I could do')
            break
        array_edges_to_eliminate = crossings[int_counter2][:]
        edges_to_eliminate = array_edges_to_eliminate[int_counter]
        # flag_el=0
        # for i in range(len(crossings)):
        #    for j in range(len(crossings[i])):
        #        #if not((T[crossings[i][j],0]<=oss) or (T[crossings[i][j],1]<=oss)):
        #        edges_to_eliminate=crossings[i][j]
        #        flag_el=1
        #        break
        #    if flag_el==1:
        #        break
        # %% Finding nodes-out-of-the-tree
        G1 = nx.Graph()
        G1.add_nodes_from([x + 1 for x in range(nodes)])
        source1 = T[edges_to_eliminate, 1].astype(int)
        T_s = np.copy(T)
        T_t = np.delete(T, edges_to_eliminate, axis=0)
        T_t = np.copy(T_t[:, :3])
        G1.add_edges_from([tuple(T_t[edge, 0:2]) for edge in range(len(T_t[:, 1]))])
        T_nodes = np.array([x for x in nx.dfs_edges(G1, source=source1)]).astype(int)
        nodes_out_tree = np.unique(T_nodes)
        if nodes_out_tree.size == 0:
            nodes_out_tree = T[edges_to_eliminate, 1].reshape(1, 1)
        # %% Finding candidate edges to incorporate to tree
        candidates = np.zeros((0, 3))
        for i in nodes_out_tree:
            aux_a = edges_c[np.where((edges_c[:, 0] == i) | (edges_c[:, 1] == i))[0][edges_c[np.where((edges_c[:, 0] == i) | (edges_c[:, 1] == i))[0], 2].argsort()]][:num_neigh_crossings]
            candidates = np.concatenate((candidates, aux_a))
        candidates = np.unique(candidates, axis=0)
        # %% Cancelling candidates that cross with installed edges
        el_ind_list = []
        for i in range(len(candidates)):
            for j in range(len(T_t)):
                node1 = candidates[i, 0].astype(int) - 1
                node2 = candidates[i, 1].astype(int) - 1
                node3 = T_t[j, 0].astype(int) - 1
                node4 = T_t[j, 1].astype(int) - 1
                line1 = np.array([[x[node1], y[node1]], [x[node2], y[node2]]])
                line2 = np.array([[x[node3], y[node3]], [x[node4], y[node4]]])
                if two_lines_intersecting(line1, line2):
                    el_ind_list += [i]
                    break
        candidates = np.delete(candidates, el_ind_list, axis=0)
        # %% Trying to add an edge
        cont = 0
        while(True):
            infeasible = 0
            # T=np.insert(T,9,np.array([1,17,-1,-1,-1,-1]),axis=0)
            T_t = np.insert(T_t, edges_to_eliminate, candidates[cont], axis=0)
            G = nx.Graph()
            G.add_nodes_from([x + 1 for x in range(nodes)])
            G.add_edges_from([tuple(T_t[edge, 0:2]) for edge in range(len(T_t[:, 1]))])
            T_d = np.array([x for x in nx.dfs_edges(G, source=1)]).astype(int)
            accumulator = np.zeros(T_d.shape[0])
            for j in range(len(T_t[:, 1])):
                k = j + 2
                continue_ite = 1
                look_up = k
                while continue_ite:
                    accumulator += (T_d[:, 1] == look_up).astype(int)
                    if ((T_d[:, 1] == look_up).astype(int).sum() > 1) or ((T_d[:, 1] == look_up).astype(int).sum() == 0):
                        # Exception('Error')
                        # print(look_up)
                        # print('Not feasible: disconnected graph')
                        infeasible = 1
                        break
                    if T_d[(T_d[:, 1] == look_up)][0, 0] == 1:
                        continue_ite = 0
                    else:
                        look_up = T_d[(T_d[:, 1] == look_up)][0, 0]
                if infeasible:
                    break
            if not(infeasible):
                T_d = np.append(T_d, np.zeros((accumulator.shape[0], 4)), axis=1)
                for k in range(T_d.shape[0]):
                    aux1 = np.argwhere((T_t[:, 0] == T_d[k, 0]) & (T_t[:, 1] == T_d[k, 1]))
                    aux2 = np.argwhere((T_t[:, 1] == T_d[k, 0]) & (T_t[:, 0] == T_d[k, 1]))
                    if aux2.size == 0:
                        T_d[k, 2] = T_t[aux1, 2]
                    if aux1.size == 0:
                        T_d[k, 2] = T_t[aux2, 2]
                    if (aux2.size == 0) and (aux1.size == 0):
                        raise Exception('Error')
                    for k in range(accumulator.shape[0]):
                        checker = 0
                        for m in range(Cables.shape[0]):
                            if accumulator[k] <= Cables[m, 1]:
                                checker = 1
                                break
                        if checker == 1:
                            T_d[k, 3] = m
                            T_d[k, 4] = accumulator[k]
                        else:
                            infeasible = 1
                            break
                    if infeasible:
                        break
                    for k in range(T_d.shape[0]):
                        T_d[k, 5] = (T_d[k, 2] / 1000) * Cables[T_d.astype(int)[k, 3], 2]
            if not(infeasible):
                int_counter = 0
                int_counter2 = 0
                T = np.copy(T_d)
                break
            else:
                T_t = np.delete(T_t, edges_to_eliminate, axis=0)
            cont += 1
            if cont == len(candidates):
                T = np.copy(T_s)
                int_counter += 1
                if int_counter == len(array_edges_to_eliminate):
                    int_counter2 += 1
                    int_counter = 0
                break
    amount_after_crossings = sum(T[:, -1])
    # print(sum(T[:,-1]))
    return T, amount_after_crossings, number_crossings
