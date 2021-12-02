# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 10:58:31 2021

@author: juru
"""
import numpy as np
from crossings import identifying_crossings_push


def residual_graph(edges_ext, nodes, oss):
    # residual_graph= Column 0: Node 1, C1: Node 2, C2: length. C3:Flow, C4: Cable type, C5: Cost edge
    to_dummy_nodes = np.zeros((oss * 2, 6))
    for i in range(oss):
        to_dummy_nodes[2 * i, 0], to_dummy_nodes[2 * i, 1], to_dummy_nodes[2 * i, 2], to_dummy_nodes[2 * i, 3], to_dummy_nodes[2 * i, 4], to_dummy_nodes[2 * i, 5] = i + 1, 0, 0, 0, -1, 0
        to_dummy_nodes[2 * i + 1, 0], to_dummy_nodes[2 * i + 1, 1], to_dummy_nodes[2 * i + 1, 2], to_dummy_nodes[2 * i + 1, 3], to_dummy_nodes[2 * i + 1, 4], to_dummy_nodes[2 * i + 1, 5] = 0, i + 1, 0, 0, -1, 0
    residual_graph = np.concatenate((np.concatenate((to_dummy_nodes, edges_ext), axis=0), np.concatenate((np.full((nodes + 1, 1), -1), np.array(range(nodes + 1)).reshape(-1, 1),
                                                                                                         np.zeros((nodes + 1, 2)), np.full((nodes + 1, 1), -1), np.zeros((nodes + 1, 1))), axis=1)), axis=0)
    return residual_graph


def residual_cost(residual_graph, oss, Cables, Cap_oss, delta, half):
    # residual_graph= Column 0: Node 1, C1: Node 2, C2: length. C3:Flow, C4: Cable type, C5: Cost edge
    residual_oss_dummy_node = np.zeros((oss * 2, 1))
    net_oss = np.zeros((oss, 1))
    # half=int(nodes*(nodes-1)/2)
    # Residual cost arcs from and to Supernode (dummy)
    for i in range(oss):
        ind = np.where((residual_graph[:half, 1] == i + 1))[0]
        net_oss[i] = sum(residual_graph[ind, 3])  # Flow entering the OSSs always positive
        if net_oss[i] + delta <= Cap_oss[i]:
            residual_oss_dummy_node[2 * i] = 0
        else:
            residual_oss_dummy_node[2 * i] = 1e11
        if delta <= net_oss[i]:
            residual_oss_dummy_node[2 * i + 1] = 0
        else:
            residual_oss_dummy_node[2 * i + 1] = 1e11
    residual_cost = np.zeros((len(residual_graph), 1))  # Residual cost array
    list_cables = np.full((len(residual_graph), 1), -1)  # New selected cable
    list_costs = np.zeros((len(residual_graph), 1))  # Cost of selected new cable
    # Residual cost arcs from and to Supernode (dummy)
    residual_cost[:oss * 2] = np.copy(residual_oss_dummy_node)
    # Residual cost arcs from OSSs to WTs
    forbidden = []
    for i in range(oss):
        ind2 = np.where((residual_graph[oss * 2 + half:oss * 2 + 2 * half, 0] == i + 1))[0] + oss * 2 + half
        for j in range(len(ind2)):
            flow = residual_graph[ind2[j], 3]
            l = residual_graph[ind2[j], 2]
            if flow >= delta:  # Flow entering the OSSs always positive
                selected_cable, cost_edge = cost(flow - delta, l, Cables)
                list_cables[ind2[j]] = selected_cable
                list_costs[ind2[j]] = cost_edge
                residual_cost[ind2[j]] = cost_edge - residual_graph[ind2[j], 5]
            else:
                residual_cost[ind2[j]] = 1e11
        forbidden += [k for k in ind2]
    # Residual cost rest of arcs
    for i in range(2 * oss, 2 * oss + 2 * half):
        if not(i in forbidden):
            flow = residual_graph[i, 3]
            l = residual_graph[i, 2]
            if i < 2 * oss + half:
                selected_cable, cost_edge = cost(abs(flow + delta), l, Cables)
                list_cables[i] = selected_cable
                list_costs[i] = cost_edge
                residual_cost[i] = cost_edge - residual_graph[i, 5]
            else:
                selected_cable, cost_edge = cost(abs(flow - delta), l, Cables)
                list_cables[i] = selected_cable
                list_costs[i] = cost_edge
                residual_cost[i] = cost_edge - residual_graph[i, 5]
    return residual_cost, list_cables, list_costs


def cost(number_wts, length, Cables):
    selected_cable, cost_edge = -1, 1e11
    if number_wts > 0:
        for i in range(len(Cables)):
            if number_wts <= Cables[i, 1]:
                selected_cable = i
                cost_edge = Cables[i, 2] * length / 1000
                break
    else:
        cost_edge = 0
    return selected_cable, cost_edge


def pushing(check_tree, oss, nodes, indices, res_graph, list_cables, list_costs, delta, half, x, y, caring):
    # half=int(nodes*(nodes-1)/2)
    new_residual_graph_temp = np.copy(res_graph)
    for i in indices:
        i = int(i)
        if (i < 2 * oss + half) and (i >= 2 * oss):
            new_residual_graph_temp[i, 3], new_residual_graph_temp[i + half, 3] = new_residual_graph_temp[i, 3] + delta, new_residual_graph_temp[i, 3] + delta
            new_residual_graph_temp[i, 4], new_residual_graph_temp[i + half, 4] = list_cables[i], list_cables[i]
            new_residual_graph_temp[i, 5], new_residual_graph_temp[i + half, 5] = list_costs[i], list_costs[i]
        elif(i >= 2 * oss + half):
            new_residual_graph_temp[i, 3], new_residual_graph_temp[i - half, 3] = new_residual_graph_temp[i, 3] - delta, new_residual_graph_temp[i, 3] - delta
            new_residual_graph_temp[i, 4], new_residual_graph_temp[i - half, 4] = list_cables[i], list_cables[i]
            new_residual_graph_temp[i, 5], new_residual_graph_temp[i - half, 5] = list_costs[i], list_costs[i]
    if (len(np.where((new_residual_graph_temp[oss * 2:oss * 2 + half, 3] != 0))[0]) == nodes - oss) or (not(check_tree)):
        matrix = new_residual_graph_temp[np.where((new_residual_graph_temp[oss * 2:oss * 2 + half, 3] != 0))[0] + oss * 2, :2]
        activator = identifying_crossings_push(x, y, matrix, caring)
        if activator:
            flag = 1
            new_residual_graph = np.copy(new_residual_graph_temp)
        else:
            flag = 0
            new_residual_graph = np.copy(res_graph)
    else:
        new_residual_graph = np.copy(res_graph)
        flag = 0
    return new_residual_graph, flag
