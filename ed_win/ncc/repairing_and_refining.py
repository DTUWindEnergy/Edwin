# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 09:02:08 2021

@author: juru
"""

import numpy as np
from two_steps_collection_system import collection_system
from filtering import main_cycle,initial_flow,cycle_cost, filtering_cycle, generating_subcycles, organizing
from residual import residual_cost, residual_graph, pushing
from crossings import detecting_and_fixing_crossings
from bellman_ford import createGraph
from figuring import output_data
from plotting_twosteps import plotting_two_steps
import time

def repairing_and_refining(x,y,Cables,oss,Cap_oss,rounding_coeff,time_limit,check_tree,caring_crossings,make_plot):
    #%% Pre-processing: Finding a tree solution (probably) with cables crossings
    t=time.time()
    nodes=int(len(x))
    wt=int(nodes-1)
    num_neigh=round(rounding_coeff*wt)
    #Line below runs two-step heuristic with single OSS. Matrix 'edges' contains Node 1 (here are OSSs and WTs), Node 2 (only WTs), edge length, all zeros, all zeros. Matrix 'T' contains  Node 1 (OSSs and WTs), Node 2 (only WTs), length, cable, flow (# WTs), cost
    edges,T,amount=collection_system(X=np.array(x),Y=np.array(y),option=3,UL=max(Cables[:,1]),Inters_const=False,Cables=Cables,plot=make_plot) #For multiple OSSs, just make sure 'edges' does not have connections between OSSs. First part always OSSs to WTs.
    t_h=time.time()
    print('Total time in milisecods of the two-steps heuristic algorithm',1000*(t_h-t))
    #print(amount)
    #%% Checking crossings and fixing them if any from the two-step heuristic
    if caring_crossings:
        #Line below runs crossing cancelling algorithm. 'T_wc' same structure as 'T','amount_after_crossings' cost of the solution in Euros, 'number_crossings' number of cables crossings 
        T_wc,amount_after_crossings,number_crossings=detecting_and_fixing_crossings(x,y,edges,T,Cables,nodes,per_num_neigh_crossings=1,plot=make_plot)
        if number_crossings!=0:
            print('Changing from  caring about crossings to not caring')
            plotting_two_steps(x,y,Cables,T_wc)
            caring_crossings=False
    else:
        T_wc=np.copy(T)
        amount_after_crossings=amount
        print('Total number of iterations after two-step heuristic algorithm',amount_after_crossings)
    t_c=time.time()
    print('Total time in milisecods of crossings correction algorithm',1000*(t_c-t_h))
    #%% Organizing: Swapping nodes for ensuring that 'edges' and 'T_wc' have in Node 1 only WTs and in Node 2 WTs and OSSs. 'edges' has Node 1, Node 2, length
    edges,T_wc=organizing(oss,edges,T_wc)
    #%% Initial flow from two-steps: 'edges_ext' includes connections between OSSs and WTs in the first part, then the 'num_neigh' nearest connections fo each WT. The other half is the inverse arcs of the first half. It also adds element of 'T_wc' not present in 'edges' to 'edges_ext'
    #'edges_ext' contains Node 1, Node 2, length, flow (#WTs) - positive in same direction of arc or negative if not (same value of an arc and its inverse), cable type (same value for arc and its inverse), cost (same value for arc and its inverse)
    #half is the number of real edges (this means exluding edges to/from dummy, from root and the inverse arcs)
    edges_ext,half=initial_flow(edges,T_wc,oss,wt,num_neigh)
    #%% Initialization
    #Line below adds to 'edges_ext' in 'residual_graph' the connection to/from dummy node (at the beginning) and from the root to the dummy, OSSs and WTs (at the end). Same structure as 'edges_ext'
    res_graph=residual_graph(edges_ext,nodes,oss) #Residual graph
    vertices_array=np.unique(res_graph[:,0]) #Preparing graph #List of residual graph nodes.
    vertices_number=int(len(vertices_array)) #Preparing graph #Amount of nodes
    edges_number=int(len(res_graph)) #Preparing graph #Amount of edges in residual graph
    graph = createGraph(vertices_number,edges_number) #Modelling graph. It creates an objective containing the number of vertices, number of edges, and list of edges with initial node, final nodes and weight.
    #%% Process
    delta_array=np.unique(abs(res_graph[np.where((res_graph[oss*2:oss*2+half,3]!=0))[0]+oss*2,3]))#Define the set of flows to push
    len_delta_array=len(delta_array)
    cont,glob_it,glob_red_it=0,1,0
    t1=time.time()
    savings=0
    while(True):
        delta=delta_array[cont]
        res_cost,list_cables,list_costs=residual_cost(res_graph,oss,Cables,Cap_oss,delta,half) #Residual costs. 'res_cost' same size as 'res_graph'. Residual cost for each row. Same for 'list_cables' and 'list_costs'
        cycle=main_cycle(edges_number,vertices_array,graph,res_graph,res_cost) #Obtaining negative cost cycle by running Bellman Ford
        costs_elements,indices=cycle_cost(cycle,res_graph,res_cost) #'costs_elements' contains the residual cost of egde arc of the cycle. 'indices' the corresponding index of each arc in the residual graph
        cycle_fil,cost_fil=filtering_cycle(cycle,costs_elements) #Eliminating trivial-redundant arcs
        list_neg_subcycles=generating_subcycles(cycle_fil,cost_fil) #Finding sub-cycles (partition to subsets) after elimination of trivial-redundant arcs
        if len(list_neg_subcycles)>0:
            #print(delta)
            #print(list_neg_subcycles)
            for i in range(len(list_neg_subcycles)):
                costs_elements1,indices1=cycle_cost(list_neg_subcycles[i],res_graph,res_cost) 
                #print(sum(costs_elements1))
                if sum(costs_elements1)<0:
                    res_graph,flag=pushing(check_tree,oss,nodes,indices1,res_graph,list_cables,list_costs,delta,half,x,y,caring_crossings)
                    if flag:
                      delta_array=np.unique(abs(res_graph[np.where((res_graph[oss*2:oss*2+half,3]!=0))[0]+oss*2,3]))
                      len_delta_array=len(delta_array)
                      cont=-1       
                      glob_red_it+=flag
                      savings+=sum(costs_elements1)
                      print('Number of improvement iteration',glob_red_it)
                      print('Saving costs after iteration [Euros]',-sum(costs_elements1))
                      #print(list_neg_subcycles[i])
                      break
        cont+=1
        #print(cont)
        if cont>len_delta_array-1:
            break
        if time.time()-t1>time_limit:
           break
        glob_it+=1
    #%% Plotting Refined layout
    tot_time=1000*(time.time()-t_c)
    print('Total time in milisecods of NCC algorithm',(tot_time))
    b_0=output_data(np.array(x),np.array(y),res_graph,Cables,oss,half,make_plot)    
    return b_0
if __name__=='__main__':		
    x=[492222.525,500969.8909,501950.8187,502211.2685,501853.0626,498013.9335,500314.8128,501654.0183,497701.6476,499642.0243,500552.246,496586.1118,496800.5807,501465.2016,499159.9637,501867.5764,489171.3392,498508.9146,491168.6029,494825.1152,493031.1916,500066.2602,498514.5013,499367.1197,489930.1506,491039.6134,499641.9833,494513.9021,498193.9834,494096.2382,495783.1103,501653.1009,501092.3999,493155.0487,494217.905,500010.9242,494746.7483,493417.2959,500770.7666,502801.3332,492694.187,488001.2683,496013.9702,500245.9346,501254.014,499032.8478,493810.2656,501380.1011,496046.55,487785.7389,497125.2603,502115.9134,487332.8902,488961.6473,484344.8074,486941.9276,485830.2433,497500.9447,494246.5202,490730.1011,497839.4168,495271.9574,495076.1178,489447.3883,488389.1837,499567.0338,498834.5034,497846.2305,491956.7566,502948.9225,492465.5536,502513.3971,495589.6189,490259.5772,498112.0478]
    y=[5723736.425,5716442.551,5728020.747,5726751.572,5721584.325,5720806.303,5718947.555,5720627.244,5719454.939,5721968.446,5736349.496,5721123.671,5734883.657,5731880.065,5725956.673,5724304.582,5728549.388,5723477.196,5733379.93,5734485.727,5724446.265,5717754.682,5732451.797,5724801.192,5734207.542,5726959.362,5736943.823,5733014.453,5730018.724,5726146.335,5725080.539,5730347.529,5727413.487,5734843.691,5722913.247,5731569.687,5728921.131,5729758.787,5735203.281,5729550.741,5725857.327,5728798.097,5723988.818,5732698.606,5733361.027,5719218.222,5727414.249,5718734.6,5730612.67,5730908.601,5725388.991,5722771.855,5733173.503,5730117.926,5732529.838,5730330.303,5732212.998,5731035.034,5730992.026,5731224.571,5727927.036,5722708.868,5735922.861,5732045.734,5733808.684,5727636.955,5733742.392,5735553.682,5734602.669,5727866.529,5732481.172,5725230.885,5728360.318,5729268.449,5736710.149]
    oss=1
    Cables=np.array([[1,7,370000],[2,11,390000],[3,13,430000]])
    ncc_ws=True
    Cap_oss=np.array([[100]])
    rounding_coeff=1
    time_limit=120
    check_tree=True
    caring_crossings=True
    make_plot=True
    T_o=repairing_and_refining(x,y,Cables,oss,Cap_oss,rounding_coeff,time_limit,check_tree,caring_crossings,make_plot)