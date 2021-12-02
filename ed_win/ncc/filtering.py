# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 08:51:44 2021

@author: juru
"""
#Add array cost
import numpy as np
from bellman_ford import neg_cycle_bellman_ford, createGraph
def organizing(oss,edges,T):
    edges=edges[:,0:3]
    a,b=np.copy(edges[:,1]),np.copy(edges[:,0])
    a1,b2=np.copy(T[:,1]),np.copy(T[:,0])
    edges[:,0],edges[:,1]=a,b
    T[:,0],T[:,1]=a1,b2
    for i in range(len(T)):
        if T[i,0]<=oss:
           a11,b22=np.copy(T[i,1]),np.copy(T[i,0])
           T[:,0],T[:,1]=a11,b22
    return edges,T
def main_cycle(edges_number,vertices_array,graph,residual_graph,residual_cost):
    for i in range(edges_number):
        graph.edge[i].src = residual_graph[i,0]
        graph.edge[i].dest = residual_graph[i,1]
        graph.edge[i].weight = residual_cost[i]
    cycle=neg_cycle_bellman_ford(vertices_array,graph,-1)
    return cycle
def cycle_cost(cycle,residual_graph,residual_cost):
    costs_elements,indices=np.zeros((int(len(cycle)/2))),np.zeros((int(len(cycle)/2)))
    for i in range(0,len(cycle)-1,2):
        ind=np.where((residual_graph[:,0]==cycle[i]) & (residual_graph[:,1]==cycle[i+1]))[0]
        if ind.size>1:
           raise Exception('Error')
        costs_elements[int(i/2)]=residual_cost[ind]
        indices[int(i/2)]=ind[0]
    return costs_elements,indices
def filtering_cycle(cycle,cost):
    len_cycle=len(cycle)
    if len_cycle>4:
        pos_eliminate,pos_eliminate_cost=[],[]
        for i in range(0,len_cycle-4,2):
            for j in range(i+2,len_cycle-1,2):
                if (cycle[i]==cycle[j+1] and cycle[i+1]==cycle[j]):
                   pos_eliminate.append(i)
                   pos_eliminate.append(i+1)
                   pos_eliminate.append(j)
                   pos_eliminate.append(j+1)
                   pos_eliminate_cost.append(int(i/2))
                   pos_eliminate_cost.append(int(j/2))               
                if  (cycle[i]==cycle[j] and cycle[i+1]==cycle[j+1]):
                   raise Exception('Why is this happening?')
                   pos_eliminate.append(i)
                   pos_eliminate.append(i+1)                
                   pos_eliminate_cost.append(int(i/2))
        cycle_fil=np.delete(cycle,pos_eliminate)
        cost_fil=np.delete(cost,pos_eliminate_cost) 
    else:
        cycle_fil,cost_fil=np.array([]),np.array([])
    return cycle_fil,cost_fil
def generating_subcycles(cycle_fil,cost_fil):
    int_cycle,int_cost=np.copy(cycle_fil),np.copy(cost_fil)
    list_neg_subcycles=[]
    if cycle_fil.size>0:
        while(True):
            vertices_array=np.unique(int_cycle)
            vertices_number=int(len(vertices_array))
            edges_number=int(len(int_cycle)/2)
            graph = createGraph(vertices_number,edges_number)
            for i in range(edges_number):
                graph.edge[i].src = int_cycle[i*2]
                graph.edge[i].dest = int_cycle[i*2+1]
                graph.edge[i].weight = int_cost[i]
            cycle_temp=neg_cycle_bellman_ford(vertices_array,graph, int_cycle[0])
            if cycle_temp.size == 0:
                break
            list_neg_subcycles.append(cycle_temp)
            pos_eliminate1,pos_eliminate2=[],[]
            for i in range(0,len(cycle_temp)-1,2):
                for j in range(0,len(int_cycle)-1,2):
                    if (cycle_temp[i]==int_cycle[j]) and (cycle_temp[i+1]==int_cycle[j+1]):
                       pos_eliminate1.append(j) 
                       pos_eliminate1.append(j+1) 
                       pos_eliminate2.append(int(j/2)) 
            int_cycle=np.delete(int_cycle,pos_eliminate1)
            int_cost=np.delete(int_cost,pos_eliminate2)
            if int_cycle.size==0:
                break
    return list_neg_subcycles
def initial_flow(edges,T,oss,wt,num_neigh):
    segm=edges[oss*wt:len(edges)]
    cons_ma=np.zeros((0,3))
    for i in range(oss+1,wt+1):
        aux_a=segm[np.where((segm[:,0]==i) | (segm[:,1]==i))[0][segm[np.where((segm[:,0]==i) | (segm[:,1]==i))[0],2].argsort()]][:num_neigh]
        cons_ma=np.concatenate((cons_ma,aux_a))
    cons_ma=np.unique(cons_ma,axis=0)
    edges_int=np.concatenate((edges[:oss*wt],cons_ma))
    half,half_o=len(edges_int),len(edges_int)
    a,b=np.copy(edges_int[:,0]),np.copy(edges_int[:,1])
    aux_b=np.concatenate((b.reshape(-1,1),a.reshape(-1,1),edges_int[:,2].reshape(-1,1)),axis=1)
    edges_int=np.concatenate((edges_int,aux_b),axis=0)
    edges_int=np.concatenate((edges_int,np.zeros((len(edges_int),1)),np.full((len(edges_int),1),-1),np.zeros((len(edges_int),1))),axis=1)
    #half=int(nodes*(nodes-1)/2)
    aux_compl_1,aux_compl_2=np.zeros((0,6)),np.zeros((0,6))
    for i in range(len(T)):        
       ind=np.where((edges_int[:half,0]==T[i,0]) & (edges_int[:half,1]==T[i,1]))[0]
       flag,flag2=0,0
       if ind.size==0:
           ind=np.where((edges_int[:half,0]==T[i,1]) & (edges_int[:half,1]==T[i,0]))[0]
           flag=1
           if ind.size==0:
               raise Exception('An error occurred')
               half_o+=1
               flag2=1
               aux_c_1,aux_c_2=np.zeros((1,6)),np.zeros((1,6))
               aux_c_1[0,0],aux_c_1[0,1],aux_c_1[0,2],aux_c_1[0,3],aux_c_1[0,4],aux_c_1[0,5]=T[i,0],T[i,1],T[i,2],T[i,4],T[i,3],T[i,5]
               aux_c_2[0,0],aux_c_2[0,1],aux_c_2[0,2],aux_c_2[0,3],aux_c_2[0,4],aux_c_2[0,5]=T[i,1],T[i,0],T[i,2],T[i,4],T[i,3],T[i,5]
               aux_compl_1=np.concatenate((aux_compl_1,aux_c_1),axis=0)
               aux_compl_2=np.concatenate((aux_compl_2,aux_c_2),axis=0)
       edges_int[ind,4]=T[i,3]
       edges_int[ind,5]=T[i,5]
       edges_int[ind+half,4]=T[i,3]
       edges_int[ind+half,5]=T[i,5]       
       if flag==0:
          edges_int[ind,3],edges_int[ind+half,3]=T[i,4],T[i,4]
       if flag==1 and flag2==0:
          edges_int[ind,3],edges_int[ind+half,3]=-T[i,4],-T[i,4] 
    edges_int=np.concatenate((edges_int[:half],aux_compl_1,edges_int[half:len(edges_int)],aux_compl_2),axis=0)
    return edges_int,half_o
if __name__ == "__main__":
    import time
    t=time.time()
    cycle=np.array([1,2,2,3,3,4,4,5,5,6,6,7,7,5,5,4,4,3,3,1])     
    cost=np.array([-2,2,0,2,0,0,-1,2,-3,-3])
    cycle_fil,cost_fil=filtering_cycle(cycle,cost)
    list_neg_subcycles=generating_subcycles(cycle_fil,cost_fil)
    tot_time=1000*(time.time()-t)
    print('Total time in milisecods',tot_time)
    
    