# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:19:39 2020

@author: juru
"""

import numpy as np
import networkx as nx

from TwoLinesIntersecting import two_lines_intersecting

def cost_function(pop,W,WindFarm,Penalization,Cable,n_wt,CoordX,CoordY):

    connected_status=False  #Constraints 1
    edges_count_status=False #Constraints 2
    capacity_status=False    #Constraints 3
    non_crossing_status=False #Constraints 4
    feeders_status=False      #Constraints 5

    G = nx.Graph() 
#    pop = x['Position']
    edges = W[pop][:,0:2]
    G.add_nodes_from([x+1 for x in range(n_wt)])
    G.add_edges_from([tuple(edge) for edge in edges]) 
#    
    #nx.draw_networkx(G, with_labels = True, node_color ='green') 
    connected_status = nx.is_connected(G)
    edges_count = G.number_of_edges()
    edges_count_status = n_wt - 1 == edges_count
    #nx.draw_networkx(G, with_labels = True, node_color ='green') 
    #=max(Cable.Capacities)
    T=np.array([x for x in nx.dfs_edges(G, source=1)]).astype(int)
    z=(W[pop,-1]).sum()*max(Cable.Price)
        
    if not connected_status:
#        print('connected_status: ', connected_status)
        z += Penalization.ConnectedComponents
    
    if not edges_count_status:
#        print('edges_count_status: ', edges_count_status)
        z += Penalization.EdgesCount
    
    if connected_status and edges_count_status:
        accumulator=np.zeros(T.shape[0])
        for j in range(n_wt-1):
            k = j + 2
            continue_ite = 1
            look_up = k
            while continue_ite:
    #            aux = T[T[:,0]==look_up]
                accumulator += (T[:,1]==look_up).astype(int)
                if (T[:,1]==look_up).astype(int).sum() > 1:
                    print('Error')
#                try:
                if T[(T[:,1]==look_up)][0, 0] == 1:
                    continue_ite = 0
#                except:
#                    print(T)
#                    print(look_up)
                else:
                    look_up=T[(T[:,1]==look_up)][0, 0]
        capacity_status = (accumulator[T[:,0]==1]<=Cable.MaxCap).all()
        if not capacity_status:
            z += Penalization.NodesFeeder*(np.max(accumulator[T[:,0]==1])-Cable.MaxCap)         
    if connected_status and edges_count_status and capacity_status:
      #%% ASSIGN: (i) LENGTH TO EACH ACTIVE EDGE (ii) CABLE TYPE TO EACH ACTIVE EDGE (iii) COST TO EACH ACTIVE EDGE
       T=np.append(T,np.zeros((accumulator.shape[0],3)),axis=1)
       for k in range(T.shape[0]):
           aux1=np.argwhere((W[:,0]==T[k,0]) & (W[:,1]==T[k,1]))
           aux2=np.argwhere((W[:,1]==T[k,0]) & (W[:,0]==T[k,1]))
           if aux2.size==0:
              T[k,2]=W[aux1,2]
           if aux1.size==0:
              T[k,2]=W[aux2,2]
       for k in range(accumulator.shape[0]):
           for l in range(Cable.Capacities.shape[0]):
               if accumulator[k]<=Cable.Capacities[l]:
                  break  
           T[k,3]=l
       for k in range(T.shape[0]):
           T[k,4]=(T[k,2]/1000)*Cable.Price[T.astype(int)[k,3]]
      #%% LINES CROSSING OUTER ROUTINE EMBEDDED WITH INNER ROUTINE
        #plt.figure(0)
       N1 = np.vstack((CoordX[edges.astype(int)[:,0]-1], CoordY[edges.astype(int)[:,0]-1])).T
       N2 = np.vstack((CoordX[edges.astype(int)[:,1]-1], CoordY[edges.astype(int)[:,1]-1])).T
       checker=0
       for k in range(N1.shape[0]):
           for l in range(N2.shape[0]-k-1):
              #print(k)
              #print(k+l+1)
              line1=np.array([N1[k],N2[k]])
              line2=np.array([N1[k+l+1],N2[k+l+1]])
              #x1 = [line1[0][0],line1[1][0]]
              #y1 = [line1[0][1],line1[1][1]]
              #plt.plot(x1, y1, label = "line 1")  
              #x2 = [line2[0][0],line2[1][0]]
              #y2 = [line2[0][1],line2[1][1]]
              #plt.plot(x2, y2, label = "line 2")    
              checker+=two_lines_intersecting(line1,line2)
       if checker==0:
         non_crossing_status=True              
      #%% DETERMINE NUMBER OF MAIN FEEDERS
       excess_feeders=0
       feeders_status = sum(T[:,0]==1) <= WindFarm.Feeders
       if feeders_status==False:
          excess_feeders=sum(T[:,0]==1)-WindFarm.Feeders        
      #%% CALCULATE ECONOMIC VALUE OF THE SOLUTION GIVEN THE PENALTIES  4 AND 5 (z)
       z=(Penalization.Crossing*checker*(1-non_crossing_status))+(Penalization.Feeders*excess_feeders*(1-feeders_status))+(sum(T[:,4]))
    #%%  Forming the array of constraints   
    cons = np.array([connected_status, edges_count_status, capacity_status,non_crossing_status,feeders_status])
#    return {'Tree':T,       #Variable T  
#            'Cost':z,       #Variable z
#            'Cons':cons,      } #Variable cons
#    print(T.shape)
    return T,z,cons
if __name__ == '__main__':
    pop=np.array([1,0,1,0,0,0,0,1,0,1,1,0,1,0,1,1,0,1,1,1,0,0,1,1,0,1,0,1,1,1,1,0,0,0,1,1,0,0,1,0,1,0,0,0,1,0,0,0,0,1,0,1,0,1,0,1,0,0,0,1,1,0,0,0,1,1,1,1,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,1,1,0,1,0,1,0,0,1,1,0,0,0,1,1,1,0,0,1,1,0,0,1,1,0,1,1,0,1,1,0,1,0,1,0,1,0,1,1,0,1,0,1,1,1,0]).astype(bool)
    #pop=np.array([0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,1,0,1,0,1,0,0,0,0,0]).astype(bool)
    print(cost_function(pop,W,WindFarm,Penalization,Cable))
#pop = np.array([0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,1,1,0,0,1,1,1,1,1,1,1,1,0,1,0,0,0,0,0,1,1,0,1,1,1,0,1,0])
#W = np.array([1,2,2943,40792056433,
#1,3,2696.56881408162,
#1,4,2502.57234543653,
#1,5,1722.54666356437,
#1,6,1903.98993489193,
#1,7,2870.00005856844,
#1,8,2627.69313617361,
#1,9,1305.96147097962,
#1,10,678.241483155260,
#1,11,1779.94372249856,
#1,12,1867.24086171472,
#1,13,580.973375657484,
#1,14,1068.72994287077,
#1,15,2429.10463306015,
#1,16,2882.76364346409,
#1,17,1804.91920412533,
#1,18,1469.30338690226,
#1,19,2239.12562222500,
#1,20,2493.63307468915,
#1,21,2677.59266922772,
#2,4,1233.42707859998,
#2,5,1262.80000000000,
#3,5,1233.42707859998,
#3,6,1262.80000000000,
#4,8,1233.42707859998,
#5,9,1233.42707859998,
#6,7,1406.00711235754,
#6,10,1233.42707859998,
#6,11,1262.80000000000,
#7,11,1233.42707859998,
#8,12,1262.80000000000,
#9,12,1233.42707859998,
#10,13,1233.42707859998,
#11,14,1233.42707859998,
#11,15,1262.80000000000,
#12,16,1233.42707859998,
#13,17,1233.42707859998,
#14,18,1233.42707859998,
#15,19,1233.42707859998,
#16,17,1406.00711235754,
#17,20,1262.80000000000,
#18,20,1233.42707859998,
#18,21,1262.80000000000,
#19,21,1233.42707859998])
#from data import get_edges, get_n_wt
##%% EXTERNAL INPUTS FOR THE 
#Edges = get_edges()
#class WindFarmObject():
#    def __init__(self, P=3.6):
#        self.P = P
#        self.GV = 33
#        self.F = 50
#        self.Feeders = 4
#WindFarm = WindFarmObject()
#WindFarm.VarSize = Edges.shape[0]  # Complete number of edges (variables)
#class PenalizationObject():
#    def __init__(self):
#        pass
#n_wt = int(get_n_wt())
#
#class CableObject():
#    def __init__(self):
#        self.ID = np.array([1,2,3,4,5,6,7,8,9,10,11]) # Cable ID
#        self.CrossSection = np.array([95,120,150,185,240,300,400,500,630,800,1000]) # Cable cross section [mm2]
#        self.NomCurrent = np.array([300,340,375,420,480,530,590,655,715,775,825]) # Current capacity [A]
#        self.Sn = np.array([17.15,19.43,21.43,24.01,27.44,30.29,33.72,37.44,40.87,44.3,47.16])
#        self.Price = np.array([223742,240134,255792,277908,311267,342883,386052,440203,498064,564661,627778])  #Unitary price [euros/km]
#class SettingsObject():
#    def __init__(self):
#        self.MaxIt = 200       #Maximum number of iterations
#        self.StallIt = 1000      #Maximum number of iterations without change of the fitness value
#        self.nPop = 100        #Number of individuals per generation
#        self.pc = 0.2          #Crossover percentage
#        self.pm = 0.2          #Mutation percentage 1 pair (Value not used, it is hardcoded in each iteration) NR
#        self.pm2 = 0.1         #Mutation percentage 2 pairs of variables (Value not used, it is hardcoded in each iteration) NR
#        self.pm3 = 0.1         #Mutation percentage 3 pairs of variables (Value not used, it is hardcoded in each iteration) NR
#        self.pm4 = 0.1         #Mutation percentage 1 variable (Value not used, it is hardcoded in each iteration) NR
#        self.AnimatedPlot = 1          #Animated plot status [0=off, 1=on]
#        self.PEdgesCut = 0.1             #Search space, reduces percentage of edges explored in the optimization by removing the larger ones for each node. All edges to the substation are always considered [1-0] 
#        self.PerformancePlots = 1      #Perfomance plots status: Creates plots related to the time performance of the GA [0=off, 1=on]
#        self.CableAvailable = np.array([7, 9, 11])-1    #Cables used for optimization process. Examples: [1:11], [1,3,6], [1:3].
#        self.beta=8
##%% Arranging classes
#Settings=SettingsObject()
#        
#Cable=CableObject()
#Cable.Available = Settings.CableAvailable                                #Cables considered for optimization (structure)
#Cable.Sbase = WindFarm.P                                      #Apparent power of WTs (MVA)
#Cable.Vbase = WindFarm.GV                                     #Grid voltage (kV) (22, 33, 45, 66, 132, 220)
#Cable.Freq = WindFarm.F                                       #Frequency (Hz)
#
#Cable.ID =Cable.ID[Cable.Available]                                       #
#Cable.CrossSection = Cable.CrossSection[Cable.Available]                   #Cable cross section (mm2)(Only cables considered for opt)
#Cable.NomCurrent= Cable.NomCurrent[Cable.Available]
#Cable.Sn= Cable.Sn[Cable.Available]                                       #Cable apparent power capacity [Only cables considered for opt)
#Cable.Capacities = np.floor(Cable.Sn/Cable.Sbase)                               #Maximum amount of WT supported for each cable
#Cable.MaxCap = np.max(Cable.Capacities)                                             #Maximum amount of WT supported from all cables
#Penalization = PenalizationObject()
#Penalization.BaseRough = (np.max(Edges[:,2])*(n_wt-1))*np.max(Cable.Price)   # Find base penalization according to the number of edges and the total length of them.
#Penalization.Base = np.floor(np.log10(Penalization.BaseRough))                       # Find order of magnitude of base penalization.
#Penalization.ConnectedComponents    = 10**(Penalization.Base+5)                 # Base penalization: Total connecitvity constraint
#Penalization.EdgesCount             = 10**(Penalization.Base+4)                 # Base penalization: Edges = Nodes - 1 constraint
#Penalization.NodesFeeder            = 10**(Penalization.Base+2)                 # Base penalization: Cable capacity constraint
#Penalization.Crossing               = 10**(Penalization.Base+1)                 # Base penalization: Cable crossings constraint
#Penalization.Feeders                = 10**(Penalization.Base+1)                 # Base penalization: Number of feeders connected to OSS
#
#cost_function(pop,W,WindFarm,Penalization,Cable)
#    
    
