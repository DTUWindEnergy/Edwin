# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 14:57:43 2021

@author: juru
"""
import numpy as np
import matplotlib.pyplot as plt
def output_data(X,Y,res_graph,Cables,oss,half,make_plot):
    ind=np.where((res_graph[oss*2:oss*2+half,3]!=0))[0]+oss*2
    T_d=np.concatenate((res_graph[ind,0].reshape(-1,1),res_graph[ind,1].reshape(-1,1),res_graph[ind,2].reshape(-1,1),res_graph[ind,4].reshape(-1,1),res_graph[ind,3].reshape(-1,1),res_graph[ind,5].reshape(-1,1)),axis=1)
    if make_plot:
        plt.figure()
        plt.plot(X[1:], Y[1:], 'r+',markersize=10, label='Turbines')
        plt.plot(X[0], Y[0], 'ro',markersize=10, label='OSS')
        for i in range(len(X)):
            plt.text(X[i]+50, Y[i]+50,str(i+1))
    colors = ['b','g','r','c','m','y','k','bg','gr','rc','cm']
    for i in range(Cables.shape[0]):
        index = T_d[:,3]==i
        if index.any():
           n1xs = X[T_d[index,0].astype(int)-1].ravel().T
           n2xs = X[T_d[index,1].astype(int)-1].ravel().T
           n1ys = Y[T_d[index,0].astype(int)-1].ravel().T
           n2ys = Y[T_d[index,1].astype(int)-1].ravel().T
           xs = np.vstack([n1xs,n2xs])
           ys = np.vstack([n1ys,n2ys])
           if make_plot:
               plt.plot(xs,ys,'{}'.format(colors[i]))
               plt.plot([],[],'{}'.format(colors[i]),label='Cable: {}'.format(i+1))
    if make_plot: plt.legend() 
    for i in range(len(T_d)):
        if T_d[i,4]<0:
            T_d[i,4]=-T_d[i,4]
        else:    
            a=np.copy(T_d[i,0])
            T_d[i,0]=np.copy(T_d[i,1])
            T_d[i,1]=a
    return T_d

