import torch
import numpy as np
from torch_geometric.utils import degree,k_hop_subgraph
from torch_geometric.datasets import Planetoid,Amazon,Coauthor,Actor,WikipediaNetwork
from rpca_ADMM import rpcaADMM 

dataset = Amazon(root='./LRGNN_master/dataset',name='Photo')


data=dataset.data

x=(data.x).numpy()
x = x - np.mean(x, axis = -1, keepdims = True)
x=torch.from_numpy(x)


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
data = data.to(device)

edge_index=data.edge_index
num_nodes=data.num_nodes

x_update= [ [] for i in range(num_nodes) ]
x_sum_ego= [ [ [] for j in range(0) ] for i in range(num_nodes) ]

k_hop=1



def avg(x_i): 
    if(len(x_i)):
        N = len(x_i)
        x = 0
        for k in range(N):
            x = x + x_i[k]
        x = x/N
        return x



for i in range(num_nodes):   
    print(i)
    subset= k_hop_subgraph([i],k_hop,edge_index)
    x_i_ego_0=x[subset[0],:].numpy() 
    if i == 1721 or i==2365:
        for k in range(len(subset[0])):
            x_sum_ego[i].append(x_i_ego_0[k,:])
        x_update[i]=avg(x_sum_ego[i])
        x_sum_ego[i]=[]
        continue
    
    h = {} 
    h=rpcaADMM(x_i_ego_0) 
    x_i_ego_1=h['X3_admm']
    
    for k in range(len(subset[0])):
        x_sum_ego[i].append(x_i_ego_1[k,:])
    x_update[i]=avg(x_sum_ego[i])
    x_sum_ego[i]=[]


np.save('./LRGNN_master/update_feats/Photo/feats_1_hop.npy',x_update)  


