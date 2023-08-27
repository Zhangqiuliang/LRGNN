# -*- coding: UTF-8 -*-
# coding=gbk   
import nni
from numpy import mat

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid,Amazon,Coauthor,Actor,WikipediaNetwork,WebKB
import math
import numpy as np
import nni

parser = argparse.ArgumentParser()
args = parser.parse_args()
dataset = Amazon(root='./LRGNN_master/dataset',name='Photo')
data = dataset[0]



data_path='./LRGNN_master/update_feats/Photo/feats_1_hop.npy'
features = np.load(data_path,allow_pickle=True) 
print(dataset.num_classes)
m=features.shape[0]
n=features.shape[1]
for i in range(m):
    for j in range(n):
            data.x[i][j]=features[i][j]



num_nodes=data.num_nodes
data.train_mask = torch.tensor([num_nodes], dtype=torch.bool)
data.val_mask = torch.tensor([num_nodes], dtype=torch.bool)
data.test_mask = torch.tensor([num_nodes], dtype=torch.bool)
splits_path='./LRGNN_master/dataset_splits/'
data_splits = np.load(splits_path+'Photo_split_20_30.npz')  
data.train_mask=torch.from_numpy(data_splits['train_mask'])
data.val_mask=torch.from_numpy(data_splits['val_mask'])
data.test_mask=torch.from_numpy(data_splits['test_mask'])

end_test_f1_list=[]
for time in range(10):

    ###################hyperparameters
    dropout = 0.4796
    alpha = 0.1
    lamda = 0.5
    hidden_dim = 64
    weight_decay1 = 0.005
    weight_decay2 = 0.001
    lr = 0.05
    patience = 100
    #####################



    class LRGNN_model(torch.nn.Module):
        def __init__(self):
            super(LRGNN_model, self).__init__()
            self.convs = torch.nn.ModuleList()
            self.convs.append(torch.nn.Linear(dataset.num_features,hidden_dim))
            self.convs.append(torch.nn.Linear(hidden_dim,dataset.num_classes))
            self.reg_params = list(self.convs[1:-1].parameters())
            self.non_reg_params = list(self.convs[0:1].parameters())+list(self.convs[-1:].parameters())

        def forward(self):
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
            _hidden = []
            x = F.dropout(x, dropout ,training=self.training)
            x = F.relu(self.convs[0](x))
            _hidden.append(x)
            for i,con in enumerate(self.convs[1:-1]):
                x = F.dropout(x, dropout ,training=self.training)
                beta = math.log(lamda/(i+1)+1)
                x = F.relu(con(x, edge_index,alpha, _hidden[0],beta,edge_weight))
            x = F.dropout(x, dropout ,training=self.training)
            x = self.convs[-1](x)
            return F.log_softmax(x, dim=1)



    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model, data = LRGNN_model().to(device), data.to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.reg_params, weight_decay=weight_decay1),
        dict(params=model.non_reg_params, weight_decay=weight_decay2)
    ], lr=lr)

    def train():
        model.train()
        optimizer.zero_grad()
        loss_train = F.nll_loss(model()[data.train_mask], data.y[data.train_mask])
        loss_train.backward()
        optimizer.step()
        return loss_train.item()


    @torch.no_grad()
    def test():
        model.eval()
        logits = model()
        loss_val = F.nll_loss(logits[data.val_mask], data.y[data.val_mask]).item()
        for _, mask in data('test_mask'):
            pred = logits[mask].max(1)[1]
            accs = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        return loss_val,accs


    best_val_loss = 9999999
    test_acc = 0
    bad_counter = 0
    best_epoch = 0
    for epoch in range(1, 1500):
        loss_tra = train()
        loss_val,acc_test_tmp = test()
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            test_acc = acc_test_tmp
            bad_counter = 0
            best_epoch = epoch
        else:
            bad_counter+=1
        
        log = 'Epoch: {:03d}, Train loss: {:.4f}, Val loss: {:.4f}, Test acc: {:.4f}'
        print(log.format(epoch, loss_tra, loss_val, test_acc))
        if bad_counter == patience:
            break
    end_test_f1_list.append(test_acc)
    log = '{:03d}best Epoch: {:03d}, Val loss: {:.4f}, Test acc: {:.4f}'
    print(log.format(time,best_epoch, best_val_loss, test_acc))

print(end_test_f1_list)
a = np.asarray(end_test_f1_list)
last_acc=np.mean(a)
print(last_acc)
print('mean:',np.mean(a))
print('standard deviation:',np.std(a)) 


