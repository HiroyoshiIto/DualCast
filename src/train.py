from __future__ import division
from __future__ import print_function

import argparse
import time
import os
import pickle

import numpy as np
import scipy.sparse as sp
from scipy.sparse import *
import networkx as nx
import matplotlib.pyplot as plt

import torch
from torch import optim

from optimizer import *
from load_data import *
from model import *

def spmat2sptensor(sparse_mat):
    shape = sparse_mat.shape
    sparse_mat = sparse_mat.tocoo()
    sparse_tensor = torch.sparse.FloatTensor(torch.LongTensor([sparse_mat.row.tolist(), sparse_mat.col.tolist()]),
                              torch.FloatTensor(sparse_mat.data.astype(np.float32)),shape)
    if torch.cuda.is_available():
        sparse_tensor = sparse_tensor.cuda()
    return sparse_tensor

def sampling_negEdge(adj, ns_rate):
    degOut = np.ravel(np.float_power(adj.sum(axis=1), 3/4))
    result_dice = np.random.multinomial(int(adj.sum()*ns_rate), degOut/degOut.sum())
    ind1_list = []
    for i,num in enumerate(result_dice):
        ind1_list.extend([i for j in range(num)])
    N_feat = adj.shape[1]
    ns_graph = lil_matrix(adj.shape)
    ind2_list = np.random.choice(np.array(range(N_feat)),len(ind1_list))
    ns_graph[ind1_list, ind2_list] = 1.
    x,y = adj.multiply(ns_graph).nonzero()
    ns_graph[x,y] = 0
    return ns_graph
        
def gae_for(args):
    print("Using {} dataset".format(args.dataset_str))

    if args.dataset_str in set(['DBLP','DBLP2','NIPS']):
        input_graph = attr_graph_dynamic_spmat_DBLP(dataset = args.dataset_str, T=args.T_full)
    elif args.dataset_str in set(['twitter']):
        input_graph = attr_graph_dynamic_spmat_twitter(dataset = args.dataset_str, T=args.T_full)
    elif args.dataset_str in set(['synth_stars','synth_stars_di','synth1']):
        input_graph = attr_graph_dynamic(dataset = args.dataset_str, T=args.T_full)
    else:
        input_graph = attr_graph_dynamic_spmat(dataset = args.dataset_str, T=args.T_full)
    
    adj = input_graph.Gmat_list
    feature = input_graph.Amat_list
    adj_neg = []
    feature_neg = []
    adj_label = []
    adjneg_label = []
    feature_label = []
    featureneg_label = []
    for t in range(input_graph.T):
        adj[t] = adj[t]
        _ = spmat2sptensor(adj[t])
        adj_label.append(_._values() / _._values())
        _neg = sampling_negEdge(adj[t], args.ns_rate)
        _neg = spmat2sptensor(_neg)
        adj_neg.append(_neg)
        adjneg_label.append(_neg._values() * 0)
        adj[t] = _

        feature_ = input_graph.Amat_list[t]
        _ = spmat2sptensor(feature_)
        feature[t] = _
        feature_label.append(_._values() / _._values())
        _neg = sampling_negEdge(feature_, args.ns_rate)
        _neg = spmat2sptensor(_neg)
        feature_neg.append(_neg)
        featureneg_label.append(_neg._values() * 0)

    model = predSN_dynamic(
            T_train = args.T_train, 
            N_nodes = adj[0].shape[0], 
            N_words = feature[0].shape[1],
            dim_edge = args.dim_edge,
            dim_interest = args.dim_interest)

    model.load_train(
            adj = adj[:args.T_train], 
            adj_neg = adj_neg[:args.T_train], 
            feature = feature[:args.T_train], 
            feature_neg = feature_neg[:args.T_train])

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    #### Fine tuning and train the forecasting parameter
    t_start = time.time()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        recovered_edge, recovered_negedge, recovered_attr, recovered_negattr, predicted_U, predicted_V, predicted_X = model()

        loss = loss_function_dynamic(model = model,
                            preds_edge=recovered_edge,
                            labels_edge=adj_label[:args.T_train],
                            preds_negedge=recovered_negedge[:args.T_train],
                            labels_negedge=adjneg_label[:args.T_train],
                            predicted_U=predicted_U,
                            predicted_V=predicted_V,
                            predicted_X=predicted_X,
                            preds_attr=recovered_attr,
                            labels_attr=feature_label[:args.T_train],
                            preds_negattr=recovered_negattr,
                            labels_negattr=featureneg_label[:args.T_train])

        loss.backward()
        optimizer.step()

    t_end = time.time() - t_start
    print("Running time: ", t_end)

    reconst_edge, reconst_negedge, reconst_attr, reconst_negattr = model.forecast(
                    test_adj=adj[args.T_train:], 
                    test_adj_neg=adj_neg[args.T_train:], 
                    test_feature=feature[args.T_train:], 
                    test_feature_neg=feature_neg[args.T_train:])

    list_error_edge, list_error_attr = forecasting_loss(args.T_full - args.T_train, 
                    preds_edge=reconst_edge, 
                    labels_edge=adj_label[args.T_train:],
                    preds_negedge=reconst_negedge,
                    labels_negedge=adjneg_label[args.T_train:],
                    preds_attr=reconst_attr,
                    labels_attr=feature_label[args.T_train:],
                    preds_negattr=reconst_negattr, 
                    labels_negattr=featureneg_label[args.T_train:])

    list_auc_edge, list_auc_attr = forecasting_auc(args.T_full - args.T_train, 
                    preds_edge=reconst_edge, 
                    labels_edge=adj_label[args.T_train:],
                    preds_negedge=reconst_negedge,
                    labels_negedge=adjneg_label[args.T_train:],
                    preds_attr=reconst_attr,
                    labels_attr=feature_label[args.T_train:],
                    preds_negattr=reconst_negattr, 
                    labels_negattr=featureneg_label[args.T_train:])

    return list_error_edge, list_error_attr, list_auc_edge, list_auc_attr

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--dim_edge', type=int, default=10, help='Dimmension of embeddings in edge space.')
    parser.add_argument('--dim_interest', type=int, default=10, help='Dimemension of embeddings in interest space.')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay parameter for Adam.')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset_str', type=str, default='NIPS', help='type of dataset.')
    parser.add_argument('--T_train', type=int, default=5, help='Number of time segments in input graph.')
    parser.add_argument('--T_full', type=int, default=10, help='Number of time segments in input graph.')
    parser.add_argument('--ns_rate', type=float, default=10, help='Negaive sampling rate.')
    parser.add_argument('--verbose', type=bool, default=False, help='Print the parameters or not.')

    args = parser.parse_args()
    args.notforecast = False

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    gae_for(args)
