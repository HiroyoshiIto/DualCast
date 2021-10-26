import numpy    as np
from scipy.sparse import *
from scipy import io

import networkx as nx
from networkx.algorithms import bipartite

import torch

import matplotlib.pyplot as plt

def spmat2sptensor(sparse_mat):
    dense = sparse_mat.todense()
    dense = torch.from_numpy(dense.astype(np.float32)).clone()
    return dense

def spmat2tensor(sparse_mat):
    shape = sparse_mat.shape
    sparse_mat = sparse_mat.tocoo()
    sparse_tensor = torch.sparse.FloatTensor(torch.LongTensor([sparse_mat.row.tolist(), sparse_mat.col.tolist()]),
                              torch.FloatTensor(sparse_mat.data.astype(np.float32)),shape)
    if torch.cuda.is_available():
        sparse_tensor = sparse_tensor.cuda()
    return sparse_tensor

class attr_graph_dynamic:
    def __init__(self,dirIn='./../data/',dataset='test',T=3):
        dirIn = dirIn + dataset
        # input G
        self.T = T
        self.G_list = []
        self.A_list = []
        self.Gmat_list = []
        self.Amat_list = []
        for t in range(T):
            G = nx.DiGraph()
            with open(dirIn + '/a.txt') as fp:
                lines = fp.readlines()
            for i,line in enumerate(lines):
                if i==0:
                    self.N = int(line.split()[0])
                    G_matrix = lil_matrix((self.N, self.N))
                else:
                    ni,nj,wij = line.split()
                    G.add_edge(int(ni),int(nj),weight=float(wij))
                    G_matrix[int(ni),int(nj)] = float(wij)
            self.G_list.append(G)
            self.Gmat_list.append(G_matrix)
            # input A
            A = nx.Graph()
            with open(dirIn + '/x.txt') as fp:
                lines = fp.readlines()
            nodes = []
            atts  = []
            edges = []
            for i,line in enumerate(lines):
                if i==0:
                    self.F = int(line.split()[1])
                    A_matrix = lil_matrix((self.N, self.F))
                else:
                    ni,nj,wij = line.split()
                    nodes.append(ni)
                    atts.append(nj+'_')
                    edges.append((ni,nj+'_'))
                    A_matrix[int(ni),int(nj)] = float(wij)
            A.add_nodes_from(nodes,bipartite=0)
            A.add_nodes_from(atts,bipartite=1)
            A.add_edges_from(edges)
            self.A_list.append(A)
            self.Amat_list.append(A_matrix)

class attr_graph_dynamic_spmat:
    def __init__(self,dirIn='./../data/',dataset='test',T=3):
        dirIn = dirIn + dataset
        # input G
        self.T = T
        self.G_list = []
        self.A_list = []
        self.Gmat_list = []
        self.Amat_list = []
        for t in range(T):
            G_matrix = io.loadmat(dirIn + '/G'+str(t)+'.mat', struct_as_record=True)['G']
            G = nx.from_scipy_sparse_matrix(G_matrix, create_using=nx.DiGraph())
            self.G_list.append(G)
            self.Gmat_list.append(G_matrix)
            A_matrix = io.loadmat(dirIn + '/A'+str(t)+'.mat', struct_as_record=True)['A']
            A = nx.DiGraph()
            self.A_list.append(A)
            self.Amat_list.append(A_matrix)

class attr_graph_dynamic_spmat_DBLP:
    def __init__(self,dirIn='./../data/',dataset='test',T=3):
        dirIn = dirIn + dataset
        # input G
        self.T = T
        self.G_list = []
        self.A_list = []
        self.Gmat_list = []
        self.Amat_list = []
        self.Gtensor_list = []
        survive = None
        for t in range(T):
            G_matrix = io.loadmat(dirIn + '/G'+str(t)+'.mat', struct_as_record=True)['G']
            if survive is None:
                survive = np.array(G_matrix.sum(axis=0))
            else:
                survive = np.multiply(survive,G_matrix.sum(axis=0))
        survive = np.ravel(survive>0)
        for t in range(T):
            G_matrix = io.loadmat(dirIn + '/G'+str(t)+'.mat', struct_as_record=True)['G']
            A_matrix = io.loadmat(dirIn + '/A'+str(t)+'.mat', struct_as_record=True)['A']
            G_matrix = G_matrix.T[survive].T
            A_matrix = A_matrix.T.dot(G_matrix).T
            A = nx.DiGraph()
            self.A_list.append(A)
            self.Amat_list.append(A_matrix)
            G_matrix = G_matrix.T.dot(G_matrix)
            G_matrix[G_matrix>0] = 1.
            G = nx.from_scipy_sparse_matrix(G_matrix, create_using=nx.DiGraph())
            self.G_list.append(G)
            self.Gmat_list.append(G_matrix)
            self.Gtensor_list.append(spmat2tensor(G_matrix))

class attr_graph_dynamic_spmat_twitter:
    def __init__(self,dirIn='./../data/',dataset='twitter',T=3):
        n_nodes_fortime = 75000
        dirIn = dirIn + dataset
        self.T = T
        self.G_list = []
        self.A_list = []
        self.Gmat_list = []
        self.Amat_list = []
        survive = None
        for t in range(T):
            G_matrix = io.loadmat(dirIn + '/G'+str(t)+'.mat', struct_as_record=True)['G']
            if survive is None:
                survive =  np.array(G_matrix.sum(axis=0)) * 1./T
            else:
                survive += np.array(G_matrix.sum(axis=0)) * 1./T
        survive = np.ravel(survive>0.1)
        for t in range(T):
            G_matrix = io.loadmat(dirIn + '/G'+str(t)+'.mat', struct_as_record=True)['G']
            A_matrix = io.loadmat(dirIn + '/A'+str(t)+'.mat', struct_as_record=True)['A']
            G_matrix = G_matrix[survive]
            G_matrix = G_matrix[:,survive][:n_nodes_fortime,:n_nodes_fortime]
            A_matrix = A_matrix[survive][:n_nodes_fortime,:]
            A = nx.DiGraph()
            self.A_list.append(A)
            self.Amat_list.append(A_matrix)
            G_matrix[G_matrix>0] = 1.
            G = nx.from_scipy_sparse_matrix(G_matrix, create_using=nx.DiGraph())
            self.G_list.append(G)
            self.Gmat_list.append(G_matrix)
        
