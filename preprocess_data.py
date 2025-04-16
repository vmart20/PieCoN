import sys
import os
import pickle as pkl
import scipy as sp
import numpy as np
import pandas as pd
import networkx as nx
import torch
from numpy.linalg import eigh
from torch_geometric.datasets import WebKB
from torch_geometric.utils import to_dense_adj
from torch_geometric.datasets import Amazon
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.datasets import Planetoid



def normalize_graph(g):
    g = np.array(g)
    g = g + g.T
    g[g > 0.] = 1.0
    deg = g.sum(axis=1).reshape(-1)
    deg[deg == 0.] = 1.0
    deg = np.diag(deg ** -0.5)
    adj = np.dot(np.dot(deg, g), deg)
    return adj


def eigen_decompositon(g):
    "The normalized (unit “length”) eigenvectors, "
    "such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i]."
    g = normalize_graph(g)
    e, u = eigh(g)
    return e, u


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def feature_normalize(x):
    x = np.array(x)
    rowsum = x.sum(axis=1, keepdims=True)
    rowsum = np.clip(rowsum, 1, 1e10)
    return x / rowsum


def load_data(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("node_raw_data/{}/ind.{}.{}".format(dataset_str, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("node_raw_data/{}/ind.{}.test.index".format(dataset_str, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.sparse.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.sparse.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    return adj, features, labels


def eig_dgl_adj_sparse(g, sm=0, lm=0):
    A = g.adj(scipy_fmt='csr')
    deg = np.array(A.sum(axis=0)).flatten()
    D_ = sp.sparse.diags(deg ** -0.5)

    A_ = D_.dot(A.dot(D_))

    if sm > 0:
        e1, u1 = sp.sparse.linalg.eigsh(A_, k=sm, which='SM', tol=1e-5)
        e1, u1 = map(torch.FloatTensor, (e1, u1))

    if lm > 0:
        e2, u2 = sp.sparse.linalg.eigsh(A_, k=lm, which='LM', tol=1e-5)
        e2, u2 = map(torch.FloatTensor, (e2, u2))

    if sm > 0 and lm > 0:
        return torch.cat((e1, e2), dim=0), torch.cat((u1, u2), dim=1)
    elif sm > 0:
        return e1, u1
    elif lm > 0:
        return e2, u2
    else:
        pass


def generate_node_data(dataset):
    
    if dataset in ['cora', 'citeseer', 'pubmed']:

        if dataset == 'pubmed':
            data = Planetoid(root=f'node_raw_data/', name='pubmed')[0]
            y = data.y
            x = data.x
            adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=len(y)).toarray()
            x = feature_normalize(x)
            e, u = eigen_decompositon(adj)

            e = torch.FloatTensor(e)
            u = torch.FloatTensor(u)
            x = torch.FloatTensor(x)
            y = torch.LongTensor(y)
        else:
            adj, x, y = load_data(dataset)
            adj = adj.todense()
            x = x.todense()
            x = feature_normalize(x)
            e, u = eigen_decompositon(adj)

            e = torch.FloatTensor(e)
            u = torch.FloatTensor(u)
            x = torch.FloatTensor(x)
            y = torch.LongTensor(y)

        torch.save([e, u, x, y, torch.tensor(adj).to_dense()],  'proc_data/{}.pt'.format(dataset))

    elif dataset in ['photo', 'computer']:
        if dataset == 'computer':
            data = Amazon(root=f'node_raw_data/', name='Computers')[0]
            y = data.y
            x = data.x
            adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=len(y)).toarray()
            x = feature_normalize(x)
            e, u = eigen_decompositon(adj)

            e = torch.FloatTensor(e)
            u = torch.FloatTensor(u)
            x = torch.FloatTensor(x)
            y = torch.LongTensor(y)
        else:
            data = np.load('node_raw_data/amazon_electronics_photo.npz', allow_pickle=True)
            adj = sp.sparse.csr_matrix((data['adj_data'], data['adj_indices'], data['adj_indptr']), 
                                shape=data['adj_shape']).toarray()
            feat = sp.sparse.csr_matrix((data['attr_data'], data['attr_indices'], data['attr_indptr']), 
                                shape=data['attr_shape']).toarray()
            x = feature_normalize(feat)
            y = data['labels']
            e, u = eigen_decompositon(adj)

            e = torch.FloatTensor(e)
            u = torch.FloatTensor(u)
            x = torch.FloatTensor(x)
            y = torch.LongTensor(y)

        torch.save([e, u, x, y, torch.tensor(adj).to_dense()],  'proc_data/{}.pt'.format(dataset))
    
    elif dataset in ['amazon_ratings']:
        data = np.load(f'node_raw_data/{dataset}.npz', allow_pickle=True)

        y = data['node_labels']
        adj = sp.sparse.csr_matrix(to_dense_adj(torch.tensor(data['edges']).T, max_num_nodes=len(y))[0]).toarray()
        feat = data['node_features']
        x = feature_normalize(feat)
        e, u = eigen_decompositon(adj)

        e = torch.FloatTensor(e)
        u = torch.FloatTensor(u)
        x = torch.FloatTensor(x)
        y = torch.LongTensor(y)

        torch.save([e, u, x, y, torch.tensor(adj).to_dense()],  'proc_data/{}.pt'.format(dataset))


    elif dataset in ['chameleon', 'squirrel', 'actor']:
        edge_df = pd.read_csv('node_raw_data/{}/'.format(dataset) + 'out1_graph_edges.txt', sep='\t')
        node_df = pd.read_csv('node_raw_data/{}/'.format(dataset) + 'out1_node_feature_label.txt', sep='\t')
        feature = node_df[node_df.columns[1]]
        y = node_df[node_df.columns[2]]

        num_nodes = len(y)
        adj = np.zeros((num_nodes, num_nodes))

        source = list(edge_df[edge_df.columns[0]])
        target = list(edge_df[edge_df.columns[1]])

        for i in range(len(source)):
            adj[source[i], target[i]] = 1.
            adj[target[i], source[i]] = 1.
    
        if dataset == 'actor':
            # for sparse features
            nfeat = 932
            x = np.zeros((len(y), nfeat))

            feature = list(feature)
            feature = [feat.split(',') for feat in feature]
            for ind, feat in enumerate(feature):
                for ff in feat:
                    x[ind, int(ff)] = 1.

            x = feature_normalize(x)
        else:
            feature = list(feature)
            feature = [feat.split(',') for feat in feature]
            new_feat = []

            for feat in feature:
                new_feat.append([int(f) for f in feat])
            x = np.array(new_feat)
            x = feature_normalize(x)

        e, u = eigen_decompositon(adj)

        e = torch.FloatTensor(e)
        u = torch.FloatTensor(u)
        x = torch.FloatTensor(x)
        y = torch.LongTensor(y)

        torch.save([e, u, x, y, torch.tensor(adj)],  'proc_data/{}.pt'.format(dataset))

    elif dataset in ["texas"]:
        data = WebKB(root='node_raw_data', name=dataset)[0]
        # edge_index_tensor = torch.tensor(data.edge_index, dtype=torch.long).t().contiguous()
        y = data.y
        x = data.x
        adj = to_dense_adj(data.edge_index, max_num_nodes=len(y))[0]
        x = feature_normalize(x)
        e, u = eigen_decompositon(adj)

        e = torch.FloatTensor(e)
        u = torch.FloatTensor(u)
        x = torch.FloatTensor(x)
        y = torch.LongTensor(y)

        torch.save([e, u, x, y, torch.tensor(adj)],  'proc_data/{}.pt'.format(dataset))


if __name__ == '__main__':
    if not os.path.exists('proc_data'):
        os.makedirs('proc_data')
    
    # generate_node_data('cora')
    # generate_node_data('citeseer')
    # generate_node_data('photo')
    generate_node_data('chameleon')
    # generate_node_data('squirrel')
    # generate_node_data('actor')
    # generate_node_data('texas')
    # generate_node_data('amazon_ratings')
    # generate_node_data('pubmed')
