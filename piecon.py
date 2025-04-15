import os, sys
sys.path.append(os.path.join(os.getcwd()))

import time
import yaml
import copy
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch_sparse import SparseTensor
from sklearn.metrics import roc_auc_score, mean_absolute_error, accuracy_score, r2_score
from models  import EigenvalueSegmentPrec
from utils import count_parameters, init_params, seed_everything, get_split, train_model, normalize_adjacency
from construct_const_filters import get_constant_filters
from scipy import signal
from copy import deepcopy
import seaborn as sns

hyperparameter_list = ['num_limits', 'save_path', 'dataset', 'cuda', 'seed', 'image', 'nlayer', 'num_heads', 'hidden_dim', 'epoch', 'lr', 'weight_decay', 'tran_dropout', 'feat_dropout', 'prop_dropout', 'norm']

def main_worker(config):
    if config["cuda"]:
        device = "cuda"
    torch.device(device)
    # device = 'cuda:{}'.format(args.cuda)
    # torch.cuda.set_device(args.seed)

    dataset = config['dataset']
    epoch = config['epoch']
    lr = config['lr']
    weight_decay = config['weight_decay']
    nclass = config['nclass']
    hidden_dim = config['hidden_dim']
    feat_dropout = config['feat_dropout']
    average_length = config['average_length']
    power = config["power"]


    data_path =  'proc_data/{}.pt'.format(dataset)
    e, u, x, y, adj = torch.load(data_path)
    e, u, x, y, adj = e.cuda(), u.cuda(), x.cuda(), y.cuda(), adj.cuda()


    norm_adj = SparseTensor.from_dense(normalize_adjacency(adj)).coalesce().cuda()

    if len(y.size()) > 1:
        if y.size(1) > 1:
            y = torch.argmax(y, dim=1)
        else:
            y = y.view(-1)

    train, valid, test = get_split(y, nclass) 
    train, valid, test = map(torch.LongTensor, (train, valid, test))
    train, valid, test = train.cuda(), valid.cuda(), test.cuda()

    # Get the number of edges
    num_nonzeros = torch.sum(adj > 0)

    const_filters = get_constant_filters(e, u, num_nonzeros, average_length, config["num_limits"], config["dataset"])

    nfeat = x.size(1)

    net = EigenvalueSegmentPrec(nclass, nfeat, hidden_dim, feat_dropout, power = power, const_filters = const_filters, norm_adj=norm_adj).cuda()

    net.apply(init_params)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    evaluation = torchmetrics.Accuracy(task='multiclass', num_classes=nclass)

    net_args = [x]
    best_val_acc, best_test_acc, best_state_dict, average_epoch_time = train_model(
        net, optimizer, evaluation, epoch, train, valid, test, y, net_args)

    return best_val_acc, best_test_acc, average_epoch_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--runs', type=int, default=1, help='Number of runs.')
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--dataset', default='chameleon')
    parser.add_argument('--lr', type=float, default=None, help='Initial learning rate.')
    parser.add_argument('--epoch', type=int, default=None, help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden_dim', type=int, default=None, help='Number of hidden dimension.')
    parser.add_argument('--feat_dropout', type=float, default=None)
    parser.add_argument('--num_limits', type=int, default=None)
    parser.add_argument('--power', type=int, default=None)

    args = parser.parse_args()
    
    config = yaml.load(open('configs/piecon_config.yaml'), Loader=yaml.SafeLoader)[args.dataset]
    
    vars_args = vars(args)
    for hyperparameter in vars_args.keys():
        if hyperparameter in vars_args and vars_args[hyperparameter] is not None:
            config[hyperparameter] = vars_args[hyperparameter]

    mean_val_acc, mean_test_acc, mean_epcoh_time = 0, 0, 0
    test_accs = []
    print(config)
    if config["runs"] > 1:
        for i in range(args.runs):
            print(f"Running with seed: {i}")
            config['seed'] = i
            seed_everything(config['seed'])
            val_acc, test_acc, epoch_time = main_worker(config)
            mean_val_acc += val_acc
            mean_test_acc += test_acc
            mean_epcoh_time += epoch_time
            test_accs.append(test_acc)
            print(f"Run {i} val acc: {val_acc:.4f}, test acc: {test_acc:.4f}, epoch time: {epoch_time:.4f}")
        mean_val_acc /= args.runs
        mean_test_acc /= args.runs
        mean_epcoh_time /= args.runs
    else:
        print(f"Running with seed: {config['seed']}")
        seed_everything(config['seed'])
        mean_val_acc, mean_test_acc, mean_epcoh_time = main_worker(config)
        test_accs.append(mean_test_acc)
    
    print(f"Mean val acc: {mean_val_acc:.4f}")
    print(f"Mean epoch time: {mean_epcoh_time:.4f}")
    print(f"Mean test acc: {mean_test_acc:.4f}")

    # Compute 95% confidence interval for test_accs
    test_accs = np.array(test_accs)
    std_error = np.std(test_accs, ddof=1) / np.sqrt(args.runs)
    confidence_interval = 100*np.max(np.abs(sns.utils.ci(sns.algorithms.bootstrap(test_accs,func=np.mean,n_boot=1000),95)-mean_test_acc.mean()))
    print(f"95% confidence interval for test accs: ±{confidence_interval:.4f}")
