import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.01)
        if module.bias is not None:
            module.bias.data.zero_()

def normalize_adjacency(adj):
    adj = adj + adj.T
    adj = (adj > 0).float()
    deg = adj.sum(axis=1).reshape(-1)
    deg[deg == 0.] = 1.0
    deg = deg.pow(-0.5)
    adj = deg.unsqueeze(-1) * (adj * deg)
    return adj

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False


def get_split(y, nclass, train_prc=0.6, val_prc = 0.2):
    
    y = y.cpu()

    percls_trn = int(round(train_prc * len(y) / nclass))
    val_lb = int(round(val_prc * len(y)))

    indices = []
    for i in range(nclass):
        index = (y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0), device=index.device)]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)
    rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]
    valid_index = rest_index[:val_lb]
    test_index = rest_index[val_lb:]

    return train_index, valid_index, test_index

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def train_model(net, optimizer, evaluation, epoch, train, valid, test, y, net_args, early_stop=False):
    res = []
    epoch_times = []
    best_state_dict = {}
    counter = 0
    best_val_acc = -1

    for idx in range(epoch):
        time1 = time.time()
        net.train()
        optimizer.zero_grad()
        logits = net(*net_args)
        loss = F.cross_entropy(logits[train], y[train])

        loss.backward()
        optimizer.step()

        net.eval()
        logits = net(*net_args)

        
        val_loss = F.cross_entropy(logits[valid], y[valid]).item()

        train_acc = evaluation(logits[train].cpu(), y[train].cpu()).item()
        val_acc = evaluation(logits[valid].cpu(), y[valid].cpu()).item()
        test_acc = evaluation(logits[test].cpu(), y[test].cpu()).item()

        res.append([train_acc, val_loss, val_acc, test_acc])

        if idx % 100 == 0 and idx != 0:
            print(f"Epoch {idx}: train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            counter = 0
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_state_dict = net.state_dict()
        else:
            counter += 1

        if counter == 200 and early_stop:
            break

        time2= time.time()
        epoch_times.append(time2 - time1)
    
    average_epoch_time = np.mean(np.array(epoch_times))

    return best_val_acc, best_test_acc, best_state_dict, average_epoch_time
