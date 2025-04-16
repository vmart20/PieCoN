import os
import numpy as np
import torch
from torch_sparse import SparseTensor

eps = 1e-7

def thresholding_algo_significance(der, lag):
    der = np.array(der)
    significance = np.zeros(len(der) + 1)
    for i in range(lag, len(der) - lag ):
        begin = max(i-lag, 0)
        end = min(i + lag + 1, len(der) - 1)
        avg_prev = np.mean(der[begin: i])
        std_prev = np.std(der[begin: i])

        avg_next = np.mean(der[i + 1: end])
        std_next = np.std(der[i + 1: end])

        significance[i] += np.abs(der[i] - avg_prev)/ (std_prev + eps)
        significance[i] += np.abs(der[i] - avg_next)/ (std_next + eps)
        if der[i] == der[i-1]:
            significance[i] = 0

    return significance

def compute_deriv(e):
    return torch.concat((torch.tensor([0]).cuda(), e[1:]-e[:-1]))

def get_constant_filters(e, u, num_nonzeros, average_length, num_limits, dataset):

    ut = u.permute(1, 0)
    deriv_e = compute_deriv(e)

    significance = thresholding_algo_significance(deriv_e.cpu(), average_length)

    max_significance = np.max(significance)

    # Select the interval with eigenvalue 0 first
    seg = ((e>=-eps) & (e<=eps)).nonzero()
    if len(seg) != 0:
        significance[int(seg[0])] = max_significance + 1
        significance[int(seg[-1]) + 1] = max_significance + 2
    significance[0] = max_significance + 4
    significance[-1] = max_significance + 3

    limits = list(np.argsort(significance)[-num_limits:])

    limits = sorted(list(set(limits)))

    if len(limits) <=2:
        limits = []

    intervals = []
    for i in range(1, len(limits)):
        intervals.append((int(limits[i-1]), int(limits[i])))
    
    print(f"Intervals: {intervals}")

    if not os.path.exists("const_filters/"):
        os.makedirs("const_filters/")
    if os.path.isfile(f'const_filters/{dataset}_{average_length}_{num_limits}.pt'):
        const_filters = torch.load(f'const_filters/{dataset}_{average_length}_{num_limits}.pt')
    else:
        print("Constructing constant filters...")
        const_filters = []
        for i, interval in enumerate(intervals):
            temp_mat = (u[:, interval[0]:interval[1]]@ut[interval[0]:interval[1],:])
            temp_mat = temp_mat * (torch.abs(temp_mat) >= eps).float()
            flattened_mat = temp_mat.flatten()
            _, top_indices = torch.topk(torch.abs(flattened_mat), num_nonzeros)

            sparsified_mat = torch.zeros_like(flattened_mat)
            sparsified_mat[top_indices] = flattened_mat[top_indices]

            sparsified_mat = sparsified_mat.view_as(temp_mat)
            pos_sparsified_mat = torch.clamp(sparsified_mat, min=0)
            pos_sparsified_mat = torch.maximum(pos_sparsified_mat, pos_sparsified_mat.T)
            neg_sparsified_mat = -torch.clamp(sparsified_mat, max=0)
            neg_sparsified_mat = torch.maximum(neg_sparsified_mat, neg_sparsified_mat.T)
            const_filters.append(SparseTensor.from_dense(pos_sparsified_mat).coalesce().cuda())
            const_filters.append(SparseTensor.from_dense(neg_sparsified_mat).coalesce().cuda())
        torch.save(const_filters,  f'const_filters/{dataset}_{average_length}_{num_limits}.pt')
    return const_filters