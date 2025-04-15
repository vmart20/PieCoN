import os, sys
# sys.path.append(os.path.join(os.getcwd()))
import torch
import torch.nn as nn


torch.autograd.set_detect_anomaly(True)

class Combination(nn.Module):
    '''
    A mod combination the bases of polynomial filters.
    Args:
        channels (int): number of feature channels.
        depth (int): number of bases to combine.
        sole (bool): whether or not use the same filter for all output channels.
    '''
    def __init__(self, channels: int, depth: int, sole=False):
        super().__init__()
        if sole:
            self.comb_weight = nn.Parameter(torch.ones((1, depth, 1)))
        else:
            self.comb_weight = nn.Parameter(torch.ones((1, depth, channels)))

    def forward(self, x):
        '''
        x: node features filtered by bases, of shape (number of nodes, depth, channels).
        '''
        x = x * self.comb_weight
        x = torch.sum(x, dim=1)
        return x
    

class EigenvalueSegmentPrec(nn.Module):

    def __init__(self, nclass, nfeat, hidden_dim=128, feat_dropout=0.0, power=10, const_filters=[], norm_adj=None):
        super(EigenvalueSegmentPrec, self).__init__()

        self.const_filters = const_filters
        self.norm_adj = norm_adj
        self.power = power

        self.emb = nn.Sequential(
            nn.Dropout(feat_dropout),
            nn.Linear(nfeat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nclass),
            nn.Dropout(feat_dropout)
        )
        self.comb = Combination(nclass, len(const_filters) + self.power + 1, sole=False)

    def forward(self, x):

        h = self.emb(x)

        ps = []

        for Matrix in self.const_filters:
            ps.append(Matrix@h)

        ps.append(h)
        for _ in range(1, self.power + 1):
            ps.append(self.norm_adj@ps[-1])
    
        ps = [p.unsqueeze(1) for p in ps]
        p = torch.cat(ps, dim=1)
        res = self.comb(p)

        return res


