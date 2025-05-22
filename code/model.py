import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import laplacian_filtering, graph_contrastive_loss, count_equal_elements
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np
from torch_geometric.nn import MessagePassing, APPNP


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class MvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MvEncoder, self).__init__()
        self.gcn = GraphConvolution(in_channels, out_channels)
        self.lin = nn.Linear(out_channels, out_channels)

    def forward(self, x, adj):
        x = F.relu(self.gcn(x, adj))
        return self.lin(x)

class ConEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConEncoder, self).__init__()
        self.lin_1 = nn.Linear(in_channels, out_channels)
        self.lin_2 = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        x = self.lin_1(x)
        return self.lin_2(x)

class Classifier(nn.Module):
    def __init__(self, hidden, num_classes):
        super(Classifier, self).__init__()
        self.lin_1 = nn.Linear(hidden, hidden)
        self.lin_2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        return self.lin_2(self.lin_1(x))

class CGCN(nn.Module):
    def __init__(self, dataset, args) -> None:
        super().__init__()
        self.args = args
        self.Mve_1 = MvEncoder(dataset.num_features, args.hidden)
        self.Mve_2 = MvEncoder(dataset.num_features, args.hidden)
        self.ConE = ConEncoder(args.hidden, args.hidden)
        self.classifier = Classifier(args.hidden, dataset.num_classes)

        self.SE = nn.Linear(args.num_nodes, args.hidden)
        self.dropout = args.dropout
    
    def forward(self, data, adj):
        x0, edge_index = data.x, data.edge_index

        z1 = self.Mve_1(x0, adj)
        z1 = F.dropout(z1, p=self.dropout, training=self.training)

        z2 = self.Mve_2(x0, adj)
        z2 = F.dropout(z2, p=self.dropout, training=self.training)

        z = (z1 + z2) / 2

        out = F.log_softmax(self.classifier(z), dim=1)

        nll = F.nll_loss(out[data.train_id], data.y[data.train_id]) 

        labels = data.y[data.train_id]

        bt_c = torch.mm(F.normalize(z1, dim=1), F.normalize(z2, dim=1).t())
        bt_loss = torch.diagonal(bt_c).add(-1).pow(2).mean() + off_diagonal(bt_c).pow(2).mean()

        cl = graph_contrastive_loss(z1, z2, labels)

        return z, out, nll + 0.1 * bt_loss + cl
