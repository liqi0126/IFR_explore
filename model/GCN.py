import torch
import torch.nn as nn
from torch_geometric.nn import DenseGCNConv, RGCNConv


class RGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RGCNEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels // 2)
        )
        self.conv1 = RGCNConv(in_channels, out_channels, num_relations=6)
        self.conv2 = RGCNConv(out_channels, out_channels, num_relations=6)
        self.conv_out = RGCNConv(out_channels, out_channels // 2, num_relations=6)

    def forward(self, x, edge_index, edge_type):
        x = x.squeeze()
        self_feat = self.encoder(x).squeeze()
        x = self.conv1(x, edge_index, edge_type).relu()
        x = self.conv2(x, edge_index, edge_type).relu()
        graph_feat = self.conv_out(x, edge_index, edge_type).squeeze()
        return torch.cat([self_feat, graph_feat], -1)


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels // 2)
        )
        self.conv1 = DenseGCNConv(in_channels, out_channels)
        self.conv2 = DenseGCNConv(out_channels, out_channels)
        self.conv_out = DenseGCNConv(out_channels, out_channels // 2)

    def forward(self, x, adj):
        self_feat = self.encoder(x).squeeze()
        x = self.conv1(x, adj).relu()
        x = self.conv2(x, adj).relu()
        graph_feat = self.conv_out(x, adj).squeeze()
        return torch.cat([self_feat, graph_feat], -1)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = DenseGCNConv(in_channels, 2 * out_channels)
        self.conv_mu = DenseGCNConv(2 * out_channels, out_channels)
        self.conv_logstd = DenseGCNConv(2 * out_channels, out_channels)

    def forward(self, x, adj):
        x = self.conv1(x, adj).relu()
        return self.conv_mu(x, adj), self.conv_logstd(x, adj)


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearEncoder, self).__init__()
        self.conv = DenseGCNConv(in_channels, out_channels)

    def forward(self, x, adj):
        return self.conv(x, adj)


class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalLinearEncoder, self).__init__()
        self.conv_mu = DenseGCNConv(in_channels, out_channels)
        self.conv_logstd = DenseGCNConv(in_channels, out_channels)

    def forward(self, x, adj):
        return self.conv_mu(x, adj), self.conv_logstd(x, adj)
