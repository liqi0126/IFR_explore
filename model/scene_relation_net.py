# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import (negative_sampling, remove_self_loops, add_self_loops)

from torch_geometric.nn import GAE, InnerProductDecoder

EPS = 1e-15

class MyGAE(GAE):
    def __init__(self, encoder, decoder=None, init='prior'):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        self.scene_attr_net = nn.Sequential(
            nn.Linear(22, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
        )
        GAE.reset_parameters(self)

        self.init = init

    def build_gcn_graph_discrete(self, onehot_idx, pos, prior_graph, know_graph, prior_thres):
        edge_index = []
        edge_type = []
        for i in range(prior_graph.shape[1]):
            for j in range(prior_graph.shape[1]):
                if prior_graph[0, i, j] > prior_thres:
                    edge_index.append(torch.tensor([i, j]))
                    edge_type.append(0)
                    edge_index.append(torch.tensor([j, i]))
                    edge_type.append(1)

        for i in range(prior_graph.shape[1]):
            for j in range(prior_graph.shape[1]):
                if (onehot_idx[0, i] == onehot_idx[0, j]).all():
                    edge_index.append(torch.tensor([i, j]))
                    edge_type.append(2)

        for i in range(prior_graph.shape[1]):
            for j in range(prior_graph.shape[1]):
                if (pos[0, i] - pos[0, j]).square().sum().sqrt() < 0.5:
                    edge_index.append(torch.tensor([i, j]))
                    edge_type.append(3)

        for i in range(prior_graph.shape[1]):
            for j in range(prior_graph.shape[1]):
                if know_graph[0, i, j] == 1:
                    edge_index.append(torch.tensor([i, j]))
                    edge_type.append(4)
                    edge_index.append(torch.tensor([j, i]))
                    edge_type.append(5)

        edge_index = torch.stack(edge_index).T.cuda()
        edge_type = torch.tensor(edge_type).cuda()
        return edge_index, edge_type

    def build_gcn_graph(self, pos, prior_graph, know_graph, prior_thres, sym=False):
        scale = 0.6
        if self.init == 'ones':
            binary_graph = torch.ones_like(prior_graph)
        elif self.init == 'zeros':
            binary_graph = torch.zeros_like(prior_graph)
        elif self.init == 'prior':
            binary_graph = prior_graph * scale
            # binary_graph = prior_graph
        elif self.init == 'prior_oh':
            binary_graph = (prior_graph > prior_thres).float() * scale
            # binary_graph = (prior_graph > prior_thres).float()
        else:
            raise NotImplementedError

        dist_graph = torch.zeros_like(prior_graph)
        for i in range(prior_graph.shape[1]):
            dist_graph[0, i] = (pos[0, i] - pos).square().sum(-1).sqrt() < 0.5
        dist_graph *= scale

        # know_graph_t = know_graph.transpose(1, 2)
        # graph = torch.max(torch.stack([binary_graph, type_graph, dist_graph, know_graph, know_graph_t], -1), -1)[0]
        # graph = torch.max(torch.stack([binary_graph, type_graph, dist_graph, know_graph], -1), -1)[0]
        # graph = torch.max(torch.stack([binary_graph, type_graph, dist_graph], -1), -1)[0]
        graph = torch.max(torch.stack([binary_graph, dist_graph], -1), -1)[0]
        # graph = torch.ones_like(binary_graph)
        # graph = binary_graph
        # graph = type_graph
        # graph = dist_graph

        # if (know_graph != -1).any():
        #     graph[know_graph != -1] = torch.max(torch.stack([know_graph[know_graph != -1],
        #                                                     type_graph[know_graph != -1],
        #                                                     dist_graph[know_graph != -1]], -1), -1)[0]
        graph[know_graph != -1] = know_graph[know_graph != -1]
        # graph[know_graph_t != -1] = know_graph_t[know_graph_t != -1]
        # if sym:
        # graph = torch.max(torch.stack([graph, graph.transpose(1, 2)], -1), -1)[0]
        return graph


    def recon_loss(self, z, scene_attr, graph, pos_edge_index, neg_edge_index=None):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
            neg_edge_index (LongTensor, optional): The negative edges to train
                against. If not given, uses negative sampling to calculate
                negative edges. (default: :obj:`None`)
        """

        pos_loss = -torch.log(
            self.decoder(z, scene_attr, graph, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index, num_nodes=z.shape[0])
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, scene_attr, graph, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    def test(self, z, graph, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, graph, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, graph, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)


class LinearLayerDecoder(torch.nn.Module):
    def __init__(self, feat_dim, middle_dim):
        super(LinearLayerDecoder, self).__init__()
        self.net = nn.Sequential(
            # nn.Linear(2*feat_dim+2, feat_dim),
            nn.Linear(2*feat_dim+1, feat_dim),
            # nn.Linear(2*feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, middle_dim),
            nn.ReLU(),
            nn.Linear(middle_dim, middle_dim),
            nn.ReLU(),
            nn.Linear(middle_dim, 1),
        )

    def forward(self, z, scene_attr, graph, edge_index, sigmoid=True):
        feat = torch.cat([z[edge_index[0]], z[edge_index[1]], graph.squeeze()[edge_index[0], edge_index[1]].unsqueeze(-1)], dim=-1)
        value = self.net(feat)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, scene_attr, graph, sigmoid=True):
        with torch.no_grad():
            feat = []
            for i in range(z.shape[0]):
                for j in range(z.shape[0]):
                    feat.append(torch.cat([z[i], z[j], graph.squeeze()[i, j].unsqueeze(-1)], dim=-1))
            feat = torch.stack(feat)
            value = self.net(feat)
            value = value.reshape(z.shape[0], z.shape[0])
            return torch.sigmoid(value) if sigmoid else value
