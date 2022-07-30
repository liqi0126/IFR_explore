
import numpy as np

from torch.nn import Parameter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.nn import DenseGCNConv, RGCNConv
from gym_IFR.envs.relation import *


class GCN(torch.nn.Module):
    def __init__(self, feat_dim=121, hidden_dim=64, use_scene_attr=True):
        super(GCN, self).__init__()
        self.conv1 = DenseGCNConv(feat_dim, hidden_dim)
        self.conv2 = DenseGCNConv(hidden_dim, hidden_dim)
        self.conv3 = DenseGCNConv(hidden_dim, hidden_dim)
        self.use_scene_attr = use_scene_attr
        self.scene_attr_net = nn.Sequential(
            nn.Linear(22, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
        )

    def forward(self, pn_feat, obj_pos, obj_rot, obj_size, obj_attr, adj):
        if isinstance(pn_feat, np.ndarray):
            pn_feat = torch.from_numpy(pn_feat)
            obj_pos = torch.from_numpy(obj_pos)
            obj_rot = torch.from_numpy(obj_rot)
            obj_size = torch.from_numpy(obj_size)
            obj_attr = torch.from_numpy(obj_attr)
            adj = torch.from_numpy(adj)
        pn_feat = pn_feat.float().cuda()
        obj_pos = obj_pos.float().cuda()
        obj_rot = obj_rot.float().cuda()
        obj_size = obj_size.float().cuda()
        obj_attr = obj_attr.float().cuda()
        adj = adj.float().cuda()

        if self.use_scene_attr:
            scene_attr_embedding = self.scene_attr_net(torch.cat([obj_pos, obj_rot, obj_size, obj_attr], -1))
            # x = torch.cat([pn_feat, obj_idx, scene_attr_embedding], -1)
            x = torch.cat([pn_feat, scene_attr_embedding], -1)
        else:
            # x = torch.cat([pn_feat, obj_idx], -1)
            x = pn_feat

        x = self.conv1(x, adj).relu()
        x = self.conv2(x, adj).relu()
        x = self.conv3(x, adj)
        return x


class RGCN(torch.nn.Module):
    def __init__(self, feat_dim=121, hidden_dim=64, use_scene_attr=True):
        super(RGCN, self).__init__()
        self.conv1 = RGCNConv(feat_dim, hidden_dim, num_relations=6)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations=6)
        self.conv3 = RGCNConv(hidden_dim, hidden_dim, num_relations=6)
        self.use_scene_attr = use_scene_attr
        self.scene_attr_net = nn.Sequential(
            nn.Linear(22, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
        )

    def forward(self, pn_feat, obj_pos, obj_rot, obj_size, obj_attr, adj):
        if isinstance(pn_feat, np.ndarray):
            pn_feat = torch.from_numpy(pn_feat)
            obj_pos = torch.from_numpy(obj_pos)
            obj_rot = torch.from_numpy(obj_rot)
            obj_size = torch.from_numpy(obj_size)
            obj_attr = torch.from_numpy(obj_attr)
            adj = torch.from_numpy(adj)
        pn_feat = pn_feat.float().cuda()
        obj_pos = obj_pos.float().cuda()
        obj_rot = obj_rot.float().cuda()
        obj_size = obj_size.float().cuda()
        obj_attr = obj_attr.float().cuda()
        edge_index, edge_type = adj

        if self.use_scene_attr:
            scene_attr_embedding = self.scene_attr_net(torch.cat([obj_pos, obj_rot, obj_size, obj_attr], -1))
            # x = torch.cat([pn_feat, obj_idx, scene_attr_embedding], -1)
            x = torch.cat([pn_feat, scene_attr_embedding], -1)
        else:
            # x = torch.cat([pn_feat, obj_idx], -1)
            x = pn_feat

        x = self.conv1(x, edge_index, edge_type).relu()
        x = self.conv2(x, edge_index, edge_type).relu()
        x = self.conv3(x, edge_index, edge_type)
        return x.unsqueeze(0)


class PPO(nn.Module):
    # def __init__(self, feat_dim, use_scene_attr, learning_rate, gamma, lmbda, epsilon, eps_clip, K_epoch, use_lstm=False):
    #     super(PPO, self).__init__()
    #     self.batch_data = []
    #     self.learning_rate = learning_rate
    #     self.gamma = gamma
    #     self.lmbda = lmbda
    #     self.epsilon = epsilon
    #     self.eps_clip = eps_clip
    #     self.K_epoch = K_epoch
    #     self.use_lstm = use_lstm
    #     self.lstm = nn.LSTM(64, 64)

    #     self.gcn = GCN(feat_dim=feat_dim, hidden_dim=64, use_scene_attr=use_scene_attr)
    #     self.fc_pi = nn.Linear(64, 128)
    #     self.fc_v = nn.Linear(64, 1)
    #     self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    # def pi(self, pn_feat, obj_idx, obj_pos, obj_rot, obj_size, obj_attr, adj, hidden):
    #     x = self.gcn(pn_feat, obj_idx, obj_pos, obj_rot, obj_size, obj_attr, adj)
    #     x = x.mean(1)
    #     x = x.view(-1, 1, 64)
    #     x, lstm_hidden = self.lstm(x, hidden)
    #     x = self.fc_pi(x)
    #     x = x[..., :adj.shape[-1]+1]
    #     if x.shape[-1] != adj.shape[-1]+1:
    #         import ipdb; ipdb.set_trace()
    #     prob = F.softmax(x, dim=2)
    #     return prob, lstm_hidden, x

    # def v(self, pn_feat, obj_idx, obj_pos, obj_rot, obj_size, obj_attr, adj, hidden):
    #     x = self.gcn(pn_feat, obj_idx, obj_pos, obj_rot, obj_size, obj_attr, adj)
    #     x = x.mean(1)
    #     x = x.view(-1, 1, 64)
    #     x, lstm_hidden = self.lstm(x, hidden)
    #     v = self.fc_v(x)
    #     return v

    def __init__(self, feat_dim, use_scene_attr, learning_rate, gamma, lmbda, epsilon, eps_clip, K_epoch, use_lstm=False):
        super(PPO, self).__init__()
        self.batch_data = []
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lmbda = lmbda
        self.epsilon = epsilon
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.use_lstm = use_lstm
        if self.use_lstm:
            self.stop_param = Parameter(torch.Tensor(1, 1, 32))
            torch.nn.init.xavier_uniform_(self.stop_param)

        self.gcn = GCN(feat_dim=feat_dim, hidden_dim=32, use_scene_attr=use_scene_attr)

        # if self.use_lstm:
        #     self.lstm = nn.LSTM(32, 32)
        self.fc_pi = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

        self.fc_v = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def pi(self, pn_feat, obj_pos, obj_rot, obj_size, obj_attr, adj, hidden):
        x = self.gcn(pn_feat, obj_pos, obj_rot, obj_size, obj_attr, adj)
        x_mean = x.mean(1, keepdim=True)
        # if self.use_lstm:
        #     x_mean, lstm_hidden = self.lstm(x_mean, hidden)
        x = torch.cat([x, x_mean.repeat(1, x.shape[1], 1)], 2)
        if self.use_lstm:
            x_stop = torch.cat([self.stop_param.repeat(x_mean.shape[0], 1, 1), x_mean], 2)
        else:
            x_stop = torch.cat([x_mean, x_mean], 2)
        x = torch.cat([x, x_stop], 1)
        x = self.fc_pi(x).squeeze(-1)
        prob = F.softmax(x, dim=-1)
        if self.use_lstm:
            # return prob, lstm_hidden, x
            return prob, (torch.tensor([0]), torch.tensor([0])), x
        else:
            return prob, (torch.tensor([0]), torch.tensor([0])), x

    def v(self, pn_feat, obj_pos, obj_rot, obj_size, obj_attr, adj, hidden):
        x = self.gcn(pn_feat, obj_pos, obj_rot, obj_size, obj_attr, adj)
        x = x.mean(1, keepdim=True)
        # if self.use_lstm:
        #     x, lstm_hidden = self.lstm(x, hidden)
        v = self.fc_v(x)
        return v


    def make_batch(self, idx):
        pn_feats_lst, adj_lst, adj_prime_lst, s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst = [], [], [], [], [], [], [], [], [], [], []
        obj_pos_lst = []
        obj_rot_lst = []
        obj_size_lst = []
        obj_attr_lst = []

        for transition in self.batch_data[idx]:
            pn_feats, obj_pos, obj_rot, obj_size, obj_attr, adj, adj_prime, s, a, r, s_prime, prob_a, h_in, h_out, done = transition

            pn_feats_lst.append(pn_feats.float().cuda())
            obj_pos_lst.append(obj_pos.float().cuda())
            obj_rot_lst.append(obj_rot.float().cuda())
            obj_size_lst.append(obj_size.float().cuda())
            obj_attr_lst.append(obj_attr.float().cuda())
            adj_lst.append(adj.float().cuda())
            adj_prime_lst.append(adj_prime.float().cuda())
            # adj_lst.append(adj)
            # adj_prime_lst.append(adj_prime)
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        pn_feats = torch.stack(pn_feats_lst)
        adj = torch.stack(adj_lst)
        adj_prime = torch.stack(adj_prime_lst)
        s = torch.stack(s_lst)
        a = torch.tensor(a_lst)
        r = torch.tensor(r_lst)
        s_prime = torch.stack(s_prime_lst)
        done_mask = torch.tensor(done_lst)
        prob_a = torch.tensor(prob_a_lst)
        obj_pos = torch.stack(obj_pos_lst)
        obj_rot = torch.stack(obj_rot_lst)
        obj_size = torch.stack(obj_size_lst)
        obj_attr = torch.stack(obj_attr_lst)

        return pn_feats, obj_pos, obj_rot, obj_size, obj_attr, adj, adj_prime, s, a, r, s_prime, done_mask, prob_a, h_in_lst[0], h_out_lst[0]

    def train_net(self):
        for ep in range(max(0, len(self.batch_data)-30), len(self.batch_data)):
        # for ep in range(len(self.batch_data)):
            pn_feats, obj_pos, obj_rot, obj_size, obj_attr, adj, adj_prime, s, a, r, s_prime, done_mask, prob_a, (h1_in, h2_in), (h1_out, h2_out) = self.make_batch(ep)
            first_hidden = (h1_in.detach(), h2_in.detach())
            second_hidden = (h1_out.detach(), h2_out.detach())

            a = a.cuda()
            r = r.cuda().float()
            prob_a = prob_a.cuda().float()
            done_mask = done_mask.cuda().float()

            for i in range(self.K_epoch):
                v_prime = self.v(pn_feats, obj_pos, obj_rot, obj_size, obj_attr, adj_prime, second_hidden).squeeze(1)
                td_target = r + self.gamma * v_prime * done_mask
                v_s = self.v(pn_feats, obj_pos, obj_rot, obj_size, obj_attr, adj, first_hidden).squeeze(1)
                delta = td_target - v_s
                delta = delta.cpu().detach().numpy()

                advantage_lst = []
                advantage = 0.0
                for item in delta[::-1]:
                    advantage = self.gamma * self.lmbda * advantage + item[0]
                    advantage_lst.append([advantage])
                advantage_lst.reverse()
                advantage = torch.tensor(advantage_lst, dtype=torch.float).cuda()

                pi, _, _ = self.pi(pn_feats, obj_pos, obj_rot, obj_size, obj_attr, adj, first_hidden)
                pi_a = pi.squeeze(1).gather(1, a)
                ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == log(exp(a)-exp(b))

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage

                loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s, td_target.detach())

                self.optimizer.zero_grad()
                loss.mean().backward(retain_graph=True)
                self.optimizer.step()
