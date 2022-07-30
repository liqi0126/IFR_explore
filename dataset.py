import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from gym_IFR.envs.util import load_pc


class BinaryPriorDataset(Dataset):
    def __init__(self, data_path, pc_base_path, idx):
        self.data_path = data_path
        self.pc_base_path = pc_base_path
        self.dirs = os.listdir(self.data_path)
        self.idx = idx % len(self.dirs)
        self.build_scene()

    def build_scene(self):
        self.intervened = np.loadtxt(f'{self.data_path}/{self.dirs[self.idx]}/intervened.txt', dtype=bool)
        self.relation_graph = np.loadtxt(f'{self.data_path}/{self.dirs[self.idx]}/relation_graph.txt', dtype=int)
        self.pc_path = np.loadtxt(f'{self.data_path}/{self.dirs[self.idx]}/pc_path.txt', dtype=str, delimiter='\t')
        self.obj_indices = np.loadtxt(f'{self.data_path}/{self.dirs[self.idx]}/obj_indices.txt', dtype=int)
        self.obj_pos = np.loadtxt(f'{self.data_path}/{self.dirs[self.idx]}/obj_pos.txt')
        self.obj_rot = np.loadtxt(f'{self.data_path}/{self.dirs[self.idx]}/obj_rot.txt')
        self.obj_size = np.loadtxt(f'{self.data_path}/{self.dirs[self.idx]}/obj_size.txt')
        self.obj_attr = np.loadtxt(f'{self.data_path}/{self.dirs[self.idx]}/obj_attr.txt')

        self.relation_graph[~self.intervened] = -1

        self.obj_groups = []
        for pc_path in self.pc_path:
            self.obj_groups.append(load_pc(self.pc_base_path, pc_path))

    def __len__(self):
        return self.intervened.sum() * self.relation_graph.shape[1]

    def __getitem__(self, idx):
        i_indices = np.where(self.intervened)[0]
        num_part = self.relation_graph.shape[1]
        i = i_indices[idx // num_part]
        j = idx % num_part

        pc_i = self.obj_groups[i]
        pc_j = self.obj_groups[j]

        pos_i = self.obj_pos[i]
        pos_j = self.obj_pos[j]
        rot_i = self.obj_rot[i]
        rot_j = self.obj_rot[j]
        size_i = self.obj_size[i]
        size_j = self.obj_size[j]
        attr_i = self.obj_attr[i]
        attr_j = self.obj_attr[j]

        relation = self.relation_graph[i, j]

        return pc_i, pc_j, pos_i, pos_j, rot_i, rot_j, size_i, size_j, attr_i, attr_j, relation


class ScenePriorDataset(Dataset):
    def __init__(self, data_path, pc_base_path, prior_network, idx, validate=False):
        self.data_path = data_path
        self.pc_base_path = pc_base_path
        self.dirs = os.listdir(self.data_path)
        self.prior_network = prior_network
        self.dirs = os.listdir(self.data_path)
        self.idx = idx % len(self.dirs)
        self.validate = validate
        self.build_scene()

    def build_scene(self):
        self.intervened = np.loadtxt(f'{self.data_path}/{self.dirs[self.idx]}/intervened.txt', dtype=bool)
        self.relation_graph = np.loadtxt(f'{self.data_path}/{self.dirs[self.idx]}/relation_graph.txt', dtype=int)
        self.pc_path = np.loadtxt(f'{self.data_path}/{self.dirs[self.idx]}/pc_path.txt', dtype=str, delimiter='\t')
        self.obj_indices = np.loadtxt(f'{self.data_path}/{self.dirs[self.idx]}/obj_indices.txt', dtype=int)
        self.obj_pos = np.loadtxt(f'{self.data_path}/{self.dirs[self.idx]}/obj_pos.txt')
        self.obj_rot = np.loadtxt(f'{self.data_path}/{self.dirs[self.idx]}/obj_rot.txt')
        self.obj_size = np.loadtxt(f'{self.data_path}/{self.dirs[self.idx]}/obj_size.txt')
        self.obj_attr = np.loadtxt(f'{self.data_path}/{self.dirs[self.idx]}/obj_attr.txt')
        self.action = np.loadtxt(f'{self.data_path}/{self.dirs[self.idx]}/action.txt', dtype=int).reshape(-1)

        self.relation_graph[~self.intervened] = -1
        # valid_action = self.action[:-1][(self.relation_graph[self.action[:-1]].sum(-1) > 0)]
        # valid_action = np.append(valid_action, self.action[-1])
        # self.action = valid_action
        if self.validate:
            self.num = 1
        else:
            self.num = len(self.action)

        with open(f'{self.data_path}/{self.dirs[self.idx]}/scene.json', 'r') as f:
            self.scene = json.load(f)['scene']

        pc_group = []
        pos_group = []
        rot_group = []
        size_group = []
        attr_group = []
        for pc_path, pos, rot, size, attr in zip(self.pc_path, self.obj_pos, self.obj_rot, self.obj_size, self.obj_attr):
            pc = load_pc(self.pc_base_path, pc_path)
            pc_group.append(pc)
            pos_group.append(pos)
            rot_group.append(rot)
            size_group.append(size)
            attr_group.append(attr)
        self.pos_group = np.stack(pos_group)
        self.rot_group = np.stack(rot_group)
        self.size_group = np.stack(size_group)
        self.attr_group = np.stack(attr_group)

        self.prior_network.eval()
        with torch.no_grad():
            self.prior_graph = self.prior_network.build_prior(pc_group, pos_group, rot_group, size_group, attr_group).cpu().numpy()
            self.pn_feat = self.prior_network.get_feat(torch.stack(pc_group))[-1].cpu().numpy()

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        # x = np.where(self.relation_graph.sum(-1) > 0)[0]
        # x = random.sample(list(x), idx)
        know_graph = -np.ones_like(self.relation_graph)
        know_graph[self.action[:idx]] = self.relation_graph[self.action[:idx]]
        return self.pn_feat, self.pos_group, self.rot_group, self.size_group, self.attr_group, \
               self.prior_graph, know_graph, self.relation_graph


if __name__ == '__main__':
    from model.binary_relation_net import RelationNetwork

    prior_net = RelationNetwork()
    prior_net.cuda()
    prior_net.eval()

    data_path = 'random_scenes/kicthen_budget_2000'
    pc_path = "data/pc_data"
    dataset = ScenePriorDataset(data_path, pc_path, prior_net, 25)
    for i in range(10):
        dataset.__getitem__(i)
