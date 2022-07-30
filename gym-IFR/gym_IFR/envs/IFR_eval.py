# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import torch

from gym import spaces

from .objects import *
from .util import load_pc
from .IFR import IFR


class IFREval(IFR):
    def __init__(self, data_path, eval_path, use_scene_attr, prior_network, pred_network, percent, prior_thres, pred_thres):
        self.data_path = data_path
        self.eval_path = eval_path
        self.percent = percent
        self.use_scene_attr = use_scene_attr
        self.pred_thres = pred_thres
        self.prior_thres = prior_thres

        self.dirs = []
        for path in os.listdir(eval_path):
            self.dirs.append(path)
        self.idx = -1

        self.prior_network = prior_network
        self.pred_network = pred_network
        if self.pred_network is not None:
            self.pred_network.eval()


    def add_obj_to_group(self, OBJ, pc_path, name, pos, rot, size, attr):
        obj = OBJ(name=name)
        obj.name = name
        obj.obj_idx = len(self.obj_groups)
        pc = load_pc(self.data_path, pc_path)
        obj.pc = pc
        obj.pc_path = pc_path
        obj.pos = pos
        obj.rot = rot
        obj.size = size
        obj.attr = attr
        self.obj_groups.append(obj)
        return obj

    def set_index(self, idx):
        self.idx = idx

    def reset(self):
        self.obj_groups = []
        self.action_list = []
        self.idx += 1

        with open(f'{self.eval_path}/{self.dirs[self.idx]}/scene.json', 'r') as f:
            self.scene = json.load(f)['scene']

        intervened = np.loadtxt(f'{self.eval_path}/{self.dirs[self.idx]}/intervened.txt', dtype=bool)
        relation_graph = np.loadtxt(f'{self.eval_path}/{self.dirs[self.idx]}/relation_graph.txt', dtype=int)
        pc_path = np.loadtxt(f'{self.eval_path}/{self.dirs[self.idx]}/pc_path.txt', dtype=str, delimiter='\t')
        obj_indices = np.loadtxt(f'{self.eval_path}/{self.dirs[self.idx]}/obj_indices.txt', dtype=int)
        obj_pos = np.loadtxt(f'{self.eval_path}/{self.dirs[self.idx]}/obj_pos.txt')
        obj_rot = np.loadtxt(f'{self.eval_path}/{self.dirs[self.idx]}/obj_rot.txt')
        obj_size = np.loadtxt(f'{self.eval_path}/{self.dirs[self.idx]}/obj_size.txt')
        obj_attr = np.loadtxt(f'{self.eval_path}/{self.dirs[self.idx]}/obj_attr.txt')

        self.obj_names = []
        for i, (idx, path, p, r, s, a) in enumerate(zip(obj_indices, pc_path, obj_pos, obj_rot, obj_size, obj_attr)):
            OBJ = InDoorObject.__subclasses__()[idx]
            name = f'{i}: {OBJ.__name__}'
            self.add_obj_to_group(OBJ, pc_path=path, name=name, pos=p, rot=r, size=s, attr=a)
            self.obj_names.append(name)

        self.obj_num = len(self.obj_groups)
        self.intervened = torch.zeros(self.obj_num, dtype=bool)
        self.relation_graph = torch.tensor(relation_graph)

        self.observation_space = spaces.MultiBinary(self.obj_num)
        self.max_step = int((self.obj_num + 1) * self.percent)
        self.taken_step = 0

        pc_group = []
        pos_group = []
        rot_group = []
        size_group = []
        attr_group = []
        for obj in self.obj_groups:
            pc_group.append(obj.pc)
            pos_group.append(obj.pos)
            rot_group.append(obj.rot)
            size_group.append(obj.size)
            attr_group.append(obj.attr)
        self.obj_pos = torch.tensor(np.stack(pos_group))
        self.obj_rot = torch.tensor(np.stack(rot_group))
        self.obj_size = torch.tensor(np.stack(size_group))
        self.obj_attr = torch.tensor(np.stack(attr_group))
        self.scene_attr = torch.cat([self.obj_pos, self.obj_rot, self.obj_size, self.obj_attr], -1)

        if self.prior_network is not None:
            self.prior_network.eval()
            self.pn_feat = self.prior_network.get_feat(torch.stack(pc_group))[-1].cpu()
            self.prior_graph = self.prior_network.build_prior(pc_group, pos_group, rot_group, size_group, attr_group).cpu()
        else:
            self.pn_feat = None

        if self.pn_feat is None:
            self.obj_feats = None
        elif self.use_scene_attr:
            with torch.no_grad():
                self.scene_attr_embedding = self.pred_network.scene_attr_net(torch.cat([self.obj_pos, self.obj_rot, self.obj_size, self.obj_attr], -1).float().cuda()).cpu()
            self.obj_feats = torch.cat([self.pn_feat, self.scene_attr_embedding], -1).float()
        else:
            # self.obj_feats = torch.cat([self.pn_feat, self.onehot_idx], -1).float()
            self.obj_feats = self.pn_feat.float()

        obs = torch.zeros(self.obj_num)
        return obs, self.pn_feat, self.obj_pos, self.obj_rot, self.obj_size, self.obj_attr, self.pred_graph(type_pos=True)

    def step(self, action):
        return self.exploration_evaluation_step(action)

