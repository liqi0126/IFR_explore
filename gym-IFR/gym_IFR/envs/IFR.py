import os
import json
import numpy as np
import random
from copy import deepcopy
import torch

import gym
from gym import spaces
from gym.utils import seeding


from .relation import *
from .objects import OBJ_NUM
from .util import load_pc


class IFR(gym.Env):
    def __init__(self, data_path="pc_data", ai2thor_scene_base_dirs=None, train=True, use_scene_attr=True, fix_size=-1, prior_network=None, pred_network=None, scene_dir=None, target='learning', init='prior', prior_thres=0.2, pred_thres=0.7, percent=1, total_budget=None):
        self.data_path = data_path
        self.use_scene_attr = use_scene_attr
        self.prior_network = prior_network
        self.pred_network = pred_network
        if self.pred_network is not None:
            self.pred_network.eval()
        self.train = train
        self.fix_size = fix_size
        self.total_budget = total_budget
        self.total_step = 0
        self.scene_dir = scene_dir
        self.target = target
        self.init = init
        self.prior_thres = prior_thres
        self.pred_thres = pred_thres
        self.percent = percent
        self.scene_idx = -1
        ai2thor_dir = []
        max_value = 0
        for base_dir in ai2thor_scene_base_dirs:
            for path in os.listdir(base_dir):
                ai2thor_dir.append(os.path.join(base_dir, path))
                max_value = max(max_value, int(path[-1]))
        # reorganize
        self.ai2thor_dir = []
        for i in range(max_value):
            for path in ai2thor_dir:
                if f'_{i}' in path:
                    self.ai2thor_dir.append(path)

        self.seed()

    def add_obj_to_group(self, OBJ, name, pos, rot, size, attr):
        obj = OBJ(name=name)
        obj.name = name
        obj.obj_idx = len(self.obj_groups)
        if obj.sapien:
            idx = random.choice(obj.sapien_id)
            if isinstance(idx, tuple):
                idx = idx[0]
            pc_path = f"{OBJ.__name__}_{idx}.xyz"
        else:
            path = random.choice(obj.ai2thor_id)
            pc_path = f"{path}.xyz"
        pc = load_pc(self.data_path, pc_path)
        obj.pc = pc
        obj.pc_path = pc_path
        obj.pos = pos
        obj.rot = rot
        obj.size = size
        obj.attr = attr
        self.obj_groups.append(obj)
        return obj


    def build_fix_size(self, obs, obj_feats, action, pred_graph):
        if self.fix_size > 0:
            obs = np.concatenate([obs, -np.ones(self.fix_size - len(obs))])
            obj_feats = np.concatenate([obj_feats, -np.ones((self.fix_size - obj_feats.shape[0], ) + obj_feats.shape[1:])])
            action = np.concatenate([action, -np.ones(self.fix_size - len(action))])
            temp = deepcopy(pred_graph)
            pred_graph = -np.ones((self.fix_size, self.fix_size))
            pred_graph[:self.obj_num, :self.obj_num] = temp
        return obs, obj_feats, action, pred_graph

    def reset(self):
        self.action_list = []
        self.obj_groups = []
        relation_list = []

        self.scene_idx += 1
        self.scene = self.ai2thor_dir[self.scene_idx % len(self.ai2thor_dir)]
        # self.idx = 0
        # scene = self.ai2thor_dir[self.idx % len(self.ai2thor_dir)]
        scene_names = np.loadtxt(f'{self.scene}/name.txt', dtype='str')
        scene_pos = np.loadtxt(f'{self.scene}/pos.txt')
        scene_rot = np.loadtxt(f'{self.scene}/rot.txt')
        scene_size = np.loadtxt(f'{self.scene}/size.txt')
        scene_attr = np.loadtxt(f'{self.scene}/attr.txt')

        counter_1to1 = {}
        # add manual relationship
        for n, p, r, s, a in zip(scene_names, scene_pos, scene_rot, scene_size, scene_attr):
            if n in AI2Thor_BINARY_RELATION_1_TO_1_MANUAL_SRC_KEY or \
                n in AI2Thor_BINARY_RELATION_1_TO_1_MANUAL_DST_KEY or \
                n in AI2Thor_BINARY_RELATION_1_TO_1_SRC_KEY or \
                n in AI2Thor_BINARY_RELATION_1_TO_1_DST_KEY:
                if n not in counter_1to1:
                    counter_1to1[n] = 0
                else:
                    counter_1to1[n] += 1
                name = f'{n}#{counter_1to1[n]}'
            else:
                name = n

            OBJ = CORRESPONDENCE_DICT[n]
            self.add_obj_to_group(OBJ, name=name, pos=p, rot=r, size=s, attr=a)

            if n in AI2Thor_BINARY_RELATION_1_TO_1_MANUAL_DST_KEY:
                SRC = CORRESPONDENCE_DICT[AI2Thor_BINARY_RELATION_1_TO_1_MANUAL_DST_DICT[n]]
                src_name = f'{AI2Thor_BINARY_RELATION_1_TO_1_MANUAL_DST_DICT[n]}#{counter_1to1[n]}'
                if self.train:
                    self.add_obj_to_group(SRC, name=src_name, pos=p + np.random.randn(3) * 0.01, rot=r, size=s, attr=a)
                else:
                    self.add_obj_to_group(SRC, name=src_name, pos=p, rot=r, size=s, attr=a)

        # build relationship
        for i, obj in enumerate(self.obj_groups):
            if obj.name in AI2Thor_SELF_RELATION_KEY: # self relation
                relation_list.append((i, i))

            if obj.name in AI2Thor_BINARY_RELATION_1_TO_ALL_SRC_KEY: # 1 to all
                for j, dst_obj in enumerate(self.obj_groups):
                    if dst_obj.name in AI2Thor_BINARY_RELATION_1_TO_ALL_SRC_DICT[obj.name]:
                        relation_list.append((i, j))

            if obj.name.split('#')[0] in AI2Thor_BINARY_RELATION_1_TO_1_MANUAL_SRC_KEY:
                for j, dst_obj in enumerate(self.obj_groups):
                    if dst_obj.name.split('#')[0] == AI2Thor_BINARY_RELATION_1_TO_1_MANUAL_SRC_DICT[obj.name.split('#')[0]] and obj.name.split('#')[1] == dst_obj.name.split('#')[1]:
                        relation_list.append((i, j))

            if obj.name.split('#')[0] in AI2Thor_BINARY_RELATION_1_TO_1_SRC_KEY:
                for j, dst_obj in enumerate(self.obj_groups):
                    if dst_obj.name.split('#')[0] == AI2Thor_BINARY_RELATION_1_TO_1_SRC_DICT[obj.name.split('#')[0]] and obj.name.split('#')[1] == dst_obj.name.split('#')[1]:
                        relation_list.append((i, j))

        self.obj_num = len(self.obj_groups)
        self.intervened = torch.zeros(self.obj_num, dtype=bool)
        self.relation_graph = torch.zeros((self.obj_num, self.obj_num), dtype=int)
        for x, y in relation_list:
            self.relation_graph[x, y] = 1
        #     print(self.obj_groups[x].name, self.obj_groups[y].name)

        # print()
        # for obj in self.obj_groups:
        #     print(obj.name, obj.pc_path)
        # import ipdb; ipdb.set_trace()

        # if self.target == 'learning' and self.total_budget is None:
        if self.percent < 1:
            self.action_space = spaces.Discrete(self.obj_num)
        else:
            self.action_space = spaces.Discrete(self.obj_num + 1)

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
            self.obj_feats = self.pn_feat.float()

        obs = torch.zeros(self.obj_num)
        return obs, self.pn_feat, self.obj_pos, self.obj_rot, self.obj_size, self.obj_attr, self.pred_graph(type_pos=True)

    def onehot_action(self, action):
        onehot = np.zeros(self.action_space.n)
        if action < self.action_space.n:
            onehot[action] = 1
        return onehot

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # def pred_graph(self, type_pos=False):
    #     pred_graph = deepcopy(self.prior_graph)
    #     pred_graph[self.intervened] = self.relation_graph[self.intervened].float()
    #     return pred_graph

    def pred_graph(self, type_pos=False):
        if self.pred_network is not None:
            self.pred_network.eval()
            know_graph = -torch.ones_like(self.relation_graph)
            know_graph[self.intervened] = self.relation_graph[self.intervened]
            know_graph = know_graph.float()
            final_graph = self.pred_network.build_gcn_graph(self.obj_pos.unsqueeze(0),
                                                            self.prior_graph.unsqueeze(0),
                                                            know_graph.unsqueeze(0), self.prior_thres).float()
            with torch.no_grad():
                z = self.pred_network.encode(self.obj_feats.cuda(), final_graph.cuda())
                pred_graph = self.pred_network.decoder.forward_all(z, self.scene_attr.cuda(), final_graph.cuda()).cpu()

            if type_pos:
                pred_graph = self.pred_network.build_gcn_graph(self.obj_pos.unsqueeze(0),
                                                               pred_graph.unsqueeze(0),
                                                               know_graph.unsqueeze(0), self.prior_thres, sym=True).squeeze()
            else:
                pred_graph[self.intervened] = self.relation_graph[self.intervened].float()
            # hack for RGCN
            # if type_pos:
            #     edge_index, edge_type = self.pred_network.build_gcn_graph_discrete(self.onehot_idx.unsqueeze(0),
            #                                                     self.obj_pos.unsqueeze(0),
            #                                                     pred_graph.unsqueeze(0),
            #                                                     know_graph.unsqueeze(0), self.prior_thres)
        else:
            pred_graph = deepcopy(self.relation_graph)

        # if type_pos:
        #     return edge_index, edge_type
        # else:
        # pred_graph[67, 50] = 0
        # pred_graph[61] = 0
        # pred_graph[61, 5] = 1
        # return self.prior_graph
        return pred_graph


    def get_obs(self, idx):
        if idx < self.relation_graph.shape[0]:
            return self.relation_graph[idx]
        else:
            return torch.zeros(self.relation_graph.shape[0]).int()

    def set_scene_dir(self, scene_dir):
        self.scene_dir = scene_dir

    def save_scene(self):
        if not os.path.exists(self.scene_dir):
            os.mkdir(self.scene_dir)

        out_dir = f"{self.scene.split('/')[-1]}_{self.scene_idx // len(self.ai2thor_dir)}"
        os.mkdir(f"{self.scene_dir}/{out_dir}")

        pc_path = []
        obj_indices = []
        obj_pos = []
        obj_rot = []
        obj_size = []
        obj_attr = []
        for obj in self.obj_groups:
            pc_path.append(obj.pc_path)
            obj_indices.append(obj.idx)
            obj_pos.append(obj.pos)
            obj_rot.append(obj.rot)
            obj_size.append(obj.size)
            obj_attr.append(obj.attr)
        pc_path = np.array(pc_path)
        obj_indices = np.array(obj_indices)
        obj_pos = np.array(obj_pos)
        obj_rot = np.array(obj_rot)
        obj_size = np.array(obj_size)
        obj_attr = np.array(obj_attr)
        np.savetxt(f'{self.scene_dir}/{out_dir}/intervened.txt', self.intervened.cpu(), fmt='%d')
        np.savetxt(f'{self.scene_dir}/{out_dir}/relation_graph.txt', self.relation_graph.cpu(), fmt='%d')
        np.savetxt(f'{self.scene_dir}/{out_dir}/pc_path.txt', pc_path, fmt='%s')
        np.savetxt(f'{self.scene_dir}/{out_dir}/obj_indices.txt', obj_indices, fmt='%d')
        np.savetxt(f'{self.scene_dir}/{out_dir}/obj_pos.txt', obj_pos)
        np.savetxt(f'{self.scene_dir}/{out_dir}/obj_rot.txt', obj_rot)
        np.savetxt(f'{self.scene_dir}/{out_dir}/obj_size.txt', obj_size)
        np.savetxt(f'{self.scene_dir}/{out_dir}/obj_attr.txt', obj_attr)
        np.savetxt(f'{self.scene_dir}/{out_dir}/action.txt', self.action_list, fmt='%d')

        with open(f'{self.scene_dir}/{out_dir}/scene.json', 'w') as f:
            json.dump({'scene': self.scene}, f)

    def learning_step(self, action):
        done = self.taken_step >= self.max_step
        if self.total_budget is not None:
            if action not in self.action_list and action < self.obj_num:
                self.total_step += 1
            if self.total_step >= self.total_budget:
                reward = 1
                done = True

        repeat = False
        if action in self.action_list:
            reward = 0
            repeat = True
        else:
            if action == self.obj_num:
                done = True
                reward = 0
            else:
                reward = 2. * torch.abs((self.pred_graph(type_pos=False) - self.relation_graph)[action]).max() - 1
                reward += 1. * self.relation_graph[action].any().float()
                # reward = 5 * torch.abs((self.pred_graph(type_pos=False) - self.relation_graph)[action]).max()
                reward = reward.item()
                self.intervened[action] = True
                self.taken_step += 1
            self.action_list.append(action)

        if done:
            self.save_scene()

        if done:
            pred = (self.pred_graph(type_pos=False) > self.pred_thres).int()
            gt = self.relation_graph
            tp = (pred & gt).sum()
            tp_fp = pred.sum()
            tp_fn = gt.sum()
            precision = tp / (tp_fp + 1e-8)
            recall = tp / (tp_fn + 1e-8)
            f1 = (2 * precision * recall) / (precision + recall + 1e-8)
        else:
            precision = 0
            recall = 0
            f1 = 0

        obs = self.get_obs(action)

        if self.total_budget is not None:
            useup = self.total_step >= self.total_budget
        else:
            useup = False

        return obs, reward, done, {"pn_feat": self.pn_feat, "obj_pos": self.obj_pos, "graph": self.pred_graph(type_pos=True), "useup": useup, "repeat": repeat, "precision": precision, "recall": recall, "f1": f1, "step": self.intervened.sum()}

    def exploration_step(self, action):
        done = self.taken_step >= self.max_step
        if self.total_budget is not None:
            if action not in self.action_list and action < self.obj_num:
                self.total_step += 1
            if self.total_step >= self.total_budget:
                reward = 1
                done = True

        repeat = False
        if action in self.action_list:
            reward = 0
            repeat = True
        else:
            if action == self.obj_num:
                done = True
                # reward = -1
                reward = 0
            else:
                # reward = 2. * torch.abs((self.pred_graph(type_pos=False) - self.relation_graph)[action]).max()
                reward = 2. * ((self.pred_graph(type_pos=False)[action] > self.pred_thres) != self.relation_graph[action]).any()
                reward += 1. * self.relation_graph[action].any().float()
                # reward -= 1
                # reward = 100 * np.abs((self.pred_graph(type_pos=False) - self.relation_graph)[action]).max()
                # reward = 10 * self.relation_graph[action].any() - 1
                # reward = 5 * torch.abs((self.pred_graph(type_pos=False) - self.relation_graph)[action]).max() - 1
                # reward = 10 * ((self.pred_graph()[action] > self.pred_thres) != self.relation_graph[action]).any() - 1
                reward = reward.item()
                self.intervened[action] = True
                self.taken_step += 1
            self.action_list.append(action)
        # if done:
        #     self.save_scene()

        if done:
            pred = (self.pred_graph(type_pos=False) > self.pred_thres).int()
            gt = self.relation_graph
            tp = (pred & gt).sum()
            tp_fp = pred.sum()
            tp_fn = gt.sum()
            precision = tp / (tp_fp + 1e-8)
            recall = tp / (tp_fn + 1e-8)
            f1 = (2 * precision * recall) / (precision + recall + 1e-8)
        else:
            precision = 0
            recall = 0
            f1 = 0

        obs = self.get_obs(action)

        if self.total_budget is not None:
            useup = self.total_step >= self.total_budget
        else:
            useup = False

        return obs, reward, done, {"pn_feat": self.pn_feat,  "obj_pos": self.obj_pos, "graph": self.pred_graph(type_pos=True), "useup": useup, "repeat": repeat, "precision": precision, "recall": recall, "f1": f1, "step": self.intervened.sum()}

    def get_pr(self):
        pred = (self.pred_graph(type_pos=False) > self.pred_thres).int()
        gt = self.relation_graph
        tp = (pred & gt).sum()
        tp_fp = pred.sum()
        tp_fn = gt.sum()
        precision = (tp / (tp_fp + 1e-8)).item()
        recall = (tp / (tp_fn + 1e-8)).item()
        f1 = (2 * precision * recall) / (precision + recall + 1e-8)
        return precision, recall, f1

    def exploration_evaluation_step(self, action):
        done = self.taken_step >= self.max_step
        repeat = False
        if action in self.action_list:
            repeat = True
        else:
            if action == self.obj_num:
                done = True
            else:
                self.taken_step += 1
                self.action_list.append(action)
                self.intervened[action] = True

        precision, recall, f1 = self.get_pr()

        obs = self.get_obs(action)
        return obs, 0, done, {"pn_feat": self.pn_feat, "obj_pos": self.obj_pos, "graph": self.pred_graph(type_pos=True), "repeat": repeat, "precision": precision, "recall": recall, "f1": f1, "step": self.intervened.sum()}


    def step(self, action):
        if self.target == 'learning':
            return self.learning_step(action)
        elif self.target == 'exploration':
            return self.exploration_step(action)
        elif self.target == 'exploration_eval':
            return self.exploration_evaluation_step(action)
        else:
            raise NotImplementedError


    def graph_reward(self, pred_graph):
        relation_mask = self.relation_graph.sum(-1) > 0
        reward = 2 * (pred_graph[relation_mask] == self.relation_graph[relation_mask]).all(-1).sum()
        reward -= 1 * (pred_graph != self.relation_graph).any(-1).sum()
        return reward

    def close(self):
        pass

