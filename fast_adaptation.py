import json
import os
import matplotlib.pyplot as plt

import argparse
import shutil

import gym
import numpy as np
import torch
import trimesh
# from PIL.Image import Image
from PIL import Image

from gym_IFR.envs.objects import InDoorObject

from agent.util import sample_action
from model.GCN import GCNEncoder
from model.binary_relation_net import BinaryRelationNetwork
from model.scene_relation_net import LinearLayerDecoder, MyGAE
from util import check_dir, set_index
from viz.mesh_render import mesh_to_img
from viz.scene_visual import render_relation_in_scene


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz_path', type=str, default=None)
    parser.add_argument('--init', type=str, default="prior")
    parser.add_argument('--policy', type=str, default="most_uncertain")
    parser.add_argument('--percent', type=float, default=0.2)
    parser.add_argument('--uncertainty_thres', type=float, default=0.05)
    parser.add_argument('--prior_thres', type=float, default=0.2)
    parser.add_argument('--use_scene_attr', action='store_true', default=False)
    parser.add_argument('--pred_thres', type=float, default=0.9)
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--prior_path', type=str, default=None)
    parser.add_argument('--pred_path', type=str, default=None)
    parser.add_argument('--data_path', type=str, default='data/pc_data')
    parser.add_argument('--eval_path', type=str, default="random_scenes/kitchen_eval_100_scenes")
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


args = get_args()


def adapt_test(env, policy, pred_thres, uncertainty_thres, early_stop, viz_base_path=None):
    reward_lst = []
    precision_lst = []
    recall_lst = []
    f1_lst = []
    step_lst = []
    for epoch in range(len(env.dirs)):
        s, pn_feats, obj_pos, obj_rot, obj_size, obj_attr, adj = env.reset()
        done = False

        if viz_base_path is not None:
            out_dir = env.dirs[env.idx % len(env.dirs)]
            viz_path = os.path.join('../interaction_viz', viz_base_path, out_dir)
            if os.path.exists(viz_path):
                shutil.rmtree(viz_path)
            os.makedirs(viz_path)

            np.savetxt(f'{viz_path}/obj_pos.txt', obj_pos)
            np.savetxt(f'{viz_path}/obj_rot.txt', obj_rot)
            np.savetxt(f'{viz_path}/obj_attr.txt', obj_attr)
            np.savetxt(f'{viz_path}/gt.txt', set_index(env.relation_graph.squeeze().cpu().numpy().astype('int')), '%2i')

            # scene_path = env.scene
            scene_path = "data/" + env.scene
            with open(f'{scene_path}/frame_meta.json', 'r') as f:
                frame_meta = json.load(f)
            frame = Image.open(f"{scene_path}/frame.png")

            # for i, obj in enumerate(env.obj_groups):
            #     pc_path = obj.pc_path
            #     inst_idx = obj.idx
            #     inst_pos = obj.pos
            #     fig = plt.figure(figsize=(4, 4))
            #     inst_mesh = trimesh.load(os.path.join(f'mesh_data/{pc_path[:-4]}.obj'))
            #     img = mesh_to_img(inst_mesh)
            #     plt.imshow(img)
            #     ax = plt.gca()
            #     ax.set_title(f"{i}: {InDoorObject.__subclasses__()[inst_idx].__name__}\n{inst_pos[0]:.3f}\n{inst_pos[1]:.3f}\n{inst_pos[2]:.3f}", pad=5)
            #     ax.axes.xaxis.set_visible(False)
            #     ax.axes.yaxis.set_visible(False)

            #     fig.savefig(f'{viz_path}/obj_{i}.png', bbox_inches='tight')
            #     plt.close()
            #     if i == 0:
            #         with open(f'{viz_path}/obj.txt', 'w') as f:
            #             f.write(f"{i}: {InDoorObject.__subclasses__()[inst_idx].__name__}\t{pc_path}\t{inst_pos[0]:.3f} {inst_pos[1]:.3f} {inst_pos[2]:.3f}\n")
            #     else:
            #         with open(f'{viz_path}/obj.txt', 'a') as f:
            #             f.write(f"{i}: {InDoorObject.__subclasses__()[inst_idx].__name__}\t{pc_path}\t{inst_pos[0]:.3f} {inst_pos[1]:.3f} {inst_pos[2]:.3f}\n")

            #     # plt.tight_layout()
            #     plt.close(fig)

        action_list = []
        p_r_f1_list = []
        a = None
        step_idx = -1
        while not done:
            step_idx += 1
            if viz_base_path is not None:
                pred = env.pred_graph()
                gt_graph = env.relation_graph
                know_graph = -torch.ones_like(gt_graph)
                know_graph[env.intervened] = gt_graph[env.intervened]
                known = know_graph.squeeze().cpu().numpy().astype('int')
                pred_oh = (pred.cpu().numpy() > pred_thres).astype('int')
                gt = gt_graph.squeeze().cpu().numpy().astype('int')
                np.savetxt(f'{viz_path}/know_{step_idx}.txt', set_index(know_graph.squeeze().cpu().numpy().astype('int')), '%2i')
                np.savetxt(f'{viz_path}/scene_{step_idx}.txt', set_index(pred.cpu().numpy()), '%2.3f')
                np.savetxt(f'{viz_path}/scene_oh_{step_idx}.txt', set_index((pred.cpu().numpy() > pred_thres).astype('int')), '%2i')
                p, r, f1 = env.get_pr()
                p_r_f1_list.append(np.array([p, r, f1]))

                render_relation_in_scene(frame_meta, frame, action_list, env.obj_pos, gt, pred_oh, known, None, f'{viz_path}/viz_{step_idx}.png')

                if a is not None:
                    if a < len(env.obj_groups):
                        inst_idx = env.obj_groups[a].idx
                        pc_path = env.obj_groups[a].pc_path
                        inst_pos = env.obj_groups[a].pos
                        log = f"{a}: {InDoorObject.__subclasses__()[inst_idx].__name__}\t{pc_path}\t{inst_pos[0]:.3f} {inst_pos[1]:.3f} {inst_pos[2]:.3f}\n"
                    else:
                        log = "finish"
                    if step_idx == 0:
                        with open(f'{viz_path}/action.txt', 'w') as f:
                            f.write(log)
                    else:
                        with open(f'{viz_path}/action.txt', 'a') as f:
                            f.write(log)

            pred_graph = env.pred_graph(type_pos=False)
            pred_graph[env.intervened] = env.relation_graph[env.intervened].float()
            prob = torch.zeros(pred_graph.shape[0])

            a = sample_action(prob, pred_graph, pred_thres, uncertainty_thres, policy, early_stop)

            s_prime, r, done, info = env.step(a)

            idx = 0
            while info['repeat']:
                pred_graph = env.pred_graph(type_pos=False)
                pred_graph[env.intervened] = env.relation_graph[env.intervened].float()
                a = sample_action(prob, pred_graph, pred_thres, uncertainty_thres, policy, early_stop)
                s_prime, r, done, info = env.step(a)
                idx += 1
                if idx > 20:
                    a = len(prob) - 1
                    break

            action_list.append(a)
            reward_lst.append(r)

            if done:
                precision = info['precision']
                recall = info['recall']
                f1 = info['f1']
                step = info['step']
                print(f'Epoch(Val) {epoch}:\tPrecision: {precision}\tRecall: {recall}\tF1: {f1}\tStep: {step}')
                if viz_base_path is not None:
                    p_r_f1_list = np.stack(p_r_f1_list)
                    np.savetxt(f'{viz_path}/performance.txt', p_r_f1_list, '%2.3f')
                precision_lst.append(precision)
                recall_lst.append(recall)
                f1_lst.append(f1)
                step_lst.append(step)
            # if viz:

    env.close()
    return np.mean(precision_lst), np.mean(recall_lst), np.mean(f1_lst), np.mean(step_lst)


if __name__ == '__main__':
    print_freq = 10
    if args.use_scene_attr:
        feat_dim = 64 + 64
    else:
        feat_dim = 64
    out_channels = 32
    scene_attr_channels = 0
    middle_channels = 32

    pc_path = "data/pc_data"

    prior_network = BinaryRelationNetwork(use_scene_attr=False, thres=args.prior_thres)
    if args.prior_path is not None:
        prior_network.load_state_dict(torch.load(args.prior_path)['state_dict'])
    prior_network.cuda()

    decoder = LinearLayerDecoder(out_channels + scene_attr_channels, middle_channels)
    pred_network = MyGAE(GCNEncoder(feat_dim, out_channels), decoder=decoder, init=args.init)
    if args.pred_path is not None:
        pred_network.load_state_dict(torch.load(args.pred_path)['state_dict'])
    pred_network.cuda()

    prior_network.eval()
    pred_network.eval()

    test_env = gym.make('gym_IFR:IFR_eval-v0',
                        data_path=args.data_path,
                        eval_path=args.eval_path,
                        use_scene_attr=args.use_scene_attr,
                        prior_network=prior_network,
                        pred_network=pred_network,
                        prior_thres=args.prior_thres,
                        percent=args.percent,
                        pred_thres=args.pred_thres)

    precision, recall, f1, step = adapt_test(test_env, args.policy, args.pred_thres, args.uncertainty_thres, args.percent == 1, viz_base_path=args.viz_path)

    state = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'step': step,
    }

    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'f1: {f1}')
    print(f'step: {step}')

