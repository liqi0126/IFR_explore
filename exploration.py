# PPO-LSTM
import os
import shutil
import argparse
from util import check_dir

import gym
import torch

from model.binary_relation_net import BinaryRelationNetwork
from model.GCN import GCNEncoder
from model.scene_relation_net import LinearLayerDecoder, MyGAE

from binary_prior_learning import train_binary_prior
from scene_prior_learning import train_scene_prior
from dataset import BinaryPriorDataset, ScenePriorDataset

from agent.PPO import PPO
from agent.util import explore


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--budget', type=int, default=5000)
    parser.add_argument('--prior_thres', type=float, default=0.2)
    parser.add_argument('--pred_thres', type=float, default=0.5)
    parser.add_argument('--init', type=str, default="prior")
    parser.add_argument('--use_scene_attr', action='store_true', default=False)
    parser.add_argument('--init_pn', type=str, default='pretrained_pn/net_network.pth')
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--data_path', type=str, default='data/pc_data')
    parser.add_argument('--prior_path', type=str, default=None)
    parser.add_argument('--pred_path', type=str, default=None)
    parser.add_argument('--rl_path', type=str, default=None)
    parser.add_argument('--bedroom', action='store_true', default=False)
    parser.add_argument('--bathroom', action='store_true', default=False)
    parser.add_argument('--kitchen', action='store_true', default=False)
    parser.add_argument('--living_room', action='store_true', default=False)
    parser.add_argument('--viz', action='store_true', default=False)

    parser.add_argument('--learning_rate', type=float, default=0.00001)
    parser.add_argument('--gamma', type=float, default=0.0)
    parser.add_argument('--lmbda', type=float, default=0.95)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--eps_clip', type=float, default=0.1)
    parser.add_argument('--K_epoch', type=int, default=10)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


args = get_args()

torch.set_printoptions(sci_mode=False)

if __name__ == '__main__':
    base_dir = os.path.join("output/interaction_learning/", args.output_dir)
    save_dir = os.path.join(base_dir, "checkpoints")

    pc_base_path = "data/pc_data"

    print_freq = 10
    out_channels = 32

    use_cuda = True
    scheduler = None
    summary_writer = None
    variational = False

    prior_network = BinaryRelationNetwork(use_scene_attr=False, thres=args.prior_thres)
    state_dict = torch.load(args.init_pn)
    prior_network.load_state_dict(state_dict, strict=False)
    if args.prior_path:
        prior_network.load_state_dict(torch.load(args.prior_path)['state_dict'])
    prior_network.cuda()

    if args.use_scene_attr:
        # feat_dim = OBJ_NUM + 64 + 64
        feat_dim = 64 + 64
    else:
        # feat_dim = OBJ_NUM + 64
        feat_dim = 64

    out_channels = 32
    # scene_attr_channels = 22
    scene_attr_channels = 0
    middle_channels = 32
    decoder = LinearLayerDecoder(out_channels + scene_attr_channels, middle_channels)
    pred_network = MyGAE(GCNEncoder(feat_dim, out_channels), decoder=decoder, init=args.init)
    if args.pred_path:
        pred_network.load_state_dict(torch.load(args.pred_path)['state_dict'])
    pred_network.cuda()

    prior_optim = torch.optim.Adam(
        prior_network.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.0001
    )

    pred_optim = torch.optim.Adam(
        pred_network.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.0001
    )

    loop = 5
    binary_f1_lst = []
    scene_f1_lst = []
    base_scene_dir = os.path.join('interact_scenes', args.output_dir)
    if os.path.exists(base_scene_dir):
        shutil.rmtree(base_scene_dir)
    os.mkdir(base_scene_dir)

    ai2thor_scene_base_dirs = []
    if args.bedroom:
        ai2thor_scene_base_dirs.append('data/ai2thor_scenes/BedRoom/train')
    if args.bathroom:
        ai2thor_scene_base_dirs.append('data/ai2thor_scenes/BathRoom/train')
    if args.kitchen:
        ai2thor_scene_base_dirs.append('data/ai2thor_scenes/Kitchen/train')
    if args.living_room:
        ai2thor_scene_base_dirs.append('data/ai2thor_scenes/LivingRoom/train')

    env = gym.make('gym_IFR:IFR-v0',
                   data_path=args.data_path,
                   ai2thor_scene_base_dirs=ai2thor_scene_base_dirs,
                   use_scene_attr=args.use_scene_attr,
                   prior_network=prior_network,
                   pred_network=pred_network,
                   train=not args.eval,
                   target='learning',
                   scene_dir=base_scene_dir,
                   init=args.init,
                   prior_thres=args.prior_thres,
                   pred_thres=args.pred_thres,
                   total_budget=args.budget)

    rl = PPO(feat_dim=feat_dim,
            use_scene_attr=args.use_scene_attr,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            lmbda=args.lmbda,
            epsilon=args.epsilon,
            eps_clip=args.eps_clip,
            K_epoch=args.K_epoch)
    rl.cuda()

    if args.rl_path:
        rl.load_state_dict(torch.load(args.rl_path)['state_dict'])
        print(f'load {args.rl_path}')

    total_idx = 0
    start_idx = 0
    for iteration in range(loop):
        prior_network.eval()
        pred_network.eval()
        scene_dir = os.path.join('interact_scenes', args.output_dir, f'inter_{iteration}')
        env.set_scene_dir(scene_dir)

        while env.total_step < (iteration+1) * env.total_budget / loop:
            explore(env, rl)
            total_idx += 1
            print(f'scene {total_idx} collected')

        prior_network.train()
        tp_sum, tp_fp_sum, tp_fn_sum = 0, 0, 0
        for idx in range(len(os.listdir(scene_dir)) * 8):
            train_dataset = BinaryPriorDataset(scene_dir, pc_base_path=pc_base_path, idx=idx)
            if train_dataset.__len__() == 0:
                continue
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
            tp, tp_fp, tp_fn = train_binary_prior(train_loader, prior_network, prior_optim, scheduler, idx, summary_writer, use_cuda, print_freq)
            precision = (1e-8 + tp) / (1e-8 + tp_fp)
            recall = (1e-8 + tp) / (1e-8 + tp_fn)
            binary_f1 = (2 * precision * recall) / (precision + recall)
        prior_network.eval()

        pred_network.train()
        for idx in range(len(os.listdir(scene_dir)) * 8):
            train_dataset = ScenePriorDataset(scene_dir, pc_base_path=pc_base_path, prior_network=prior_network, idx=idx)
            if train_dataset.__len__() == 0:
                continue
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
            scene_tp, scene_tp_fp, tp_fn, _, _ = train_scene_prior(train_loader, pred_network, pred_optim, scheduler, idx, summary_writer, use_cuda, args.prior_thres, args.pred_thres, print_freq, use_scene_attr=args.use_scene_attr)
            precision = (1e-8 + scene_tp) / (1e-8 + scene_tp_fp)
            recall = (1e-8 + scene_tp) / (1e-8 + tp_fn)
            scene_f1 = (2 * precision * recall) / (precision + recall)
            scene_f1 = scene_f1.item()
        pred_network.eval()

        state = {
            'state_dict': prior_network.state_dict(),
            'f1': binary_f1
        }
        filename = os.path.join(check_dir(save_dir), f'prior_network{iteration}.tar')
        print(f'=> saving checkpoint to {filename}')
        torch.save(state, filename)

        state = {
            'state_dict': pred_network.state_dict(),
            'f1': scene_f1
        }
        filename = os.path.join(check_dir(save_dir), f'pred_network{iteration}.tar')
        print(f'=> saving checkpoint to {filename}')
        torch.save(state, filename)

        state = {
            'state_dict': rl.state_dict(),
        }

        filename = os.path.join(check_dir(save_dir), f'rl{iteration}.tar')
        print(f'=> saving checkpoint to {filename}')
        torch.save(state, filename)

        start_idx = total_idx
        if env.total_step >= env.total_budget:
            break
