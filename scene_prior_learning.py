import torch
import trimesh

from PIL import Image

from torch.utils.tensorboard import SummaryWriter

import os
import time
import json
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from gym_IFR.envs.relation import *
from viz.scene_visual import render_relation_in_scene
from viz.mesh_render import mesh_to_img
from dataset import ScenePriorDataset
from util import AverageMeter, ProgressMeter, check_dir, set_index
from model.binary_relation_net import BinaryRelationNetwork
from model.GCN import GCNEncoder
from model.scene_relation_net import LinearLayerDecoder, MyGAE


def train_scene_prior(train_loader, model, optimizer, scheduler, epoch, summary_writer, use_cuda, prior_thres, pred_thres, print_freq=10, use_scene_attr=False):
    p_learn_meter = AverageMeter('P(learn)', ':6.2f')
    p_prior_meter = AverageMeter('P(prior)', ':6.2f')
    r_learn_meter = AverageMeter('R(learn)', ':6.2f')
    r_prior_meter = AverageMeter('R(prior)', ':6.2f')
    # acc_prior_meter = AverageMeter('ACC(prior)', ':6.2f')
    # acc_learn_meter = AverageMeter('ACC(learn)', ':6.2f')

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc_prior_meter = AverageMeter('ACC(prior)', ':6.2f')
    acc_learn_meter = AverageMeter('ACC(learn)', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        # [batch_time, data_time, losses, acc_prior_meter, acc_learn_meter, auc_meter, ap_meter],
        [p_learn_meter, p_prior_meter, r_learn_meter, r_prior_meter],
        prefix="Epoch(Train): [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    pred_tp_sum = 0
    pred_tp_fp_sum = 0
    tp_fn_sum = 0
    prior_tp_sum = 0
    prior_tp_fp_sum = 0
    for i, (pn_feat, pos, rot, size, attr, prior_graph, know_graph, gt_graph) in enumerate(train_loader):
        if use_cuda:
            pn_feat = pn_feat.float().cuda()
            pos = pos.float().cuda()
            rot = rot.float().cuda()
            size = size.float().cuda()
            attr = attr.float().cuda()
            prior_graph = prior_graph.float().cuda()
            know_graph = know_graph.float().cuda()
            gt_graph = gt_graph.float().cuda()

        scene_attr = torch.cat([pos, rot, size, attr], -1)
        if use_scene_attr:
            scene_attr_embedding = model.scene_attr_net(scene_attr)
            # obj_feats = torch.cat([pn_feat, onehot_idx, scene_attr_embedding], -1)
            obj_feats = torch.cat([pn_feat, scene_attr_embedding], -1)
        else:
            # obj_feats = torch.cat([pn_feat, onehot_idx], -1)
            obj_feats = pn_feat

        # measure data loading time
        data_time.update(time.time() - end)

        # forward through the network
        final_graph = model.build_gcn_graph(pos, prior_graph, know_graph, prior_thres)
        z = model.encode(obj_feats, final_graph)

        # edge_index, edge_type = model.build_gcn_graph_discrete(onehot_idx, pos, prior_graph, know_graph, prior_thres)
        # z = model.encode(obj_feats, edge_index, edge_type)
        # final_graph = know_graph

        pos_edge_index = torch.stack(torch.where(gt_graph[0] == 1))
        neg_edge_index = torch.stack(torch.where(gt_graph[0] == 0))
        loss = model.recon_loss(z, scene_attr, final_graph, pos_edge_index, neg_edge_index)
        pred = model.decoder.forward_all(z, scene_attr, final_graph)
        prior_acc = ((prior_graph[gt_graph != -1] > prior_thres) == (gt_graph[gt_graph != -1])).float().mean()
        acc = ((pred[gt_graph.squeeze() != -1] > pred_thres) == (gt_graph[gt_graph != -1])).float().mean()
        prior_mask = (prior_graph[gt_graph != -1] > prior_thres).bool()
        gt_mask = gt_graph[gt_graph != -1].bool()
        pred_mask = (pred[gt_graph.squeeze() != -1] > pred_thres).bool()
        pred_tp = (pred_mask & gt_mask).float().sum()
        pred_tp_fp = pred_mask.float().sum()
        prior_tp = (prior_mask & gt_mask).float().sum()
        prior_tp_fp = prior_mask.float().sum()
        tp_fn = gt_mask.float().sum()

        pred_tp_sum += pred_tp
        pred_tp_fp_sum += pred_tp_fp
        prior_tp_sum += prior_tp
        prior_tp_fp_sum += prior_tp_fp
        tp_fn_sum += tp_fn

        p_learn_meter.update(pred_tp / (pred_tp_fp + 1e-8), pred_tp_fp)
        p_prior_meter.update(prior_tp / (prior_tp_fp + 1e-8), prior_tp_fp)
        r_learn_meter.update(pred_tp / (tp_fn + 1e-8), tp_fn)
        r_prior_meter.update(prior_tp / (tp_fn + 1e-8), tp_fn)

        step = epoch * len(train_loader) + i
        losses.update(loss.item(), obj_feats.shape[0])
        acc_prior_meter.update(prior_acc.item(), obj_feats.shape[0])
        acc_learn_meter.update(acc.item(), obj_feats.shape[0])
        # auc_meter.update(auc.item(), obj_feats.shape[0])
        # ap_meter.update(ap.item(), obj_feats.shape[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step(step)

        # log
        if summary_writer is not None:
            summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step)
            summary_writer.add_scalar('train_acc', acc, step)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

    return pred_tp_sum, pred_tp_fp_sum, tp_fn_sum, prior_tp_sum, prior_tp_fp_sum


batch_time = AverageMeter('Time', ':6.3f')
data_time = AverageMeter('Data', ':6.3f')
losses = AverageMeter('Loss', ':.4e')
p_learn_meter = AverageMeter('P(learn)', ':6.2f')
p_prior_meter = AverageMeter('P(prior)', ':6.2f')
r_learn_meter = AverageMeter('R(learn)', ':6.2f')
r_prior_meter = AverageMeter('R(prior)', ':6.2f')


def validate(val_loader, model, epoch, summary_writer, use_cuda, prior_thres, pred_thres, print_freq=10, viz_base_path=None, use_scene_attr=False):
    # acc_prior_meter = AverageMeter('ACC(prior)', ':6.2f')
    # acc_learn_meter = AverageMeter('ACC(learn)', ':6.2f')

    progress = ProgressMeter(
        len(val_loader),
        # [batch_time, data_time, losses, acc_prior_meter, acc_learn_meter, auc_meter, ap_meter],
        [p_learn_meter, p_prior_meter, r_learn_meter, r_prior_meter],
        prefix="Epoch(Val):   [{}]".format(epoch))

    if viz_base_path is not None:
        viz_path = os.path.join(viz_base_path, val_loader.dataset.dirs[val_loader.dataset.idx])
        if not os.path.exists(viz_path):
            os.mkdir(viz_path)

    # switch to train mode
    model.eval()
    end = time.time()
    pred_tp_sum = 0
    pred_tp_fp_sum = 0
    tp_fn_sum = 0
    prior_tp_sum = 0
    prior_tp_fp_sum = 0
    with torch.no_grad():
        for i, (pn_feat, onehot_idx, pos, rot, size, attr, prior_graph, know_graph, gt_graph) in enumerate(val_loader):
            if use_cuda:
                pn_feat = pn_feat.float().cuda()
                onehot_idx = onehot_idx.float().cuda()
                pos = pos.float().cuda()
                rot = rot.float().cuda()
                size = size.float().cuda()
                attr = attr.float().cuda()
                prior_graph = prior_graph.float().cuda()
                know_graph = know_graph.float().cuda()
                gt_graph = gt_graph.float().cuda()

            scene_attr = torch.cat([pos, rot, size, attr], -1)
            if use_scene_attr:
                scene_attr_embedding = model.scene_attr_net(scene_attr)
                obj_feats = torch.cat([pn_feat, onehot_idx, scene_attr_embedding], -1)
            else:
                obj_feats = torch.cat([pn_feat, onehot_idx], -1)

            # measure data loading time
            data_time.update(time.time() - end)

            # forward through the network
            final_graph = model.build_gcn_graph(onehot_idx, pos, prior_graph, know_graph, prior_thres)
            z = model.encode(obj_feats, final_graph)

            # edge_index, edge_type = model.build_gcn_graph_discrete(onehot_idx, pos, prior_graph, know_graph, prior_thres)
            # z = model.encode(obj_feats, edge_index, edge_type)
            # final_graph = know_graph

            pos_edge_index = torch.stack(torch.where(gt_graph[0] == 1))
            neg_edge_index = torch.stack(torch.where(gt_graph[0] == 0))
            loss = model.recon_loss(z, scene_attr, final_graph, pos_edge_index, neg_edge_index)
            pred = model.decoder.forward_all(z, scene_attr, final_graph)

            if viz_base_path is not None:
                known = know_graph.squeeze().cpu().numpy().astype('int')
                pred_oh = (pred.cpu().numpy() > pred_thres).astype('int')
                gt = gt_graph.squeeze().cpu().numpy().astype('int')
                np.savetxt(f'{viz_path}/know_{i}.txt', set_index(know_graph.squeeze().cpu().numpy().astype('int')), '%2i')
                np.savetxt(f'{viz_path}/scene_{i}.txt', set_index(pred.cpu().numpy()), '%2.3f')
                np.savetxt(f'{viz_path}/scene_oh_{i}.txt', set_index((pred.cpu().numpy() > pred_thres).astype('int')), '%2i')
                np.savetxt(f'{viz_path}/instance_{i}.txt', set_index(prior_graph.squeeze().cpu().numpy()), '%2.3f')
                np.savetxt(f'{viz_path}/instance_oh_{i}.txt', set_index((prior_graph.squeeze().cpu().numpy() > prior_thres).astype('int')), '%2i')

                obj_pos = val_loader.dataset.obj_pos
                scene_path = val_loader.dataset.scene
                with open(f'{scene_path}/frame_meta.json', 'r') as f:
                    frame_meta = json.load(f)
                frame = Image.open(f"{scene_path}/frame.png")
                render_relation_in_scene(frame_meta, frame, obj_pos, gt, pred_oh, known, f'{viz_path}/viz_{i}.png')

            # prior_acc = ((prior_graph[gt_graph != -1] > prior_thres) == gt_graph[gt_graph != -1]).float().mean()
            acc = ((pred[gt_graph.squeeze() != -1] > pred_thres) == gt_graph[gt_graph != -1]).float().mean()
            prior_mask = (prior_graph[gt_graph != -1] > prior_thres).bool()
            gt_mask = gt_graph[gt_graph != -1].bool()
            pred_mask = (pred[gt_graph.squeeze() != -1] > pred_thres).bool()
            pred_tp = (pred_mask & gt_mask).float().sum()
            pred_tp_fp = pred_mask.float().sum()
            prior_tp = (prior_mask & gt_mask).float().sum()
            prior_tp_fp = prior_mask.float().sum()
            tp_fn = gt_mask.float().sum()

            pred_tp_sum += pred_tp.item()
            pred_tp_fp_sum += pred_tp_fp.item()
            prior_tp_sum += prior_tp.item()
            prior_tp_fp_sum += prior_tp_fp.item()
            tp_fn_sum += tp_fn.item()

            # if i >= len(tp_list):
            #     tp_list.append(pred_tp)
            #     tp_fp_list.append(pred_tp_fp)
            #     tp_fn_list.append(tp_fn)
            # else:
            #     tp_list[i] = pred_tp
            #     tp_fp_list[i] = pred_tp_fp
            #     tp_fn_list[i] = tp_fn

            p_learn_meter.update(pred_tp / (pred_tp_fp + 1e-8), pred_tp_fp)
            p_prior_meter.update(prior_tp / (prior_tp_fp + 1e-8), prior_tp_fp)
            r_learn_meter.update(pred_tp / (tp_fn + 1e-8), tp_fn)
            r_prior_meter.update(prior_tp / (tp_fn + 1e-8), tp_fn)

            step = epoch * len(val_loader) + i
            losses.update(loss.item(), obj_feats.shape[0])
            # auc_meter.update(auc.item(), obj_feats.shape[0])
            # ap_meter.update(ap.item(), obj_feats.shape[0])

            # log
            if summary_writer is not None:
                summary_writer.add_scalar('val_acc', acc, step)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)


    if viz_base_path is not None:
        # pred = torch.stack(pred).float().sigmoid()
        # fig = plt.figure(figsize=(10 * pred.shape[0], 10))
        # gs = mpl.gridspec.GridSpec(pred.shape[0], 1, height_ratios=tuple([1] * pred.shape[0]))
        # fig = plt.figure(figsize=(10 * pred.shape[0], 10))
        # gs = mpl.gridspec.GridSpec(pred.shape[0], 1, height_ratios=tuple([1] * pred.shape[0]))
        # gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)
        np.savetxt(f'{viz_path}/obj_pos.txt', val_loader.dataset.obj_pos)
        np.savetxt(f'{viz_path}/obj_rot.txt', val_loader.dataset.obj_rot)
        np.savetxt(f'{viz_path}/obj_size.txt', val_loader.dataset.obj_size)
        np.savetxt(f'{viz_path}/gt.txt', set_index(gt_graph.squeeze().cpu().numpy().astype('int')), '%2i')

        for i, (obj_idx, pc_path, obj_pos, obj_rot, obj_size) in enumerate(zip(val_loader.dataset.obj_indices, val_loader.dataset.pc_path, val_loader.dataset.obj_pos, val_loader.dataset.obj_rot, val_loader.dataset.obj_size)):
            fig = plt.figure(figsize=(4, 4))
            inst_mesh = trimesh.load(os.path.join(f'mesh_data/{pc_path[:-4]}.obj'))
            img = mesh_to_img(inst_mesh)
            plt.imshow(img)
            ax = plt.gca()
            ax.set_title(f"{i}: {InDoorObject.__subclasses__()[obj_idx].__name__}\n{obj_pos[0]:.3f}\n{obj_pos[1]:.3f}\n{obj_pos[2]:.3f}", pad=5)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)

            fig.savefig(f'{viz_path}/obj_{i}.png', bbox_inches='tight')
            plt.close()
            if i == 0:
                with open(f'{viz_path}/obj.txt', 'w') as f:
                    f.write(f"{i}: {InDoorObject.__subclasses__()[obj_idx].__name__}\t{pc_path}\t{obj_pos[0]:.3f} {obj_pos[1]:.3f} {obj_pos[2]:.3f}\n")
            else:
                with open(f'{viz_path}/obj.txt', 'a') as f:
                    f.write(f"{i}: {InDoorObject.__subclasses__()[obj_idx].__name__}\t{pc_path}\t{obj_pos[0]:.3f} {obj_pos[1]:.3f} {obj_pos[2]:.3f}\n")

        # plt.tight_layout()
        plt.close(fig)

    return pred_tp_sum, pred_tp_fp_sum, tp_fn_sum, prior_tp_sum, prior_tp_fp_sum


def parse_args():
    '''parameters'''
    parser = argparse.ArgumentParser('Scene Prior')
    parser.add_argument('--bs', type=int, default=1, help='batch size in testing [default: 1]')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--use_cuda', action='store_true', default=False)
    parser.add_argument('--prior_thres', type=float, default=0.2)
    parser.add_argument('--pred_thres', type=float, default=0.95)
    parser.add_argument('--use_scene_attr', action='store_true', default=False)
    parser.add_argument('--validate', action='store_true', default=False)
    parser.add_argument('--data_path', type=str, default="random_scenes/kitchen_budget_2000")
    parser.add_argument('--init', type=str, default="prior")
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--binary_prior_path', type=str, default=None)
    parser.add_argument('--viz_path', type=str, default=None)
    parser.add_argument('--pc_base_path', type=str, default="data/pc_data")
    parser.add_argument('--pn_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default="baseline")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.join("output/scene_prior/", args.output_dir)
    log_dir = os.path.join(base_dir, "tensorboard")
    save_dir = os.path.join(base_dir, "checkpoints")

    scheduler = None
    summary_writer = SummaryWriter(log_dir)

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/result.log' % (base_dir))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    prior_net = BinaryRelationNetwork(use_scene_attr=False, thres=args.prior_thres)
    if args.binary_prior_path is not None:
        prior_net.load_state_dict(torch.load(args.binary_prior_path)['state_dict'])
    prior_net.cuda()
    prior_net.eval()

    if args.viz_path is not None:
        viz_base_path = os.path.join('scene_prior_viz', args.viz_path)
        if not os.path.exists(viz_base_path):
            os.mkdir(viz_base_path)
    else:
        viz_base_path = None

    if args.use_scene_attr:
        # feat_dim = OBJ_NUM + 64 + 64
        feat_dim = 64 + 64
    else:
        # feat_dim = OBJ_NUM + 64
        feat_dim = 64

    out_channels = 32
    scene_attr_channels = 0
    middle_channels = 32
    decoder = LinearLayerDecoder(out_channels + scene_attr_channels, middle_channels)
    model = MyGAE(GCNEncoder(feat_dim, out_channels), decoder=decoder, init=args.init)

    if args.resume_path is not None:
        state_dict = torch.load(args.resume_path)['state_dict']
        model.load_state_dict(state_dict)
    if args.use_cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.wd
    )

    if args.validate:
        pred_tp = 0
        pred_tp_fp = 0
        tp_fn = 0
        prior_tp = 0
        prior_tp_fp = 0
        for epoch in range(args.epoch):
            val_dataset = ScenePriorDataset(args.data_path, pc_base_path=args.pc_base_path, prior_network=prior_net, idx=epoch, validate=True)
            if val_dataset.__len__() == 0:
                continue
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=0, drop_last=False)
            pred_tp_sum, pred_tp_fp_sum, tp_fn_sum, prior_tp_sum, prior_tp_fp_sum = validate(val_loader, model, epoch, summary_writer, args.use_cuda, args.prior_thres, args.pred_thres, args.print_freq, viz_base_path, use_scene_attr=args.use_scene_attr)
            pred_tp += pred_tp_sum
            pred_tp_fp += pred_tp_fp_sum
            tp_fn += tp_fn_sum
            prior_tp += prior_tp_sum
            prior_tp_fp += prior_tp_fp_sum

        pred_precision = pred_tp / pred_tp_fp
        pred_recall = pred_tp / tp_fn
        pred_f1 = (2 * pred_precision * pred_recall) / (pred_precision + pred_recall)

        prior_precision = prior_tp / prior_tp_fp
        prior_recall = prior_tp / tp_fn
        prior_f1 = (2 * prior_precision * prior_recall) / (prior_precision + prior_recall)
        print(f'pred precision: {pred_precision}')
        print(f'pred recall: {pred_recall}')
        print(f'pred f1: {pred_f1}')
        print(f'prior precision: {prior_precision}')
        print(f'prior recall: {prior_recall}')
        print(f'prior f1: {prior_f1}')

        return

    best_f1 = 0
    for epoch in range(args.epoch):
        train_dataset = ScenePriorDataset(args.data_path, pc_base_path=args.pc_base_path, prior_network=prior_net, idx=epoch)
        if train_dataset.__len__() == 0:
            continue
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=0, drop_last=False)
        pred_tp, pred_tp_fp, tp_fn, prior_tp, prior_tp_fp = train_scene_prior(train_loader, model, optimizer, scheduler, epoch, summary_writer, args.use_cuda, args.prior_thres, args.pred_thres, args.print_freq, use_scene_attr=args.use_scene_attr)

        pred_precision = pred_tp / pred_tp_fp
        pred_recall = pred_tp / tp_fn
        pred_f1 = (2 * pred_precision * pred_recall) / (pred_precision + pred_recall)

        if pred_f1 >= best_f1:
            best_f1 = pred_f1

            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'f1': best_f1,
            }
            filename = os.path.join(check_dir(save_dir), 'best_model.tar')
            logger.info(f'=> saving checkpoint to {filename}')
            torch.save(state, filename)

    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'f1': best_f1,
    }
    filename = os.path.join(check_dir(save_dir), 'last_model.tar')
    logger.info(f'=> saving checkpoint to {filename}')
    torch.save(state, filename)


if __name__ == '__main__':
    main()
