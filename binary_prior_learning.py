import os
import time
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from gym_IFR.envs.relation import *

import torch
from torch.utils.tensorboard import SummaryWriter

from dataset import BinaryPriorDataset
from util import AverageMeter, ProgressMeter, check_dir, render_pc, set_index
from model.binary_relation_net import BinaryRelationNetwork

batch_time = AverageMeter('Time', ':6.3f')
data_time = AverageMeter('Data', ':6.3f')
losses = AverageMeter('Loss', ':.4e')
acc = AverageMeter('Acc', ':6.2f')
precision_meter = AverageMeter('Precision', ':6.2f')
recall_meter = AverageMeter('Recall', ':6.2f')


def train_binary_prior(train_loader, model, optimizer, scheduler, epoch, summary_writer, use_cuda, print_freq=10):
    progress = ProgressMeter(
        len(train_loader),
        # [batch_time, data_time, losses, acc, precision_meter, recall_meter, iou_src, iou_dst],
        [precision_meter, recall_meter],
        prefix="Epoch(Train): [{}]".format(epoch))

    # switch to train mode
    model.train()

    tp_sum, tp_fp_sum, tp_fn_sum = 0, 0, 0
    end = time.time()
    for i, (src_pc, dst_pc, src_pos, dst_pos, src_rot, dst_rot, src_size, dst_size, src_attr, dst_attr, gt_relation) in enumerate(train_loader):
        if use_cuda:
            src_pc = src_pc.float().cuda()
            dst_pc = dst_pc.float().cuda()
            src_pos = src_pos.float().cuda()
            dst_pos = dst_pos.float().cuda()
            src_rot = src_rot.float().cuda()
            dst_rot = dst_rot.float().cuda()
            src_size = src_size.float().cuda()
            dst_size = dst_size.float().cuda()
            src_attr = src_attr.float().cuda()
            dst_attr = dst_attr.float().cuda()
            gt_relation = gt_relation.cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        # forward through the network
        relation = model(src_pc, dst_pc, src_pos, dst_pos, src_rot, dst_rot, src_size, dst_size, src_attr, dst_attr)  # B x N x 3, B x P

        # for each type of loss, compute losses per data
        loss, accuracy, tp, tp_fp, tp_fn = model.get_loss(relation, gt_relation)
        tp_sum += tp
        tp_fp_sum += tp_fp
        tp_fn_sum += tp_fn

        # measure accuracy and record loss
        losses.update(loss.item(), src_pc.shape[0])
        acc.update(accuracy.item(), src_pc.shape[0])
        precision_meter.update(tp / (tp_fp + 1e-8), tp_fp)
        recall_meter.update(tp / (tp_fn + 1e-8), tp_fn)

        step = epoch * len(train_loader) + i

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step(step)

        # log
        if summary_writer is not None:
            summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step)
            summary_writer.add_scalar('train_acc', accuracy, step)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

    return tp_sum, tp_fp_sum, tp_fn_sum


def validate(val_loader, model, epoch, summary_writer, use_cuda, print_freq=10, thres=0.2, viz_base_path=None):
    progress = ProgressMeter(
        len(val_loader),
        # [batch_time, data_time, losses, acc, precision_meter, recall_meter, iou_src, iou_dst],
        [precision_meter, recall_meter],
        prefix="Epoch(Val):   [{}]".format(epoch))

    # switch to train mode
    model.eval()

    tp_sum, tp_fp_sum, tp_fn_sum = 0, 0, 0
    pred = []
    gt = []
    with torch.no_grad():
        end = time.time()
        for i, (src_pc, dst_pc, src_pos, dst_pos, src_rot, dst_rot, src_size, dst_size, src_attr, dst_attr, gt_relation) in enumerate(val_loader):
            if use_cuda:
                src_pc = src_pc.float().cuda()
                dst_pc = dst_pc.float().cuda()
                src_pos = src_pos.float().cuda()
                dst_pos = dst_pos.float().cuda()
                src_rot = src_rot.float().cuda()
                dst_rot = dst_rot.float().cuda()
                src_size = src_size.float().cuda()
                dst_size = dst_size.float().cuda()
                src_attr = src_attr.float().cuda()
                dst_attr = dst_attr.float().cuda()
                gt_relation = gt_relation.cuda()

            # measure data loading time
            data_time.update(time.time() - end)

            # forward through the network
            relation = model(src_pc, dst_pc, src_pos, dst_pos, src_rot, dst_rot, src_size, dst_size, src_attr, dst_attr)  # B x N x 3, B x P
            gt.append(gt_relation)
            pred.append(relation)

            # for each type of loss, compute losses per data
            loss, accuracy, tp, tp_fp, tp_fn = model.get_loss(relation, gt_relation)
            tp_sum += tp
            tp_fp_sum += tp_fp
            tp_fn_sum += tp_fn

            # measure accuracy and record loss
            losses.update(loss.item(), src_pc.shape[0])
            acc.update(accuracy.item(), src_pc.shape[0])
            precision_meter.update(tp / (tp_fp + 1e-8), tp_fp)
            recall_meter.update(tp / (tp_fn + 1e-8), tp_fn)

            step = epoch * len(val_loader) + i

            if summary_writer is not None:
                summary_writer.add_scalar('val_acc', accuracy, step)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

    if viz_base_path is not None:
        viz_path = os.path.join(viz_base_path, val_loader.dataset.dirs[val_loader.dataset.idx])
        if not os.path.exists(viz_path):
            os.mkdir(viz_path)

        pred = torch.stack(pred).float().sigmoid()
        gt = torch.stack(gt)
        # fig = plt.figure(figsize=(2, 1 * pred.shape[0]))
        # gs = mpl.gridspec.GridSpec(1, pred.shape[0], width_ratios=tuple([1] * pred.shape[0]))
        # fig = plt.figure(figsize=(10 * pred.shape[0], 10))
        # gs = mpl.gridspec.GridSpec(pred.shape[0], 1, height_ratios=tuple([1] * pred.shape[0]))
        # gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

        obj_indices = torch.where(dst_idx > 0)[1].cpu().numpy()

        dst_pos = dst_pos.cpu().numpy()
        dst_pc = dst_pc.cpu().numpy()

        for i in range(dst_pc.shape[0]):
            # ax = plt.subplot(gs[i], projection='3d')
            # ax.get_xaxis().set_ticks([])
            # ax.get_yaxis().set_ticks([])
            # ax.get_zaxis().set_ticks([])
            # ax.set_title(InDoorObject.__subclasses__()[obj_indices[i]].__name__ +
            #             '\n' + str(dst_pos[i][0].cpu().numpy()) +
            #             '\n' + str(dst_pos[i][1].cpu().numpy()) +
            #             '\n' + str(dst_pos[i][2].cpu().numpy())
            #             , pad=5)
            # render_pc(ax, dst_pc[i].cpu().numpy(), np.ones(dst_pc[i].shape[0], dtype=bool))

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # ax = plt.subplot(gs[i], projection='3d')
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.get_zaxis().set_ticks([])
            ax.set_title(f"{i}: {InDoorObject.__subclasses__()[obj_indices[i]].__name__}\n{dst_pos[i][0]:.3f}\n{dst_pos[i][1]:.3f}\n{dst_pos[i][2]:.3f}", pad=5)
            render_pc(ax, dst_pc[i], np.ones(dst_pc[i].shape[0], dtype=bool))
            fig.savefig(f'{viz_path}/obj_{i}.png', bbox_inches='tight')
            plt.close()

            if i == 0:
                with open(f'{viz_path}/obj.txt', 'w') as f:
                    f.write(f"{i}: {InDoorObject.__subclasses__()[obj_indices[i]].__name__}\t{dst_pos[i][0]:.3f} {dst_pos[i][1]:.3f} {dst_pos[i][2]:.3f}\n")
            else:
                with open(f'{viz_path}/obj.txt', 'a') as f:
                    f.write(f"{i}: {InDoorObject.__subclasses__()[obj_indices[i]].__name__}\t{dst_pos[i][0]:.3f} {dst_pos[i][1]:.3f} {dst_pos[i][2]:.3f}\n")

        # plt.tight_layout()
        # fig.savefig(f'{viz_path}/obj.png', bbox_inches='tight')
        # plt.close(fig)
        np.savetxt(f'{viz_path}/pred.txt', set_index(pred.cpu().numpy()), '%.3f')
        np.savetxt(f'{viz_path}/pred_oh.txt', set_index((pred.cpu().numpy() > thres).astype('int')), '%2i')
        np.savetxt(f'{viz_path}/gt.txt', set_index(gt.cpu().numpy().astype('int')), '%2i')

    # return acc.avg
    return tp_sum, tp_fp_sum, tp_fn_sum, pred


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Binary Prior')
    parser.add_argument('--bs', type=int, default=16, help='batch size in testing [default: 16]')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0001)
    parser.add_argument('--pos_weight', type=float, default=1.)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--prior_thres', type=float, default=0.2)
    parser.add_argument('--use_cuda', action='store_true', default=False)
    parser.add_argument('--use_scene_attr', action='store_true', default=False)
    parser.add_argument('--validate', action='store_true', default=False)
    parser.add_argument('--data_path', type=str, default="random_scenes/kitchen_budget_2000")
    parser.add_argument('--init_pn', type=str, default='pretrained_pn/net_network.pth')
    parser.add_argument('--resume_path', type=str, default="")
    parser.add_argument('--pc_base_path', type=str, default="data/pc_data")
    parser.add_argument('--checkpoint', type=str, default="")
    parser.add_argument('--viz_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    base_dir = os.path.join("output/binary_prior/", args.output_dir)
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

    model = BinaryRelationNetwork(use_scene_attr=args.use_scene_attr, thres=args.prior_thres)
    state_dict = torch.load(args.init_pn)
    # state_dict_v2 = copy.deepcopy(state_dict)
    # for key in state_dict:
    #     if 'encoder.fc_layer2' in key:
    #         state_dict_v2.pop(key)
    model.load_state_dict(state_dict, strict=False)

    if os.path.exists(args.resume_path):
        model.load_state_dict(torch.load(args.resume_path)['state_dict'])
    if args.use_cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.wd
    )

    if args.viz_path is not None:
        viz_base_path = os.path.join('binary_prior_viz', args.viz_path)
        if not os.path.exists(viz_base_path):
            os.mkdir(viz_base_path)
    else:
        viz_base_path = None

    if args.validate:
        tp_total, tp_fp_total, tp_fn_total = 0, 0, 0
        # thres_lst = np.arange(0.1, 1, 0.1)
        # precision_lst = []
        # recall_lst = []
        # f1_lst = []
        # for thres in thres_lst:
        #     model.thres = thres
        for epoch in range(args.epoch):
            val_dataset = BinaryPriorDataset(args.data_path, pc_base_path=args.pc_base_path, idx=epoch)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset.obj_indices), shuffle=False, num_workers=0, drop_last=False)
            tp, tp_fp, tp_fn, pred = validate(val_loader, model, epoch, summary_writer, args.use_cuda, args.print_freq, args.prior_thres, viz_base_path)
            tp_total, tp_fp_total, tp_fn_total = tp_total + tp, tp_fp_total + tp_fp, tp_fn_total + tp_fn
        precision = tp_total / tp_fp_total
        recall = tp_total / tp_fn_total
        f1 = (2 * precision * recall) / (precision + recall)
        print(f'precision: {precision}')
        print(f'recall: {recall}')
        print(f'f1: {f1}')
            # precision_lst.append(precision.item())
            # recall_lst.append(recall.item())
            # f1_lst.append(f1.item())
        # print(thres_lst)
        # print(precision_lst)
        # print(recall_lst)
        # print(f1_lst)
        return

    best_f1 = 0
    for epoch in range(args.epoch):
        train_dataset = BinaryPriorDataset(args.data_path, pc_base_path=args.pc_base_path, idx=epoch)
        if train_dataset.__len__() < 1:
            continue
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

        tp_sum, tp_fp_sum, tp_fn_sum = train_binary_prior(train_loader, model, optimizer, scheduler, epoch, summary_writer, args.use_cuda, args.print_freq)
        precision = tp_sum / (tp_fp_sum + 1e-8)
        recall = tp_sum / (tp_fn_sum + 1e-8)
        f1 = (2 * precision * recall) / (precision + recall + 1e-8)

        # val_acc = validate(val_loader, model, epoch, summary_writer, args.use_cuda, args.print_freq)

        if f1 >= best_f1:
            best_f1 = f1
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': best_f1,
            }

            filename = os.path.join(check_dir(save_dir), 'best_model.tar')
            logger.info(f'=> saving checkpoint to {filename}')
            torch.save(state, filename)

    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'acc': acc,
    }
    filename = os.path.join(check_dir(save_dir), 'last_model.tar')
    logger.info(f'=> saving checkpoint to {filename}')
    torch.save(state, filename)


if __name__ == '__main__':
    main()
